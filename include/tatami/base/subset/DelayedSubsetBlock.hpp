#ifndef TATAMI_DELAYED_SUBSET_BLOCK
#define TATAMI_DELAYED_SUBSET_BLOCK

#include "../Matrix.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <memory>

/**
 * @file DelayedSubsetBlock.hpp
 *
 * @brief Delayed subsetting to a single contiguous block.
 *
 * This is a specialized implementation that is more efficient than the `tatami::DelayedSubset` class.
 */

namespace tatami {

/**
 * @brief Delayed subsetting to a contiguous block.
 *
 * Implements delayed subsetting (i.e., slicing) of a matrix to a single contiguous block of rows or columns.
 * This is a specialized implementation that is more efficient than the `tatami::DelayedSubset` class.
 * This operation is "delayed" in that it is only evaluated on request, e.g., with `row()` or friends.
 *
 * @tparam margin_ Dimension along which the subsetting is to occur.
 * If 0, the subset is applied to the rows; if 1, the subset is applied to the columns.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 */
template<int margin_, typename Value_, typename Index_>
class DelayedSubsetBlock : public Matrix<Value_, Index_> {
public:
    /**
     * @param p Pointer to the underlying (pre-subset) matrix.
     * @param f Index of the start of the block. This should be a row index if `margin_ = 0` and a column index otherwise.
     * @param l Index of the one-past-the-end of the block.
     */
    DelayedSubsetBlock(std::shared_ptr<const Matrix<Value_, Index_> > p, Index_ f, Index_ l) : mat(p), block_start(f), block_length(l - f) {}

    /**
     * @copydoc DelayedSubsetBlock
     */
    DelayedSubsetBlock(std::shared_ptr<Matrix<Value_, Index_> > p, Index_ f, Index_ l) : mat(p), block_start(f), block_length(l - f) {}

private:
    std::shared_ptr<const Matrix<Value_, Index_> > mat;
    size_t block_start, block_length;

public:
    Index_ nrow() const {
        if constexpr(margin_==0) {
            return block_length;
        } else {
            return mat->nrow();
        }
    }

    Index_ ncol() const {
        if constexpr(margin_==0) {
            return mat->ncol();
        } else {
            return block_length;
        }
    }

    bool sparse() const {
        return mat->sparse();
    }

    bool prefer_rows() const {
        return mat->prefer_rows();
    }

    std::pair<double, double> dimension_preference() const {
        return mat->dimension_preference();
    }

    using Matrix<Value_, Index_>::dense_column;

    using Matrix<Value_, Index_>::dense_row;

    using Matrix<Value_, Index_>::sparse_column;

    using Matrix<Value_, Index_>::sparse_row;

    /**********************************
     ***** Along-subset extractor *****
     **********************************/
private:
    template<DimensionSelectionType selection_>
    static constexpr DimensionSelectionType define_inner_selection_type() {
        if constexpr(selection_ == DimensionSelectionType::FULL) {
            return DimensionSelectionType::BLOCK;
        } else {
            return selection_;
        }
    }

    template<DimensionSelectionType selection_, bool sparse_>
    struct AlongExtractor : public Extractor<selection_, sparse_, Value_, Index_> {
        AlongExtractor(const DelayedSubsetBlock* parent, const Options<Index_>& opt) {
            if constexpr(selection_ == DimensionSelectionType::FULL) {
                this->full_length = (margin_ == 0 ? parent->nrow() : parent->ncol());
                this->internal = new_extractor<margin_ != 0, sparse_>(parent->mat.get(), parent->block_start, parent->block_length, opt);
            }
        }

        AlongExtractor(const DelayedSubsetBlock* parent, const Options<Index_>& opt, Index_ bs, Index_ bl) {
            if constexpr(selection_ == DimensionSelectionType::BLOCK) {
                this->block_start = bs;
                this->block_length = bl;
                this->internal = new_extractor<margin_ != 0, sparse_>(parent->mat.get(), bs + parent->block_start, bl, opt);
            }
        }

        AlongExtractor(const DelayedSubsetBlock* parent, const Options<Index_>& opt, const Index_* is, size_t il) {
            if constexpr(selection_ == DimensionSelectionType::INDEX) {
                indices = std::vector<Index_>(is, is + il);

                // Shifting the block for the underlying matrix.
                for (auto& x : indices) {
                    x += parent->block_start;
                }
                this->internal = new_extractor<margin_ != 0, sparse_>(parent->mat.get(), indices.data(), il, opt);

                // And then shifting it back for index_start() reporting purposes.
                for (auto& x : indices) {
                    x -= parent->block_start;
                }
                this->index_length = il;
            }
        }

        const Index_* index_start() const {
            if constexpr(selection_ == DimensionSelectionType::INDEX) {
                return indices.data();
            } else {
                return NULL;
            }
        }

    protected:
        std::unique_ptr<Extractor<define_inner_selection_type<selection_>(), sparse_, Value_, Index_> > internal;
        typename std::conditional<selection_ == DimensionSelectionType::INDEX, std::vector<Index_>, bool>::type indices;
    };

    template<DimensionSelectionType selection_>
    struct DenseAlongExtractor : public AlongExtractor<selection_, false> {
        template<typename ...Args_>
        DenseAlongExtractor(const DelayedSubsetBlock* p, Args_... args) : AlongExtractor<selection_, false>(p, args...) {}

        const Value_* fetch(Index_ i, Value_* buffer) {
            return this->internal->fetch(i, buffer);
        }
    };

    template<DimensionSelectionType selection_>
    struct SparseAlongExtractor : public AlongExtractor<selection_, true> {
        template<typename ...Args_>
        SparseAlongExtractor(const DelayedSubsetBlock* p, Args_... args) : AlongExtractor<selection_, true>(p, args...), offset(p->block_start) {}

        SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
            auto out = this->internal->fetch(i, vbuffer, ibuffer);
            if (out.index) {
                if (offset) {
                    for (Index_ j = 0; j < out.number; ++j) {
                        ibuffer[j] = out.index[j] - offset;
                    }
                    out.index = ibuffer;
                }
            } else {
                out.index = NULL;
            }
            return out;
        }
    protected:
        Index_ offset;
    };

    /***********************************
     ***** Across-subset extractor *****
     ***********************************/
private:
    template<DimensionSelectionType selection_, bool sparse_>
    struct AcrossExtractor : public Extractor<selection_, sparse_, Value_, Index_> {
        AcrossExtractor(std::unique_ptr<Extractor<selection_, sparse_, Value_, Index_> > i, Index_ start) : internal(std::move(i)), offset(start) {
            if constexpr(selection_ == DimensionSelectionType::FULL) {
                this->full_length = this->internal->full_length;
            } else if constexpr(selection_ == DimensionSelectionType::BLOCK) {
                this->block_start = this->internal->block_start;
                this->block_length = this->internal->block_length;
            } else if constexpr(selection_ == DimensionSelectionType::INDEX) {
                this->index_length= this->internal->index_length;
            }
        }

        const Index_* index_start() const {
            return internal->index_start();
        }
    protected:
        std::unique_ptr<Extractor<selection_, sparse_, Value_, Index_> > internal;
        Index_ offset;
    };

    template<DimensionSelectionType selection_>
    struct DenseAcrossExtractor : public AcrossExtractor<selection_, false> {
        DenseAcrossExtractor(std::unique_ptr<Extractor<selection_, false, Value_, Index_> > i, Index_ start) : AcrossExtractor<selection_, false>(std::move(i), start) {}

        const Value_* fetch(Index_ i, Value_* buffer) {
            return this->internal->fetch(i + this->offset, buffer);
        }
    };

    template<DimensionSelectionType selection_>
    struct SparseAcrossExtractor : public AcrossExtractor<selection_, true> {
        SparseAcrossExtractor(std::unique_ptr<Extractor<selection_, true, Value_, Index_> > i, Index_ start) : AcrossExtractor<selection_, true>(std::move(i), start) {}

        SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
            return this->internal->fetch(i + this->offset, vbuffer, ibuffer);
        }
    };

    /********************************
     ***** Populating extractor *****
     ********************************/
private:
    template<bool accrow_, DimensionSelectionType selection_, bool sparse_, typename ... Args_>
    std::unique_ptr<Extractor<selection_, sparse_, Value_, Index_> > populate(const Options<Index_>& opt, Args_... args) const {
        std::unique_ptr<Extractor<selection_, sparse_, Value_, Index_> > output;

        if constexpr(accrow_ != (margin_ == 0)) {
            if constexpr(sparse_) {
                output.reset(new SparseAlongExtractor<selection_>(this, opt, args...));
            } else {
                output.reset(new DenseAlongExtractor<selection_>(this, opt, args...));
            }
        } else {
            auto ptr = new_extractor<accrow_, sparse_>(this->mat.get(), args..., opt);
            if constexpr(sparse_) {
                output.reset(new SparseAcrossExtractor<selection_>(std::move(ptr), this->block_start));
            } else {
                output.reset(new DenseAcrossExtractor<selection_>(std::move(ptr), this->block_start));
            }
        }

        return output;
    }

public:
    std::unique_ptr<FullDenseExtractor<Value_, Index_> > dense_row(const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::FULL, false>(opt);
    }

    std::unique_ptr<BlockDenseExtractor<Value_, Index_> > dense_row(Index_ block_start, Index_ block_length, const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::BLOCK, false>(opt, block_start, block_length);
    }

    std::unique_ptr<IndexDenseExtractor<Value_, Index_> > dense_row(const Index_* index_start, size_t index_length, const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::INDEX, false>(opt, index_start, index_length);
    }

    std::unique_ptr<FullDenseExtractor<Value_, Index_> > dense_column(const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::FULL, false>(opt);
    }

    std::unique_ptr<BlockDenseExtractor<Value_, Index_> > dense_column(Index_ block_start, Index_ block_length, const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::BLOCK, false>(opt, block_start, block_length);
    }

    std::unique_ptr<IndexDenseExtractor<Value_, Index_> > dense_column(const Index_* index_start, size_t index_length, const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::INDEX, false>(opt, index_start, index_length);
    }

public:
    std::unique_ptr<FullSparseExtractor<Value_, Index_> > sparse_row(const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::FULL, true>(opt);
    }

    std::unique_ptr<BlockSparseExtractor<Value_, Index_> > sparse_row(Index_ block_start, Index_ block_length, const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::BLOCK, true>(opt, block_start, block_length);
    }

    std::unique_ptr<IndexSparseExtractor<Value_, Index_> > sparse_row(const Index_* index_start, size_t index_length, const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::INDEX, true>(opt, index_start, index_length);
    }

    std::unique_ptr<FullSparseExtractor<Value_, Index_> > sparse_column(const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::FULL, true>(opt);
    }

    std::unique_ptr<BlockSparseExtractor<Value_, Index_> > sparse_column(Index_ block_start, Index_ block_length, const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::BLOCK, true>(opt, block_start, block_length);
    }

    std::unique_ptr<IndexSparseExtractor<Value_, Index_> > sparse_column(const Index_* index_start, size_t index_length, const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::INDEX, true>(opt, index_start, index_length);
    }
};

/**
 * A `make_*` helper function to enable partial template deduction of supplied types.
 *
 * @tparam margin_ Dimension along which the addition is to occur.
 * If 0, the subset is applied to the rows; if 1, the subset is applied to the columns.
 * @tparam MAT A specialized `Matrix`, to be automatically deducted.
 * @tparam V Vector containing the subset indices, to be automatically deducted.
 *
 * @param p Pointer to the underlying (pre-subset) `Matrix`.
 * @param f Index of the start of the block. This should be a row index if `margin_ = 0` and a column index otherwise.
 * @param l Index of the one-past-the-end of the block.
 *
 * @return A pointer to a `DelayedSubsetBlock` instance.
 */
template<int margin_, class MAT>
std::shared_ptr<MAT> make_DelayedSubsetBlock(std::shared_ptr<MAT> p, size_t f, size_t l) {
    return std::shared_ptr<MAT>(new DelayedSubsetBlock<margin_, typename MAT::value_type, typename MAT::index_type>(p, f, l));
}

}

#endif
