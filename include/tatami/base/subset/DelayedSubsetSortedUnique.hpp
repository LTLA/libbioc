#ifndef TATAMI_DELAYED_SUBSET_SORTED_UNIQUE_HPP
#define TATAMI_DELAYED_SUBSET_SORTED_UNIQUE_HPP

#include "../Matrix.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <memory>

/**
 * @file DelayedSubsetSortedUnique.hpp
 *
 * @brief Delayed subsetting with sorted and unique row/column indices.
 */

namespace tatami {

/**
 * @brief Delayed subsetting of a matrix with sorted, unique indices.
 *
 * Implements delayed subsetting (i.e., slicing) on the rows or columns of a matrix, given a vector of sorted and unique indices.
 * This operation is "delayed" in that it is only evaluated on request, e.g., with `row()` or friends.
 *
 * @tparam margin_ Dimension along which the subsetting is to occur.
 * If 0, the subset is applied to the rows; if 1, the subset is applied to the columns.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 * @tparam IndexStorage_ Vector containing the subset indices.
 */
template<int margin_, typename Value_, typename Index_, class IndexStorage_>
class DelayedSubsetSortedUnique : public Matrix<Value_, Index_> {
    static constexpr bool storage_has_data = has_data<Index_, IndexStorage_>::value;
public:
    /**
     * @param p Pointer to the underlying (pre-subset) matrix.
     * @param idx Vector of 0-based indices to use for subsetting on the rows (if `margin_ = 0`) or columns (if `margin_ = 1`).
     * This should be sorted and unique.
     * @param check Whether to check `idx` for sorted and unique values.
     */
    DelayedSubsetSortedUnique(std::shared_ptr<const Matrix<Value_, Index_> > p, IndexStorage_ idx, bool check = true) : mat(std::move(p)) {
        if constexpr(storage_has_data) {
            indices = std::move(idx);
        } else {
            indices = std::vector<Index_>(idx.begin(), idx.end());
        }

        if (check) {
            for (size_t i = 1, end = indices.size(); i < end; ++i) {
                if (indices[i] <= indices[i-1]) {
                    throw std::runtime_error("indices should be unique and sorted");
                }
            }
        }

        size_t mapping_dim = margin_ == 0 ? mat->nrow() : mat->ncol();
        mapping_single.resize(mapping_dim);
        for (Index_ i = 0, end = indices.size(); i < end; ++i) {
            mapping_single[indices[i]] = i;
        }
    }

private:
    std::shared_ptr<const Matrix<Value_, Index_> > mat;
    typename std::conditional<storage_has_data, IndexStorage_, std::vector<Index_> >::type indices;
    std::vector<Index_> mapping_single;

public:
    Index_ nrow() const {
        if constexpr(margin_==0) {
            return indices.size();
        } else {
            return mat->nrow();
        }
    }

    Index_ ncol() const {
        if constexpr(margin_==0) {
            return mat->ncol();
        } else {
            return indices.size();
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

    /*********************************************
     ************ Parallel extraction ************
     *********************************************/
private:
    template<DimensionSelectionType selection_, bool sparse_>
    struct ParallelWorkspaceBase : public Extractor<selection_, sparse_, Value_, Index_> {
        ParallelWorkspaceBase(const DelayedSubsetSortedUnique* parent, const Options<Index_>& opt) {
            if constexpr(selection_ == DimensionSelectionType::FULL) {
                this->full_length = parent->indices.size();
                internal = new_extractor<margin_ != 0, sparse_>(parent->mat.get(), parent->indices.data(), parent->indices.size(), opt);
            }
        }

        ParallelWorkspaceBase(const DelayedSubsetSortedUnique* parent, const Options<Index_>& opt, Index_ bs, Index_ bl) {
            if constexpr(selection_ == DimensionSelectionType::BLOCK) {
                this->block_start = bs;
                this->block_length = bl;
                internal = new_extractor<margin_ != 0, sparse_>(parent->mat.get(), parent->indices.data() + bs, bl, opt);
            }
        }

        ParallelWorkspaceBase(const DelayedSubsetSortedUnique* parent, const Options<Index_>& opt, const Index_* is, size_t il) {
            if constexpr(selection_ == DimensionSelectionType::INDEX) {
                indices.reserve(il);
                for (size_t i = 0; i < il; ++i) {
                    indices.push_back(parent->indices[is[i]]);
                }
                internal = new_extractor<margin_ != 0, sparse_>(parent->mat.get(), indices.data(), il, opt);

                this->index_length = il;
                std::copy(is, is + il, indices.begin());
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
        std::unique_ptr<Extractor<DimensionSelectionType::INDEX, sparse_, Value_, Index_> > internal;
        typename std::conditional<selection_ == DimensionSelectionType::INDEX, std::vector<Index_>, bool>::type indices;
    };

    template<DimensionSelectionType selection_>
    struct DenseParallelWorkspace : public ParallelWorkspaceBase<selection_, false> {
        template<typename ... Args_>
        DenseParallelWorkspace(const DelayedSubsetSortedUnique* parent, Args_... args) : ParallelWorkspaceBase<selection_, false>(parent, args...) {}

        const Value_* fetch(Index_ i, Value_* buffer) {
            return this->internal->fetch(i, buffer);
        }
    };

    template<DimensionSelectionType selection_>
    struct SparseParallelWorkspace : public ParallelWorkspaceBase<selection_, true> {
        template<typename ... Args_>
        SparseParallelWorkspace(const DelayedSubsetSortedUnique* p, Args_... args) : ParallelWorkspaceBase<selection_, true>(parent, args...), parent(p) {}

        SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
            auto raw = this->internal->fetch(i, vbuffer, ibuffer);
            if (raw.index) {
                auto icopy = ibuffer;
                for (size_t i = 0; i < raw.number; ++i, ++icopy) {
                    *icopy = parent->mapping_single[raw.index[i]];
                }
                raw.index = ibuffer;
            } else {
                raw.index = NULL;
            }
            return raw;
        }

    protected:
        const DelayedSubsetSortedUnique* parent;
    };

    /**************************************************
     ************ Perpendicular extraction ************
     **************************************************/
private:
    template<DimensionSelectionType selection_, bool sparse_>
    struct PerpendicularWorkspaceBase : public Extractor<selection_, sparse_, Value_, Index_> {
        PerpendicularWorkspaceBase(const DelayedSubsetSortedUnique* p, const Options<Index_>& opt) : parent(p) {
            if constexpr(selection_ == DimensionSelectionType::FULL) {
                this->full_length = parent->indices.size();
                internal = new_extractor<margin_ == 0, sparse_>(parent->mat.get(), opt);
            }
        }

        PerpendicularWorkspaceBase(const DelayedSubsetSortedUnique* p, const Options<Index_>& opt, Index_ bs, Index_ bl) : parent(p) {
            if constexpr(selection_ == DimensionSelectionType::BLOCK) {
                this->block_start = bs;
                this->block_length = bl;
                internal = new_extractor<margin_ == 0, sparse_>(parent->mat.get(), bs, bl, opt);
            }
        }

        PerpendicularWorkspaceBase(const DelayedSubsetSortedUnique* p, const Options<Index_>& opt, const Index_* is, size_t il) : parent(p) {
            if constexpr(selection_ == DimensionSelectionType::INDEX) {
                internal = new_extractor<margin_ == 0, sparse_>(parent->mat.get(), indices.data(), il, opt);
                this->index_length = il;
            }
        }

        const Index_* index_start() const {
            if constexpr(selection_ == DimensionSelectionType::INDEX) {
                return internal->index_start();
            } else {
                return NULL;
            }
        }

    protected:
        const DelayedSubsetSortedUnique* parent;
        std::unique_ptr<Extractor<selection_, sparse_, Value_, Index_> > internal;
    };

    template<DimensionSelectionType selection_>
    struct DensePerpendicularWorkspace : public PerpendicularWorkspaceBase<selection_, false> {
        template<typename ... Args_>
        DensePerpendicularWorkspace(const DelayedSubsetSortedUnique* parent, Args_... args) : PerpendicularWorkspaceBase<selection_, false>(parent, args...) {}

        const Value_* fetch(Index_ i, Value_* buffer) {
            return this->internal->fetch(this->parent->indices[i], buffer);
        }
    };

    template<DimensionSelectionType selection_>
    struct SparsePerpendicularWorkspace : public PerpendicularWorkspaceBase<selection_, true> {
        template<typename ... Args_>
        SparsePerpendicularWorkspace(const DelayedSubsetSortedUnique* parent, Args_... args) : PerpendicularWorkspaceBase<selection_, true>(parent, args...) {}

        SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
            return this->internal->fetch(this->parent->indices[i], vbuffer, ibuffer);
        }
    };

    /******************************************
     ************ Public overrides ************
     ******************************************/
private:
    template<bool accrow_, DimensionSelectionType selection_, bool sparse_, typename ... Args_>
    std::unique_ptr<Extractor<selection_, sparse_, Value_, Index_> > populate(const Options<Index_>& opt, Args_... args) const {
        std::unique_ptr<Extractor<selection_, sparse_, Value_, Index_> > output;

        if constexpr(accrow_ == (margin_ == 0)) {
            // TODO: fiddle with the access limits in 'opt'.
            if constexpr(sparse_) {
                output.reset(new SparsePerpendicularWorkspace(this, opt, args...));
            } else {
                output.reset(new DensePerpendicularWorkspace(this, opt, args...));
            }
        } else {
            if constexpr(sparse_) {
                output.reset(new SparseParallelWorkspace(this, opt, args...));
            } else {
                output.reset(new DenseParallelWorkspace(this, opt, args...));
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
    std::unique_ptr<FullDenseExtractor<Value_, Index_> > sparse_row(const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::FULL, true>(opt);
    }

    std::unique_ptr<BlockDenseExtractor<Value_, Index_> > sparse_row(Index_ block_start, Index_ block_length, const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::BLOCK, true>(opt, block_start, block_length);
    }

    std::unique_ptr<IndexDenseExtractor<Value_, Index_> > sparse_row(const Index_* index_start, size_t index_length, const Options<Index_>& opt) const {
        return populate<true, DimensionSelectionType::INDEX, true>(opt, index_start, index_length);
    }

    std::unique_ptr<FullDenseExtractor<Value_, Index_> > sparse_column(const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::FULL, true>(opt);
    }

    std::unique_ptr<BlockDenseExtractor<Value_, Index_> > sparse_column(Index_ block_start, Index_ block_length, const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::BLOCK, true>(opt, block_start, block_length);
    }

    std::unique_ptr<IndexDenseExtractor<Value_, Index_> > sparse_column(const Index_* index_start, size_t index_length, const Options<Index_>& opt) const {
        return populate<false, DimensionSelectionType::INDEX, true>(opt, index_start, index_length);
    }
};

}

#endif
