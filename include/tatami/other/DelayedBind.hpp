#ifndef TATAMI_DELAYED_BIND_HPP
#define TATAMI_DELAYED_BIND_HPP

#include "../base/Matrix.hpp"
#include "../utils/new_extractor.hpp"
#include "../utils/ConsecutiveOracle.hpp"
#include "../utils/FixedOracle.hpp"
#include "../utils/PseudoOracularExtractor.hpp"
#include "../utils/copy.hpp"

#include <numeric>
#include <algorithm>
#include <memory>
#include <array>
#include <type_traits>

/**
 * @file DelayedBind.hpp
 *
 * @brief Delayed combining of multiple `tatami::Matrix` objects.
 *
 * This is equivalent to the `DelayedAbind` class in the **DelayedArray** package.
 */

namespace tatami {

/**
 * @cond
 */
namespace DelayedBind_internal {

/**********************
 *** Dense parallel ***
 **********************/

template<typename Index_, class Initialize_>
Index_ initialize_parallel_block(
    const std::vector<Index_>& cumulative, 
    const std::vector<Index_>& mapping,
    Index_ block_start, 
    Index_ block_length, 
    Initialize_ init) 
{
    if (mapping.empty()) {
        return 0;
    }

    Index_ start_index = mapping[block_start];
    Index_ actual_start = block_start - cumulative[start_index];
    Index_ block_end = block_start + block_length;

    for (Index_ index = start_index, nmats = cumulative.size() - 1; index < nmats; ++index) {
        Index_ submat_end = cumulative[index + 1]; 
        bool not_final = (block_end > submat_end);
        Index_ actual_end = (not_final ? submat_end : block_end) - cumulative[index];
        init(index, actual_start, actual_end - actual_start);
        if (!not_final) {
            break;
        }
        actual_start = 0;
    }

    return start_index;
}

template<typename Index_, class Initialize_>
void initialize_parallel_index(
    const std::vector<Index_>& cumulative, 
    const std::vector<Index_>& mapping,
    const std::vector<Index_>& indices, 
    Initialize_ init) 
{
    Index_ counter = 0, il = indices.size();
    while (counter < il) {
        Index_ first_index = indices[counter];
        Index_ bind_index = mapping[first_index];
        Index_ lower = cumulative[bind_index];
        Index_ upper = cumulative[bind_index + 1];

        // Creating the slice with one element already.
        auto slice_ptr = std::make_shared<std::vector<Index_> >(1, first_index - lower);
        ++counter;

        while (counter < il && indices[counter]  < upper) {
            slice_ptr->push_back(indices[counter] - lower);
            ++counter;
        }

        init(bind_index, std::move(slice_ptr));
    }
}

template<bool oracle_, typename Value_, typename Index_>
class  ParallelDense : public DenseExtractor<oracle_, Value_, Index_> {
public:
    ParallelDense(
        const std::vector<Index_>&, // Not used, just provided for consistency with other constructors.
        const std::vector<Index_>&,
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        const Options& opt)
    {
        my_exts.reserve(matrices.size());
        my_count.reserve(matrices.size());
        for (const auto& m : matrices) {
            my_count.emplace_back(row ? m->ncol() : m->nrow());
            my_exts.emplace_back(new_extractor<false, oracle_>(m.get(), row, oracle, opt));
        }
    }

    ParallelDense(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>& mapping,
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start, 
        Index_ block_length, 
        const Options& opt)
    {
        my_exts.reserve(matrices.size());
        my_count.reserve(matrices.size());
        initialize_parallel_block(
            cumulative, 
            mapping,
            block_start, 
            block_length,
            [&](Index_ i, Index_ sub_block_start, Index_ sub_block_length) {
                my_count.emplace_back(sub_block_length);
                my_exts.emplace_back(new_extractor<false, oracle_>(matrices[i].get(), row, oracle, sub_block_start, sub_block_length, opt));
            }
        );
    }

    ParallelDense(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>& mapping,
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        VectorPtr<Index_> indices_ptr,
        const Options& opt)
    {
        my_exts.reserve(matrices.size());
        my_count.reserve(matrices.size());
        initialize_parallel_index(
            cumulative, 
            mapping,
            *indices_ptr,
            [&](Index_ i, VectorPtr<Index_> sub_indices_ptr) {
                my_count.emplace_back(sub_indices_ptr->size());
                my_exts.emplace_back(new_extractor<false, oracle_>(matrices[i].get(), row, oracle, std::move(sub_indices_ptr), opt));
            }
        );
    }

public:
    const Value_* fetch(Index_ i, Value_* buffer) {
        auto copy = buffer;
        for (Index_ x = 0, end = my_count.size(); x < end; ++x) {
            auto ptr = my_exts[x]->fetch(i, copy); 
            auto num = my_count[x];
            copy_n(ptr, num, copy);
            copy += num;
        }
        return buffer;
    }

private:
    std::vector<std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > > my_exts;
    std::vector<Index_> my_count;
};

/***********************
 *** Sparse parallel ***
 ***********************/

template<bool oracle_, typename Value_, typename Index_>
class ParallelFullSparse : public SparseExtractor<oracle_, Value_, Index_> {
public:
    ParallelFullSparse(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>&, // not actually used, just provided for consistency with the other constructors.
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        const Options& opt) : 
        my_cumulative(cumulative),
        my_needs_value(opt.sparse_extract_value),
        my_needs_index(opt.sparse_extract_index)
    {
        my_exts.reserve(matrices.size());
        for (const auto& m : matrices) {
            my_exts.emplace_back(new_extractor<true, oracle_>(m.get(), row, oracle, opt));
        }
    }

    SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto vcopy = value_buffer;
        auto icopy = index_buffer;
        Index_ accumulated = 0;

        for (Index_ x = 0, end = my_cumulative.size() - 1; x < end; ++x) {
            auto range = my_exts[x]->fetch(i, vcopy, icopy); 
            accumulated += range.number;
            if (my_needs_value) {
                copy_n(range.value, range.number, vcopy);
                vcopy += range.number;
            }
            if (my_needs_index) {
                auto offset = my_cumulative[x];
                for (Index_ y = 0; y < range.number; ++y) {
                    icopy[y] = range.index[y] + offset;
                }
                icopy += range.number;
            }
        }

        return SparseRange<Value_, Index_>(accumulated, (my_needs_value ? value_buffer : NULL), (my_needs_index ? index_buffer : NULL));
    }

private:
    const std::vector<Index_>& my_cumulative;
    bool my_needs_value, my_needs_index;
    std::vector<std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > > my_exts;
};

template<bool oracle_, typename Value_, typename Index_>
class ParallelBlockSparse : public SparseExtractor<oracle_, Value_, Index_> {
public:
    ParallelBlockSparse(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>& mapping,
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start, 
        Index_ block_length, 
        const Options& opt) : 
        my_cumulative(cumulative), 
        my_needs_value(opt.sparse_extract_value), 
        my_needs_index(opt.sparse_extract_index) 
    {
        my_exts.reserve(matrices.size());
        my_start_matrix = initialize_parallel_block(
            my_cumulative, 
            mapping,
            block_start, 
            block_length,
            [&](Index_ i, Index_ sub_block_start, Index_ sub_block_length) {
                my_exts.emplace_back(new_extractor<true, oracle_>(matrices[i].get(), row, oracle, sub_block_start, sub_block_length, opt));
            }
        );
    }

    SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto vcopy = value_buffer;
        auto icopy = index_buffer;
        Index_ count = 0;

        for (Index_ x = 0, end = my_exts.size(); x < end; ++x) {
            auto range = my_exts[x]->fetch(i, vcopy, icopy);
            count += range.number;
            if (my_needs_value) {
                copy_n(range.value, range.number, vcopy);
                vcopy += range.number;
            }
            if (my_needs_index) {
                Index_ offset = my_cumulative[x + my_start_matrix];
                for (Index_ y = 0; y < range.number; ++y) {
                    icopy[y] = range.index[y] + offset;
                }
                icopy += range.number;
            }
        }

        return SparseRange<Value_, Index_>(count, (my_needs_value ? value_buffer : NULL), (my_needs_index ? index_buffer : NULL));
    }

private:
    const std::vector<Index_>& my_cumulative;
    bool my_needs_value, my_needs_index;
    std::vector<std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > > my_exts;
    Index_ my_start_matrix;
};

template<bool oracle_, typename Value_, typename Index_>
class ParallelIndexSparse : public SparseExtractor<oracle_, Value_, Index_> {
public:
    ParallelIndexSparse(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>& mapping,
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        VectorPtr<Index_> indices_ptr,
        const Options& opt) : 
        my_cumulative(cumulative),
        my_needs_value(opt.sparse_extract_value), 
        my_needs_index(opt.sparse_extract_index) 
    {
        my_exts.reserve(matrices.size());
        my_which_matrix.reserve(matrices.size());
        initialize_parallel_index(
            my_cumulative, 
            mapping,
            *indices_ptr,
            [&](Index_ i, VectorPtr<Index_> sub_indices_ptr) {
                my_which_matrix.emplace_back(i);
                my_exts.emplace_back(new_extractor<true, oracle_>(matrices[i].get(), row, oracle, std::move(sub_indices_ptr), opt));
            }
        );
    }

    SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto vcopy = value_buffer;
        auto icopy = index_buffer;
        Index_ count = 0;

        for (Index_ x = 0, end = my_which_matrix.size(); x < end; ++x) {
            auto range = my_exts[x]->fetch(i, vcopy, icopy);
            count += range.number;
            if (my_needs_value) {
                copy_n(range.value, range.number, vcopy);
                vcopy += range.number;
            }

            if (my_needs_index) {
                Index_ offset = my_cumulative[my_which_matrix[x]];
                for (Index_ y = 0; y < range.number; ++y) {
                    icopy[y] = range.index[y] + offset;
                }
                icopy += range.number;
            }
        }

        return SparseRange<Value_, Index_>(count, (my_needs_value ? value_buffer : NULL), (my_needs_index ? index_buffer : NULL));
    }

private:
    const std::vector<Index_>& my_cumulative;
    bool my_needs_value, my_needs_index;
    std::vector<std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > > my_exts;
    std::vector<Index_> my_which_matrix;
};

/*********************
 *** Perpendicular ***
 *********************/

template<typename Value_, typename Index_>
class MyopicPerpendicularDense : public MyopicDenseExtractor<Value_, Index_> {
public:
    template<typename ... Args_>
    MyopicPerpendicularDense(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>& mapping, 
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row, 
        const Args_& ... args) : 
        my_cumulative(cumulative),
        my_mapping(mapping)
    {
        my_exts.reserve(matrices.size());
        for (const auto& m : matrices) {
            my_exts.emplace_back(m->dense(row, args...));
        }
    }

    const Value_* fetch(Index_ i, Value_* buffer) {
        Index_ chosen = my_mapping[i];
        return my_exts[chosen]->fetch(i - my_cumulative[chosen], buffer);
    }

private:
    const std::vector<Index_>& my_cumulative;
    const std::vector<Index_>& my_mapping;
    std::vector<std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > > my_exts;
};

template<typename Value_, typename Index_>
class MyopicPerpendicularSparse : public MyopicSparseExtractor<Value_, Index_> {
public:
    template<typename ... Args_>
    MyopicPerpendicularSparse(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>& mapping, 
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        const Args_& ... args) : 
        my_cumulative(cumulative),
        my_mapping(mapping)
    {
        my_exts.reserve(matrices.size());
        for (const auto& m : matrices) {
            my_exts.emplace_back(m->sparse(row, args...));
        }
    }

    SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        Index_ chosen = my_mapping[i];
        return my_exts[chosen]->fetch(i - my_cumulative[chosen], vbuffer, ibuffer);
    }

private:
    const std::vector<Index_>& my_cumulative;
    const std::vector<Index_>& my_mapping;
    std::vector<std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > > my_exts;
};

template<typename Index_, class Initialize_>
void initialize_perp_oracular(
    const std::vector<Index_>& cumulative, 
    const std::vector<Index_>& mapping, 
    const Oracle<Index_>* oracle, 
    std::vector<Index_>& chosen, 
    Initialize_ init) 
{
    size_t ntotal = oracle->total();
    chosen.reserve(ntotal);

    struct Predictions {
        bool consecutive = true;
        Index_ start = 0;
        Index_ number = 0;
        std::vector<Index_> predictions;

        void add(Index_ p) {
            if (consecutive) {
                if (number == 0) {
                    start = p;
                    number = 1;
                    return;
                }
                if (number + start == p) {
                    ++number;
                    return;
                }
                consecutive = false;
                predictions.resize(number);
                std::iota(predictions.begin(), predictions.end(), start);
            }

            predictions.push_back(p);
        }
    };

    Index_ nmats = cumulative.size() - 1;
    std::vector<Predictions> predictions(nmats);
    for (size_t i = 0; i < ntotal; ++i) {
        auto prediction = oracle->get(i);
        Index_ choice = mapping[prediction];
        chosen.push_back(choice);
        predictions[choice].add(prediction - cumulative[choice]);
    }

    for (Index_ x = 0; x < nmats; ++x) {
        auto& current = predictions[x];
        if (current.consecutive) {
            if (current.number) {
                init(x, std::make_shared<ConsecutiveOracle<Index_> >(current.start, current.number));
            }
        } else {
            if (!current.predictions.empty()) {
                init(x, std::make_shared<FixedVectorOracle<Index_> >(std::move(current.predictions)));
            }
        }
    }
}

template<typename Value_, typename Index_>
class OracularPerpendicularDense : public OracularDenseExtractor<Value_, Index_> {
public:
    template<typename ... Args_>
    OracularPerpendicularDense(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>& mapping, 
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        std::shared_ptr<const Oracle<Index_> > ora, 
        const Args_& ... args)
    {
        my_exts.resize(matrices.size());
        initialize_perp_oracular(
            cumulative,
            mapping,
            ora.get(),
            segments,
            [&](Index_ x, std::shared_ptr<const Oracle<Index_> > subora) {
                my_exts[x] = matrices[x]->dense(row, std::move(subora), args...);
            }
        );
    }

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto chosen = segments[used];
        auto output = my_exts[chosen]->fetch(i, buffer);
        ++used;
        return output;
    }

private:
    std::vector<Index_> segments;
    std::vector<std::unique_ptr<OracularDenseExtractor<Value_, Index_> > > my_exts;
    size_t used = 0;
};

template<typename Value_, typename Index_>
class OracularPerpendicularSparse : public OracularSparseExtractor<Value_, Index_> {
public:
    template<typename ... Args_>
    OracularPerpendicularSparse(
        const std::vector<Index_>& cumulative, 
        const std::vector<Index_>& mapping, 
        const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& matrices, 
        bool row,
        std::shared_ptr<const Oracle<Index_> > ora, 
        const Args_& ... args)
    {
        my_exts.resize(matrices.size());
        initialize_perp_oracular(
            cumulative,
            mapping,
            ora.get(),
            segments,
            [&](Index_ x, std::shared_ptr<const Oracle<Index_> > subora) {
                my_exts[x] = matrices[x]->sparse(row, std::move(subora), args...);
            }
        );
    }

    SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto chosen = segments[used];
        auto output = my_exts[chosen]->fetch(i, vbuffer, ibuffer);
        ++used;
        return output;
    }

private:
    std::vector<Index_> segments;
    std::vector<std::unique_ptr<OracularSparseExtractor<Value_, Index_> > > my_exts;
    size_t used = 0;
};

}
/**
 * @endcond
 */

/**
 * @brief Delayed combining of a matrix.
 *
 * Implements delayed combining by rows or columns of a matrix.
 * This operation is "delayed" in that it is only performed on a row-by-row or column-by-column basis during data extraction.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 */
template<typename Value_, typename Index_>
class DelayedBind : public Matrix<Value_, Index_> {
public:
    /**
     * @param matrices Pointers to the matrices to be combined.
     * All matrices to be combined should have the same number of columns (if `row = true`) or rows (otherwise).
     * @param by_row Whether to combine matrices by the rows (i.e., the output matrix has number of rows equal to the sum of the number of rows in `matrices`).
     * If false, combining is applied by the columns.
     */
    DelayedBind(std::vector<std::shared_ptr<const Matrix<Value_, Index_> > > matrices, bool by_row) : 
        my_matrices(std::move(matrices)), my_by_row(by_row), my_cumulative(my_matrices.size()+1) 
    {
        size_t sofar = 0;
        for (size_t i = 0, nmats = my_matrices.size(); i < nmats; ++i) {
            auto& current = my_matrices[i];
            Index_ primary, secondary;
            if (my_by_row) {
                primary = current->nrow();
                secondary = current->ncol();
            } else {
                primary = current->ncol();
                secondary = current->nrow();
            }

            if (i == 0) {
                my_otherdim = secondary;
            } else if (my_otherdim != secondary) {
                throw std::runtime_error("all 'my_matrices' should have the same number of " + (my_by_row ? std::string("columns") : std::string("rows")));
            }

            // Removing the matrices that don't contribute anything,
            // so we don't have to deal with their overhead.
            if (primary > 0) {
                if (sofar != i) {
                    my_matrices[sofar] = std::move(current);
                }
                my_cumulative[sofar + 1] = my_cumulative[sofar] + primary;
                ++sofar;
            }
        }

        my_cumulative.resize(sofar + 1);
        my_matrices.resize(sofar);

        // At this point, the number of matrices must be no greater than the
        // number of rows/columns of the combined matrix (as we've removed all
        // non-contributing submatrices) and thus should fit into 'Index_';
        // hence, using Index_ for the mapping should not overflow.
        my_mapping.reserve(my_cumulative.back());
        for (Index_ i = 0, nmats = my_matrices.size(); i < nmats; ++i) {
            my_mapping.insert(my_mapping.end(), (my_by_row ? my_matrices[i]->nrow() : my_matrices[i]->ncol()), i);
        }

        double denom = 0;
        for (const auto& x : my_matrices) {
            double total = static_cast<double>(x->nrow()) * static_cast<double>(x->ncol());
            denom += total;
            my_sparse_prop += total * x->is_sparse_proportion();
            my_by_row_prop += total * x->prefer_rows_proportion();
        }
        if (denom) {
            my_sparse_prop /= denom;
            my_by_row_prop /= denom;
        }

        for (int d = 0; d < 2; ++d) {
            my_uses_oracle[d] = false;
            for (const auto& x : my_matrices) {
                if (x->uses_oracle(d)) {
                    my_uses_oracle[d] = true;
                    break;
                }
            }
        }
    }

    /**
     * @param matrices Pointers to the matrices to be combined.
     * All matrices to be combined should have the same number of columns (if `row = true`) or rows (otherwise).
     * @param by_row Whether to combine matrices by the rows (i.e., the output matrix has number of rows equal to the sum of the number of rows in `matrices`).
     * If false, combining is applied by the columns.
     */
    DelayedBind(const std::vector<std::shared_ptr<Matrix<Value_, Index_> > >& matrices, bool by_row) : 
        DelayedBind(std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >(matrices.begin(), matrices.end()), by_row) {}

private:
    std::vector<std::shared_ptr<const Matrix<Value_, Index_> > > my_matrices;
    bool my_by_row;

    Index_ my_otherdim = 0;
    std::vector<Index_> my_cumulative;
    std::vector<Index_> my_mapping;

    double my_sparse_prop = 0, my_by_row_prop = 0;
    std::array<bool, 2> my_uses_oracle;

public:
    Index_ nrow() const {
        if (my_by_row) {
            return my_cumulative.back();
        } else {
            return my_otherdim;
        }
    }

    Index_ ncol() const {
        if (my_by_row) {
            return my_otherdim;
        } else {
            return my_cumulative.back();
        }
    }

    bool is_sparse() const {
        return my_sparse_prop > 0.5;
    }

    double is_sparse_proportion() const {
        return my_sparse_prop;
    }

    bool prefer_rows() const {
        return my_by_row_prop > 0.5;
    }

    double prefer_rows_proportion() const {
        return my_by_row_prop;
    }

    bool uses_oracle(bool row) const {
        return my_uses_oracle[row];
    }

    using Matrix<Value_, Index_>::dense;

    using Matrix<Value_, Index_>::sparse;

    /**
     * @cond
     */
public:
    const std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >& seeds() const {
        return my_matrices;
    }

    bool by_row() const {
        return my_by_row;
    }
    /**
     * @endcond
     */

    /**********************************
     ********** Myopic dense **********
     **********************************/
public:
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->dense(row, opt);
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::MyopicPerpendicularDense<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelDense<false, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, false, opt);
        }
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->dense(row, block_start, block_length, opt);
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::MyopicPerpendicularDense<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, block_start, block_length, opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelDense<false, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, false, block_start, block_length, opt);
        }
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->dense(row, std::move(indices_ptr), opt);
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::MyopicPerpendicularDense<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(indices_ptr), opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelDense<false, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, false, std::move(indices_ptr), opt);
        }
    }

    /***********************************
     ********** Myopic sparse **********
     ***********************************/
private:
    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->sparse(row, opt);
        } else  if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::MyopicPerpendicularSparse<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelFullSparse<false, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, false, opt);
        }
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->sparse(row, block_start, block_length, opt);
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::MyopicPerpendicularSparse<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, block_start, block_length, opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelBlockSparse<false, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, false, block_start, block_length, opt);
        }
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->sparse(row, std::move(indices_ptr), opt);
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::MyopicPerpendicularSparse<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(indices_ptr), opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelIndexSparse<false, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, false, std::move(indices_ptr), opt);
        }
    }

    /************************************
     ********** Oracular dense **********
     ************************************/
public:
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->dense(row, std::move(oracle), opt);
        } else if (!my_uses_oracle[row]) {
            return std::make_unique<PseudoOracularDenseExtractor<Value_, Index_> >(std::move(oracle), dense(row, opt));
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::OracularPerpendicularDense<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelDense<true, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), opt);
        }
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->dense(row, std::move(oracle), block_start, block_length, opt);
        } else if (!my_uses_oracle[row]) {
            return std::make_unique<PseudoOracularDenseExtractor<Value_, Index_> >(std::move(oracle), dense(row, block_start, block_length, opt));
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::OracularPerpendicularDense<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), block_start, block_length, opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelDense<true, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), block_start, block_length, opt);
        }
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->dense(row, std::move(oracle), std::move(indices_ptr), opt);
        } else if (!my_uses_oracle[row]) {
            return std::make_unique<PseudoOracularDenseExtractor<Value_, Index_> >(std::move(oracle), dense(row, std::move(indices_ptr), opt));
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::OracularPerpendicularDense<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), std::move(indices_ptr), opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelDense<true, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), std::move(indices_ptr), opt);
        }
    }

    /*************************************
     ********** Oracular sparse **********
     *************************************/
private:
    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->sparse(row, std::move(oracle), opt);
        } else if (!my_uses_oracle[row]) {
            return std::make_unique<PseudoOracularSparseExtractor<Value_, Index_> >(std::move(oracle), sparse(row, opt));
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::OracularPerpendicularSparse<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelFullSparse<true, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), opt);
        }
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->sparse(row, std::move(oracle), block_start, block_length, opt);
        } else if (!my_uses_oracle[row]) {
            return std::make_unique<PseudoOracularSparseExtractor<Value_, Index_> >(std::move(oracle), sparse(row, block_start, block_length, opt));
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::OracularPerpendicularSparse<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), block_start, block_length, opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelBlockSparse<true, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), block_start, block_length, opt);
        }
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        if (my_matrices.size() == 1) {
            return my_matrices[0]->sparse(row, std::move(oracle), std::move(indices_ptr), opt);
        } else if (!my_uses_oracle[row]) {
            return std::make_unique<PseudoOracularSparseExtractor<Value_, Index_> >(std::move(oracle), sparse(row, std::move(indices_ptr), opt));
        } else if (row == my_by_row) {
            return std::make_unique<DelayedBind_internal::OracularPerpendicularSparse<Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), std::move(indices_ptr), opt);
        } else {
            return std::make_unique<DelayedBind_internal::ParallelIndexSparse<true, Value_, Index_> >(my_cumulative, my_mapping, my_matrices, row, std::move(oracle), std::move(indices_ptr), opt);
        }
    }
};

/**
 * A `make_*` helper function for delayed combining of `Matrix` instances.
 * This will typically return a `DelayedBind` instance wrapping the provided `matrices`, with some exceptions:
 *
 * - If `matrices` contains only one matrix, it is returned directly, eliminating the need for an extra `DelayedBind` wrapper altogether.
 * - If any entry of `matrices` is a `DelayedBind` using the same `row`, its component matrices are merged into the current `DelayedBind` operation.
 *   This eliminates an unnecessary `DelayedBind` layer.
 *
 * If these simplifications are undesirable, users should call the `DelayedBind` constructor directly.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 *
 * @param matrices Pointers to `Matrix` objects.
 * @param row Whether to combine matrices by the rows (i.e., the output matrix has number of rows equal to the sum of the number of rows in `matrices`).
 * If false, combining is applied by the columns.
 *
 * @return A pointer to a `Matrix` representing the combined `matrices`.
 */
template<typename Value_, typename Index_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedBind(std::vector<std::shared_ptr<const Matrix<Value_, Index_> > > matrices, bool row) {
    if (matrices.size() == 1) {
        return std::const_pointer_cast<Matrix<Value_, Index_> >(matrices.front());
    }

    bool to_simplify = false;
    for (const auto& m : matrices) {
        auto cptr = dynamic_cast<const DelayedBind<Value_, Index_>*>(m.get());
        if (cptr != NULL && cptr->by_row() == row) {
            to_simplify = true;
            break;
        }
    }
    if (!to_simplify) {
        return std::shared_ptr<Matrix<Value_, Index_> >(new DelayedBind<Value_, Index_>(std::move(matrices), row));
    }

    std::vector<std::shared_ptr<const Matrix<Value_, Index_> > > new_matrices;
    new_matrices.reserve(matrices.size());
    for (auto& m : matrices) {
        auto cptr = dynamic_cast<const DelayedBind<Value_, Index_>*>(m.get());
        if (cptr != NULL && cptr->by_row() == row) {
            const auto& seeds = cptr->seeds();
            new_matrices.insert(new_matrices.end(), seeds.begin(), seeds.end());
        } else {
            new_matrices.push_back(std::move(m));
        }
    }
    return std::shared_ptr<Matrix<Value_, Index_> >(new DelayedBind<Value_, Index_>(std::move(new_matrices), row));
}

/**
 * @cond
 */
template<typename Value_, typename Index_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedBind(std::vector<std::shared_ptr<Matrix<Value_, Index_> > > matrices, bool row) {
    return make_DelayedBind(std::vector<std::shared_ptr<const Matrix<Value_, Index_> > >(matrices.begin(), matrices.end()), row);
}
/**
 * @endcond
 */

/**
 * @cond
 */
// Back-compatibility.
template<int margin_, typename Value_, typename Index_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedBind(std::vector<std::shared_ptr<const Matrix<Value_, Index_> > > matrices) {
    return make_DelayedBind(std::move(matrices), margin_ == 0);
}

template<int margin_, typename Value_, typename Index_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedBind(std::vector<std::shared_ptr<Matrix<Value_, Index_> > > matrices) {
    return make_DelayedBind(std::move(matrices), margin_ == 0);
}
/**
 * @endcond
 */

}

#endif
