#ifndef TATAMI_DELAYED_SUBSET
#define TATAMI_DELAYED_SUBSET

#include "typed_matrix.hpp"
#include <algorithm>
#include <memory>

/**
 * @file DelayedSubset.hpp
 *
 * Delayed subsetting, equivalent to the `DelayedSubset` class in the **DelayedArray** package.
 */

namespace tatami {

/**
 * @brief Delayed subsetting of a matrix.
 *
 * Implements delayed subsetting (i.e., slicing) on the rows or columns of a matrix.
 * This operation is "delayed" in that it is only evaluated on request, e.g., with `row()` or friends.
 *
 * @tparam MARGIN Dimension along which the addition is to occur.
 * If 0, the subset is applied to the rows; if 1, the subset is applied to the columns.
 * @tparam T Type of matrix value.
 * @tparam V Vector containing the subset indices.
 * @tparam IDX Type of index value.
 */
template<int MARGIN, typename T, typename IDX, class V>
class DelayedSubset : public typed_matrix<T, IDX> {
public:
    /**
     * @param p Pointer to the underlying (pre-subset) matrix.
     * @param idx Vector of 0-based indices to use for subsetting on the rows (if `MARGIN = 0`) or columns (if `MARGIN = 1`).
     */
    DelayedSubset(std::shared_ptr<const typed_matrix<T, IDX> > p, const V& idx) : mat(p), indices(idx) {}

    /**
     * @copydoc DelayedSubset
     */
    DelayedSubset(std::shared_ptr<typed_matrix<T, IDX> > p, const V& idx) : mat(p), indices(idx) {}

    /**
     * @copydoc DelayedSubset
     */
    DelayedSubset(std::shared_ptr<const typed_matrix<T, IDX> > p, V&& idx) : mat(p), indices(idx) {}

    /**
     * @copydoc DelayedSubset
     */
    DelayedSubset(std::shared_ptr<typed_matrix<T, IDX> > p, V&& idx) : mat(p), indices(idx) {}

    ~DelayedSubset() {}
public:
    const T* row(size_t r, T* buffer, size_t start, size_t end, workspace* work=nullptr) const {
        if constexpr(MARGIN==1) {
            subset_expanded<true>(r, buffer, start, end, work);
            return buffer;
        } else {
            return mat->row(indices[r], buffer, start, end, work);
        }
    }

    const T* column(size_t c, T* buffer, size_t start, size_t end, workspace* work=nullptr) const {
        if constexpr(MARGIN==1) {
            return mat->column(indices[c], buffer, start, end, work);
        } else {
            subset_expanded<false>(c, buffer, start, end, work);
            return buffer;
        }
    }

    using typed_matrix<T, IDX>::column;

    using typed_matrix<T, IDX>::row;

public:
    sparse_range<T, IDX> sparse_row(size_t r, T* out_values, IDX* out_indices, size_t start, size_t end, workspace* work=nullptr, bool sorted=true) const {
        if constexpr(MARGIN==1) {
            auto total = subset_sparse<true>(r, out_values, out_indices, start, end, work, sorted);
            return sparse_range<T, IDX>(total, out_values, out_indices);
        } else {
            return mat->sparse_row(indices[r], out_values, out_indices, start, end, work, sorted);
        }
    }

    sparse_range<T, IDX> sparse_column(size_t c, T* out_values, IDX* out_indices, size_t start, size_t end, workspace* work=nullptr, bool sorted=true) const {
        if constexpr(MARGIN==1) {
            return mat->sparse_column(indices[c], out_values, out_indices, start, end, work, sorted);
        } else {
            auto total = subset_sparse<false>(c, out_values, out_indices, start, end, work, sorted);
            return sparse_range<T, IDX>(total, out_values, out_indices);
        }
    }

    using typed_matrix<T, IDX>::sparse_column;

    using typed_matrix<T, IDX>::sparse_row;

public:
    /**
     * @return Number of rows after any subsetting is applied.
     */
    size_t nrow() const {
        if constexpr(MARGIN==0) {
            return indices.size();
        } else {
            return mat->nrow();
        }
    }
    
    /**
     * @return Number of columns after any subsetting is applied.
     */
    size_t ncol() const {
        if constexpr(MARGIN==0) {
            return mat->ncol();
        } else {
            return indices.size();
        }
    }

    /**
     * @return A null pointer or a shared pointer to a `workspace` object, depending on the underlying (pre-subsetted) matrix.
     */
    std::shared_ptr<workspace> new_workspace() const {
        return mat->new_workspace();
    }

    /**
     * @return The sparsity status of the underlying (pre-subsetted) matrix.
     */
    bool sparse() const {
        return mat->sparse();
    }

private:
    std::shared_ptr<const typed_matrix<T, IDX> > mat;
    V indices;

    template<bool ROW>
    void subset_expanded(size_t r, T* buffer, size_t start, size_t end, workspace* work) const {
        while (start < end) {
            auto original = start;
            auto previdx = indices[start];
            ++start;
            while (start < end && indices[start] == previdx + 1) {
                previdx = indices[start];
                ++start;
            }

            const T* ptr = NULL;
            size_t n = start - original;
            previdx = indices[original];
            if constexpr(ROW) {
                ptr = mat->row(r, buffer, previdx, previdx + n, work);
            } else {
                ptr = mat->column(r, buffer, previdx, previdx + n, work);
            }

            if (ptr != buffer) {
                std::copy(ptr, ptr + n, buffer);
            }
            buffer += n;
        }
        return;
    }

    template<bool ROW>
    size_t subset_sparse(size_t r, T* out_values, IDX* out_indices, size_t start, size_t end, workspace* work, bool sorted) const {
        size_t total = 0;
        while (start < end) {
            auto original = start;
            auto previdx = indices[start];
            ++start;
            while (start < end && indices[start] == previdx + 1) {
                previdx = indices[start];
                ++start;
            }

            size_t n = start - original;
            previdx = indices[original];
            sparse_range<T, IDX> range;
            if constexpr(ROW) {
                range = mat->sparse_row(r, out_values, out_indices, previdx, previdx + n, work, sorted);
            } else {
                range = mat->sparse_column(r, out_values, out_indices, previdx, previdx + n, work, sorted);
            }

            if (out_values != range.value) {
                std::copy(range.value, range.value + range.number, out_values);
            }
            for (size_t i = 0; i < range.number; ++i) {
                out_indices[i] = range.index[i] - previdx + original;
            }

            total += range.number;
            out_indices += range.number;
            out_values += range.number;
        }

        return total;
    }
};

// Separate classes defined to allow template deduction.
template<typename T, typename IDX, class V>
class DelayedRowSubset : public DelayedSubset<0, T, IDX, V> {
public:
    /**
     * @param p Pointer to the underlying (pre-subset) matrix.
     * @param idx Vector of 0-based indices to use for subsetting on the rows (if `MARGIN = 0`) or columns (if `MARGIN = 1`).
     */
    DelayedRowSubset(std::shared_ptr<const typed_matrix<T, IDX> > p, const V& idx) : DelayedSubset<0, T, IDX, V>(p, idx) {}

    /**
     * @copydoc DelayedRowSubset
     */
    DelayedRowSubset(std::shared_ptr<typed_matrix<T, IDX> > p, const V& idx) : DelayedSubset<0, T, IDX, V>(p, idx) {}

    /**
     * @copydoc DelayedRowSubset
     */
    DelayedRowSubset(std::shared_ptr<const typed_matrix<T, IDX> > p, V&& idx) : DelayedSubset<0, T, IDX, V>(p, std::forward<V>(idx)) {}

    /**
     * @copydoc DelayedRowSubset
     */
    DelayedRowSubset(std::shared_ptr<typed_matrix<T, IDX> > p, V&& idx) : DelayedSubset<0, T, IDX, V>(p, std::forward<V>(idx)) {}

    ~DelayedRowSubset() {}
};

template<typename T, typename IDX, class V>
class DelayedColumnSubset : public DelayedSubset<1, T, IDX, V> {
public:
    /**
     * @param p Pointer to the underlying (pre-subset) matrix.
     * @param idx Vector of 0-based indices to use for subsetting on the rows (if `MARGIN = 0`) or columns (if `MARGIN = 1`).
     */
    DelayedColumnSubset(std::shared_ptr<const typed_matrix<T, IDX> > p, const V& idx) : DelayedSubset<1, T, IDX, V>(p, idx) {}

    /**
     * @copydoc DelayedColumnSubset
     */
    DelayedColumnSubset(std::shared_ptr<typed_matrix<T, IDX> > p, const V& idx) : DelayedSubset<1, T, IDX, V>(p, idx) {}

    /**
     * @copydoc DelayedColumnSubset
     */
    DelayedColumnSubset(std::shared_ptr<const typed_matrix<T, IDX> > p, V&& idx) : DelayedSubset<1, T, IDX, V>(p, std::forward<V>(idx)) {}

    /**
     * @copydoc DelayedColumnSubset
     */
    DelayedColumnSubset(std::shared_ptr<typed_matrix<T, IDX> > p, V&& idx) : DelayedSubset<1, T, IDX, V>(p, std::forward<V>(idx)) {}

    ~DelayedColumnSubset() {}
};

}

#endif