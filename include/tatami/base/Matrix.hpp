#ifndef TATAMI_MATRIX_H
#define TATAMI_MATRIX_H

#include "SparseRange.hpp"
#include "Workspace.hpp"

/**
 * @file Matrix.hpp
 *
 * Virtual class for a matrix with a defined type.
 */

namespace tatami {

/**
 * @brief Virtual class for a matrix with a defined type.
 * 
 * @tparam T Type of the matrix values.
 * @tparam IDX Type of the row/column indices.
 */
template <typename T, typename IDX = int>
class Matrix {
public:
    ~Matrix() {}

    /** 
     * Type of the value to be returned by getters.
     */
    typedef T value;

    /** 
     * Type of the index to be returned by the sparse getters.
     */
    typedef IDX index;
public:
    /**
     * @return Number of rows.
     */
    virtual size_t nrow() const = 0;

    /**
     * @return Number of columns.
     */
    virtual size_t ncol() const = 0;

    /**
     * @param row Should a workspace for row extraction be returned?
     *
     * @return A shared pointer to a `Workspace` for row or column extraction, or a null pointer if no workspace is required.
     * Defaults to returning a null pointer if no specialized method is provided in derived classes.
     */
    virtual std::shared_ptr<Workspace> new_workspace(bool row) const { return nullptr; }

    /**
     * @return Is this matrix sparse?
     * Defaults to `false` if no specialized method is provided in derived classes.
     */
    virtual bool sparse() const { return false; }

    /**
     * @return The preferred dimension for extracting values.
     * If `true`, row-wise extraction is preferred; if `false`, column-wise extraction is preferred.
     * Defaults to `false` if no specialized method is provided in derived classes.
     */
    virtual bool prefer_rows() const { return false; }

    /**
     * @return A `pair` containing the number of matrix elements that prefer row-level access (`first`) or column-level access (`second`).
     *
     * This method is useful for determining the return value of `prefer_rows()` in combined matrices consisting of both row- and column-preferred submatrices.
     * In such cases, the net preference can be determined based on the combined size of the submatrices for each preference.
     *
     * For simpler matrices, the return value contains the total size of the matrix in one of the `double`s and zero in the other.
     */
    virtual std::pair<double, double> dimension_preference () const {
        double size = static_cast<double>(nrow()) * static_cast<double>(ncol());
        if (prefer_rows()) {
            return std::make_pair(size, 0.0);
        } else {
            return std::make_pair(0.0, size);
        }
    }

public:
    /**
     * `buffer` may not necessarily be filled upon extraction if a pointer can be returned to the underlying data store.
     * This can be checked by comparing the returned pointer to `buffer`; if they are the same, `buffer` has been filled.
     *
     * If `work` is not a null pointer, it should have been generated by `new_workspace()` with `row = true`.
     * This is optional and should only affect the efficiency of extraction, not the contents.
     *
     * @param r Index of the row.
     * @param buffer Pointer to an array with enough space for at least `last - first` values.
     * @param first First column to extract for row `r`.
     * @param last One past the last column to extract in `r`.
     * @param work Pointer to a workspace.
     *
     * @return Pointer to the values of row `r`, starting from the value in the `first` column and containing `last - first` valid entries.
     */
    virtual const T* row(size_t r, T* buffer, size_t first, size_t last, Workspace* work=nullptr) const = 0;

    /**
     * `buffer` may not necessarily be filled upon extraction if a pointer can be returned to the underlying data store.
     * This can be checked by comparing the returned pointer to `buffer`; if they are the same, `buffer` has been filled.
     *
     * If `work` is not a null pointer, it should have been generated by `new_workspace()` with `row = false`.
     * This is optional and should only affect the efficiency of extraction, not the contents.
     *
     * @param c Index of the column.
     * @param buffer Pointer to an array with enough space for at least `last - first` values.
     * @param first First row to extract for column `c`.
     * @param last One past the last row to extract in `c`.
     * @param work Pointer to a workspace.
     *
     * @return Pointer to the values of column `c`, starting from the value in the `first` row and containing `last - first` valid entries.
     */
    virtual const T* column(size_t c, T* buffer, size_t first, size_t last, Workspace* work=nullptr) const = 0;

    /**
     * @param r Index of the row.
     * @param buffer Pointer to an array with enough space for at least `ncol()` values.
     * @param work Pointer to a workspace, see `row()` for details.
     *
     * @return Pointer to the values of row `r`.
     */
    const T* row(size_t r, T* buffer, Workspace* work=nullptr) const {
        return row(r, buffer, 0, this->ncol(), work);
    }

    /**
     * @param c Index of the column.
     * @param buffer Pointer to an array with enough space for at least `nrow()` values.
     * @param work Pointer to a workspace, see `column()` for details.
     *
     * @return Pointer to the values of column `c`.
     */
    const T* column(size_t c, T* buffer, Workspace* work=nullptr) const {
        return column(c, buffer, 0, this->nrow(), work);
    }

public:
    /**
     * @param r Index of the row.
     * @param buffer Pointer to an array with enough space for at least `last - first` values.
     * @param first First column to extract for row `r`.
     * @param last One past the last column to extract in `r`.
     * @param work Pointer to a workspace, see `row()` for details.
     *
     * @return The array at `buffer` is filled with the values of row `r`, starting from the value in the `first` column and containing `last - first` valid entries.
     * `buffer` itself is returned.
     */
    const T* row_copy(size_t r, T* buffer, size_t first, size_t last, Workspace* work=nullptr) const {
        auto ptr = row(r, buffer, first, last, work);
        copy_over(ptr, buffer, last - first);
        return buffer;
    }

    /**
     * @param c Index of the column.
     * @param buffer Pointer to an array with enough space for at least `last - first` values.
     * @param first First row to extract for column `c`.
     * @param last One past the last row to extract in `c`.
     * @param work Pointer to a workspace, see `column()` for details.
     *
     * @return The array at `buffer` is filled with the values of column `c`, starting from the value in the `first` row and containing `last - first` valid entries.
     * `buffer` itself is returned.
     */
    const T* column_copy(size_t c, T* buffer, size_t first, size_t last, Workspace* work=nullptr) const {
        auto ptr = column(c, buffer, first, last, work);
        copy_over(ptr, buffer, last - first);
        return buffer;
    }

    /**
     * @param r Index of the row.
     * @param buffer Pointer to an array with enough space for at least `ncol()` values.
     * @param work Pointer to a workspace, see `row()` for details.
     *
     * @return The array at `buffer` is filled with all values of row `r`.
     * `buffer` itself is returned.
     */
    const T* row_copy(size_t r, T* buffer, Workspace* work=nullptr) const {
        return row_copy(r, buffer, 0, this->ncol(), work);
    }

    /**
     * @param c Index of the column.
     * @param buffer Pointer to an array with enough space for at least `nrow()` values.
     * @param work Pointer to a workspace, see `column()` for details.
     *
     * @return The array at `buffer` is filled with all values of column `c`.
     * `buffer` itself is returned.
     */
    const T* column_copy(size_t c, T* buffer, Workspace* work=nullptr) const {
        return column_copy(c, buffer, 0, this->nrow(), work);
    }

private:
    static void copy_over(const T* src, T* dest, size_t n) {
        if (src!=dest) {
            std::copy(src, src + n, dest);
        }
        return;
    }

public:
    /**
     * A more convenient but (slightly) less efficient version of the `row()` method.
     * Callers do not have to supply `buffer`; instead a new allocation is performed every time.
     *
     * @param r Index of the row.
     * @param first First column to extract for row `r`.
     * @param last One past the last column to extract in `r`.
     * @param work Pointer to a workspace, see `row()` for details.
     *
     * @return A vector containing the values of row `r`, starting from the value in the `first` column and containing `last - first` valid entries.
     */
    std::vector<T> row(size_t r, size_t first, size_t last, Workspace* work=nullptr) const {
        std::vector<T> output(last - first);
        auto ptr = row_copy(r, output.data(), first, last, work);
        return output;
    }

    /**
     * A more convenient but (slightly) less efficient version of the `column()` method.
     * Callers do not have to supply `buffer`; instead a new allocation is performed every time.
     *
     * @param c Index of the column.
     * @param first First row to extract for column `c`.
     * @param last One past the last row to extract in `c`.
     * @param work Pointer to a workspace, see `column()` for details.
     *
     * @return A vector containing the values of column `c`, starting from the value in the `first` row and containing `last - first` valid entries.
     */
    std::vector<T> column(size_t c, size_t first, size_t last, Workspace* work=nullptr) const {
        std::vector<T> output(last - first);
        auto ptr = column_copy(c, output.data(), first, last, work);
        return output;
    }

    /**
     * @param r Index of the row.
     * @param work Pointer to a workspace, see `row()` for details.
     *
     * @return A vector containing all values of row `r`.
     */
    std::vector<T> row(size_t r, Workspace* work=nullptr) const {
        return row(r, 0, this->ncol(), work);
    }

    /**
     * @param c Index of the column.
     * @param work Pointer to a workspace, see `column()` for details.
     *
     * @return A vector containing all values of column `c`.
     */
    std::vector<T> column(size_t c, Workspace* work=nullptr) const {
        return column(c, 0, this->nrow(), work);
    }

public:
    /**
     * `vbuffer` may not necessarily be filled upon extraction if a pointer can be returned to the underlying data store.
     * This be checked by comparing the returned `SparseRange::value` pointer to `vbuffer`; if they are the same, `vbuffer` has been filled. 
     * The same applies for `ibuffer` and the returned `SparseRange::index` pointer.
     *
     * Values in `vbuffer` are not guaranteed to be non-zero.
     * If zeroes are explicitly initialized in the underlying representation, they will be reported here.
     * However, one can safely assume that all values _not_ in `vbuffer` are zero.
     *
     * If `work` is not a null pointer, it should have been generated by `new_workspace()` with `row = false`.
     * This is optional and should only affect the efficiency of extraction, not the contents.
     *
     * Setting `sorted = false` can reduce computational work in situations where the order of non-zero elements does not matter.
     *
     * @param r Index of the row.
     * @param vbuffer Pointer to an array with enough space for at least `last - first` values.
     * @param ibuffer Pointer to an array with enough space for at least `last - first` indices.
     * @param first First column to extract for row `r`.
     * @param last One past the last column to extract in `r`.
     * @param work Pointer to a workspace.
     * @param sorted Should the non-zero elements be sorted by their indices?
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `r` from column `first` up to `last`.
     * This also contains pointers to arrays containing their column indices and values.
     */
    virtual SparseRange<T, IDX> sparse_row(size_t r, T* vbuffer, IDX* ibuffer, size_t first, size_t last, Workspace* work=nullptr, bool sorted=true) const {
        const T* val = row(r, vbuffer, first, last, work);
        for (size_t i = first; i < last; ++i) {
            ibuffer[i - first] = i;
        }
        return SparseRange(last - first, val, ibuffer); 
    }

    /**
     * `vbuffer` may not necessarily be filled upon extraction if a pointer can be returned to the underlying data store.
     * This be checked by comparing the returned `SparseRange::value` pointer to `vbuffer`; if they are the same, `vbuffer` has been filled. 
     * The same applies for `ibuffer` and the returned `SparseRange::index` pointer.
     *
     * Values in `vbuffer` are not guaranteed to be non-zero.
     * If zeroes are explicitly initialized in the underlying representation, they will be reported here.
     * However, one can safely assume that all values _not_ in `vbuffer` are zero.
     *
     * If `work` is not a null pointer, it should have been generated by `new_workspace()` with `row = true`.
     * This is optional and should only affect the efficiency of extraction, not the contents.
     *
     * Setting `sorted = false` can reduce computational work in situations where the order of non-zero elements does not matter.
     *
     * @param c Index of the column.
     * @param vbuffer Pointer to an array with enough space for at least `last - first` values.
     * @param ibuffer Pointer to an array with enough space for at least `last - first` indices.
     * @param first First row to extract for column `c`.
     * @param last One past the last row to extract in `c`.
     * @param work Pointer to a workspace.
     * @param sorted Should the non-zero elements be sorted by their indices?
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `c` from column `first` up to `last`.
     * This also contains pointers to arrays containing their row indices and values.
     */
    virtual SparseRange<T, IDX> sparse_column(size_t c, T* vbuffer, IDX* ibuffer, size_t first, size_t last, Workspace* work=nullptr, bool sorted=true) const {
        const T* val = column(c, vbuffer, first, last, work);
        for (size_t i = first; i < last; ++i) {
            ibuffer[i - first] = i;
        }
        return SparseRange(last - first, val, ibuffer); 
    }

    /**
     * @param r Index of the row.
     * @param vbuffer Pointer to an array with enough space for at least `ncol()` values.
     * @param ibuffer Pointer to an array with enough space for at least `ncol()` indices.
     * @param work Pointer to a workspace, see comments in `sparse_row()`.
     * @param sorted Should the non-zero elements be sorted by their indices?
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `r`.
     * This also contains pointers to arrays containing their column indices and values.
     */
    SparseRange<T, IDX> sparse_row(size_t r, T* vbuffer, IDX* ibuffer, Workspace* work=nullptr, bool sorted=true) const {
        return sparse_row(r, vbuffer, ibuffer, 0, this->ncol(), work, sorted);
    }

    /**
     * @param c Index of the column.
     * @param vbuffer Pointer to an array with enough space for at least `nrow()` values.
     * @param ibuffer Pointer to an array with enough space for at least `nrow()` indices.
     * @param work Pointer to a workspace, see comments in `sparse_column()`.
     * @param sorted Should the non-zero elements be sorted by their indices?
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `c`.
     * This also contains pointers to arrays containing their row indices and values.
     */
    SparseRange<T, IDX> sparse_column(size_t c, T* vbuffer, IDX* ibuffer, Workspace* work=nullptr, bool sorted=true) const {
        return sparse_column(c, vbuffer, ibuffer, 0, this->nrow(), work, sorted);
    }

public:
    /**
     * @param r Index of the row.
     * @param vbuffer Pointer to an array with enough space for at least `last - first` values.
     * @param ibuffer Pointer to an array with enough space for at least `last - first` indices.
     * @param first First column to extract for row `r`.
     * @param last One past the last column to extract in `r`.
     * @param copy Whether the non-zero values and/or indices should be copied into `vbuffer` and `ibuffer`, respectively.
     * @param work Pointer to a workspace, see comments in `sparse_row()`.
     * @param sorted Should the non-zero elements be sorted by their indices? See `sparse_row()` for details.
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `r` from column `first` up to `last`.
     * This also contains pointers to arrays containing their column indices and values.
     * Depending on `copy`, values and incides will be copied into `vbuffer` and/or `ibuffer`.
     */
    SparseRange<T, IDX> sparse_row_copy(size_t r, T* vbuffer, IDX* ibuffer, size_t first, size_t last, SparseCopyMode copy, Workspace* work=nullptr, bool sorted=true) const {
        auto output = sparse_row(r, vbuffer, ibuffer, first, last, work, sorted);
        
        if ((copy == SPARSE_COPY_BOTH || copy == SPARSE_COPY_INDEX) && output.index != ibuffer) {
            copy_over(output.index, ibuffer, last - first);
            output.index = ibuffer;
        }

        if ((copy == SPARSE_COPY_BOTH || copy == SPARSE_COPY_VALUE) && output.value != vbuffer) {
            copy_over(output.value, vbuffer, last - first);
            output.value = vbuffer;
        }

        return output;
    }

    /**
     * @param c Index of the column.
     * @param vbuffer Pointer to an array with enough space for at least `last - first` values.
     * @param ibuffer Pointer to an array with enough space for at least `last - first` indices.
     * @param first First row to extract for column `c`.
     * @param last One past the last row to extract in `c`.
     * @param copy Whether the non-zero values and/or indices should be copied into `vbuffer` and `ibuffer`, respectively.
     * @param work Pointer to a workspace, see comments in `sparse_column()`.
     * @param sorted Should the non-zero elements be sorted by their indices? See `sparse_column()` for details.
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `c` from column `first` up to `last`.
     * This also contains pointers to arrays containing their row indices and values.
     * Depending on `copy`, values and incides will be copied into `vbuffer` and/or `ibuffer`.
     */
    SparseRange<T, IDX> sparse_column_copy(size_t c, T* vbuffer, IDX* ibuffer, size_t first, size_t last, SparseCopyMode copy, Workspace* work=nullptr, bool sorted=true) const {
        auto output = sparse_column(c, vbuffer, first, last, work);

        if ((copy == SPARSE_COPY_BOTH || copy == SPARSE_COPY_INDEX) && output.index != ibuffer) {
            copy_over(output.index, ibuffer, last - first);
            output.index = ibuffer;
        }

        if ((copy == SPARSE_COPY_BOTH || copy == SPARSE_COPY_VALUE) && output.value != vbuffer) {
            copy_over(output.value, vbuffer, last - first);
            output.value = vbuffer;
        }

        return output;
    }

    /**
     * @param r Index of the row.
     * @param vbuffer Pointer to an array with enough space for at least `ncol()` values.
     * @param ibuffer Pointer to an array with enough space for at least `ncol()` indices.
     * @param copy Whether the non-zero values and/or indices should be copied into `vbuffer` and `ibuffer`, respectively.
     * @param work Pointer to a workspace, see comments in `sparse_row()`.
     * @param sorted Should the non-zero elements be sorted by their indices? See `sparse_row()` for details.
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `r`.
     * This also contains pointers to arrays containing their column indices and values.
     * Depending on `copy`, values and incides will be copied into `vbuffer` and/or `ibuffer`.
     */
    SparseRange<T, IDX> sparse_row_copy(size_t r, T* vbuffer, IDX* ibuffer, SparseCopyMode copy, Workspace* work=nullptr, bool sorted=true) const {
        return sparse_row_copy(r, vbuffer, ibuffer, 0, this->ncol(), copy, work, sorted);
    }

    /**
     * @param c Index of the column.
     * @param vbuffer Pointer to an array with enough space for at least `nrow()` values.
     * @param ibuffer Pointer to an array with enough space for at least `nrow()` indices.
     * @param copy Whether the non-zero values and/or indices should be copied into `vbuffer` and `ibuffer`, respectively.
     * @param work Pointer to a workspace, see comments in `sparse_column()`.
     * @param sorted Should the non-zero elements be sorted by their indices? See `sparse_column()` for details.
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `c`.
     * This also contains pointers to arrays containing their row indices and values.
     * Depending on `copy`, values and incides will be copied into `vbuffer` and/or `ibuffer`.
     */
    SparseRange<T, IDX> sparse_column_copy(size_t c, T* vbuffer, IDX* ibuffer, SparseCopyMode copy, Workspace* work=nullptr, bool sorted=true) const {
        return sparse_column_copy(c, vbuffer, ibuffer, 0, this->nrow(), copy, work, sorted);
    }

public:
    /**
     * @param r Index of the row.
     * @param first First column to extract for row `r`.
     * @param last One past the last column to extract in `r`.
     * @param work Pointer to a workspace, see comments in `sparse_row()`.
     * @param sorted Should the non-zero elements be sorted by their indices? See `sparse_row()` for details.
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `r` from column `first` up to `last`.
     * This also contains pointers to arrays containing their column indices and values.
     * Depending on `copy`, values and incides will be copied into `vbuffer` and/or `ibuffer`.
     */
    SparseRangeCopy<T, IDX> sparse_row(size_t r, size_t first, size_t last, Workspace* work=nullptr, bool sorted=true) const {
        SparseRangeCopy<T, IDX> output(last - first);
        sparse_row_copy(r, output.value.data(), output.index.data(), first, last, SPARSE_COPY_BOTH, work, sorted);
        return output;
    }

    /**
     * @param c Index of the column.
     * @param first First row to extract for column `c`.
     * @param last One past the last row to extract in `c`.
     * @param work Pointer to a workspace, see comments in `sparse_column()`.
     * @param sorted Should the non-zero elements be sorted by their indices? See `sparse_column()` for details.
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `c` from column `first` up to `last`.
     * This also contains pointers to arrays containing their row indices and values.
     * Depending on `copy`, values and incides will be copied into `vbuffer` and/or `ibuffer`.
     */
    SparseRangeCopy<T, IDX> sparse_column(size_t c, size_t first, size_t last, Workspace* work=nullptr, bool sorted=true) const {
        SparseRangeCopy<T, IDX> output(last - first);
        sparse_column_copy(c, output.value.data(), output.index.data(), first, last, SPARSE_COPY_BOTH, work, sorted);
        return output;
    }

    /**
     * @param r Index of the row.
     * @param work Pointer to a workspace, see comments in `sparse_row()`.
     * @param sorted Should the non-zero elements be sorted by their indices? See `sparse_row()` for details.
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `r`.
     * This also contains pointers to arrays containing their column indices and values.
     * Depending on `copy`, values and incides will be copied into `vbuffer` and/or `ibuffer`.
     */
    SparseRangeCopy<T, IDX> sparse_row(size_t r, Workspace* work=nullptr, bool sorted=true) const {
        return sparse_row(r, 0, this->ncol(), work, sorted);
    }

    /**
     * @param c Index of the column.
     * @param work Pointer to a workspace, see comments in `sparse_column()`.
     * @param sorted Should the non-zero elements be sorted by their indices? See `sparse_column()` for details.
     *
     * @return A `SparseRange` object containing the number of non-zero elements in `c`.
     * This also contains pointers to arrays containing their row indices and values.
     * Depending on `copy`, values and incides will be copied into `vbuffer` and/or `ibuffer`.
     */
    SparseRangeCopy<T, IDX> sparse_column(size_t c, Workspace* work=nullptr, bool sorted=true) const {
        return sparse_column(c, 0, this->nrow(), work, sorted);
    }
};

/**
 * A convenient shorthand for the most common use case of double-precision matrices.
 */
using NumericMatrix = Matrix<double, int>;

}

#endif
