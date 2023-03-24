#ifndef TATAMI_COMPRESSED_SPARSE_MATRIX_H
#define TATAMI_COMPRESSED_SPARSE_MATRIX_H

#include "Matrix.hpp"
#include "has_data.hpp"
#include "SparseRange.hpp"

#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <stdexcept>

/**
 * @file CompressedSparseMatrix.hpp
 *
 * @brief Compressed sparse matrix representation. 
 *
 * `typedef`s are provided for the usual row and column formats. 
 */

namespace tatami {

/**
 * @brief Compressed sparse matrix representation.
 *
 * @tparam ROW Whether this is a compressed sparse row representation.
 * If `false`, a compressed sparse column representation is used instead.
 * @tparam T Type of the matrix values.
 * @tparam IDX Type of the row/column indices.
 * @tparam U Vector class used to store the matrix values internally.
 * This does not necessarily have to contain `T`, as long as the type is convertible to `T`.
 * Methods should be available for `size()`, `begin()`, `end()` and `[]`.
 * If a method is available for `data()` that returns a `const T*`, it will also be used.
 * @tparam V Vector class used to store the row/column indices internally.
 * This does not necessarily have to contain `IDX`, as long as the type is convertible to `IDX`.
 * Methods should be available for `size()`, `begin()`, `end()` and `[]`.
 * If a method is available for `data()` that returns a `const IDX*`, it will also be used.
 * @tparam W Vector class used to store the column/row index pointers.
 * Methods should be available for `size()`, `begin()`, `end()` and `[]`.
 */
template<bool ROW, typename T, typename IDX = int, class U = std::vector<T>, class V = std::vector<IDX>, class W = std::vector<size_t> >
class CompressedSparseMatrix : public Matrix<T, IDX> {
public:
    /**
     * @param nr Number of rows.
     * @param nc Number of columns.
     * @param vals Vector of non-zero elements.
     * @param idx Vector of row indices (if `ROW=false`) or column indices (if `ROW=true`) for the non-zero elements.
     * @param ptr Vector of index pointers.
     * @param check Should the input vectors be checked for validity?
     *
     * If `check=true`, the constructor will check that `vals` and `idx` have the same length;
     * `ptr` is ordered with first and last values set to 0 and the number of non-zero elements, respectively;
     * and `idx` is ordered within each interval defined by successive elements of `ptr`.
     */
    CompressedSparseMatrix(size_t nr, size_t nc, U vals, V idx, W ptr, bool check=true) : nrows(nr), ncols(nc), values(std::move(vals)), indices(std::move(idx)), indptrs(std::move(ptr)) {
        check_values(check); 
        return;
    }

private:
    size_t nrows, ncols;
    U values;
    V indices;
    W indptrs;

    void check_values(bool check) {
        if (!check) {
            return;
        }

        if (values.size() != indices.size()) {
            throw std::runtime_error("'values' and 'indices' should be of the same length");
        }

        if (ROW) {
            if (indptrs.size() != nrows + 1){
                throw std::runtime_error("length of 'indptrs' should be equal to 'nrows + 1'");
            }
        } else {
            if (indptrs.size() != ncols + 1){
                throw std::runtime_error("length of 'indptrs' should be equal to 'ncols + 1'");
            }
        }

        if (indptrs[0] != 0) {
            throw std::runtime_error("first element of 'indptrs' should be zero");
        }
        if (indptrs[indptrs.size() - 1] != indices.size()) {
            throw std::runtime_error("last element of 'indptrs' should be equal to length of 'indices'");
        }

        size_t counter = 0;
        for (size_t i = 1; i < indptrs.size(); ++i) {
            if (indptrs[i] < indptrs[i-1]) {
                throw std::runtime_error("'indptrs' should be in increasing order");
            }

            if (counter < indices.size()) {
                auto previous = indices[counter];
                ++counter;
                while (counter < indptrs[i]) {
                    if (previous >= indices[counter]) {
                        if (ROW) {
                            throw std::runtime_error("'indices' should be strictly increasing within each row");
                        } else {
                            throw std::runtime_error("'indices' should be strictly increasing within each column");
                        }
                    }
                    ++counter;
                }
            }
        }

        return;
    }

public:
    size_t nrow() const { return nrows; }

    size_t ncol() const { return ncols; }

    /**
     * @return `true`.
     */
    bool sparse() const { return true; }

    /**
     * @return `true` if `ROW = true` (for `CompressedSparseRowMatrix` objects), otherwise returns `false` (for `CompressedSparseColumnMatrix` objects).
     */
    bool prefer_rows() const { return ROW; }

    using Matrix<T, IDX>::row;

    using Matrix<T, IDX>::column;

    using Matrix<T, IDX>::sparse_row;

    using Matrix<T, IDX>::sparse_column;

private:
    struct SecondaryWorkspaceBase {
        typedef typename std::remove_reference<decltype(std::declval<W>()[0])>::type indptr_type;
        typedef typename std::remove_reference<decltype(std::declval<V>()[0])>::type index_type;

        SecondaryWorkspaceBase(size_t max_index, const V& idx, const W& idp, size_t start, size_t length) : current_indptrs(idp.begin() + start, idp.begin() + length), current_indices(length) {
            /* Here, the general idea is to store a local copy of the actual
             * row indices (for CSC matrices; column indices, for CSR matrices)
             * so that we don't have to keep on doing cache-unfriendly look-ups
             * for the indices based on the pointers that we do have. This assumes
             * that the density is so low that updates to the local indices are
             * rare relative to the number of comparisons to those same indices.
             * Check out the `secondary_dimension()` function for how this is used.
             */
            auto idpIt = idp.begin() + start;
            for (size_t i = 0; i < length; ++i, ++idpIt) {
                current_indices[i] = (*idpIt < *(idpIt + 1) ? idx[*idpIt] : max_index);
            }
            return;
        } 

        SecondaryWorkspaceBase(size_t max_index, const V& idx, const W& idp, size_t length, const IDX* indices) : current_indptrs(length), current_indices(length) {
            for (size_t i0 = 0; i0 < length; ++i0) {
                auto i = indices[i0];
                current_indptrs[i0] = idp[i];
                current_indices[i0] = (idp[i] < idp[i + 1] ? idx[idp[i]] : max_index);
            }
            return;
        } 

        size_t previous_request = 0;
        std::vector<indptr_type> current_indptrs; // the current position of the pointer

        // The current index being pointed to, i.e., current_indices[0] <= indices[current_ptrs[0]]. 
        // If current_ptrs[0] is out of range, the current index is instead set to the maximum index
        // (e.g., max rows for CSC matrices).
        std::vector<index_type> current_indices;

        // Efficiency boosts for reverse consecutive searches. These aren't populated until they're required,
        // given that forward consecutive searches are far more common.
        // 'below_indices' holds 1-past-the-index of the non zero element immediately below 'previous_request'.
        std::vector<index_type> below_indices;
        bool store_below = false;
    };

public:
    /**
     * @cond
     */
    struct CompressedSparseSecondaryWorkspace : public Workspace<!ROW> {
        CompressedSparseSecondaryWorkspace(size_t max_index, const V& idx, const W& idp) : core(max_index, idx, idp, 0, idp.size() - 1) {}
        SecondaryWorkspaceBase core;
    };
    /**
     * @endcond
     */

    std::shared_ptr<RowWorkspace> new_row_workspace() const { 
        if constexpr(ROW) {
            return nullptr; 
        } else {
            return std::shared_ptr<RowWorkspace>(new CompressedSparseSecondaryWorkspace(nrows, indices, indptrs));
        }
    }

    std::shared_ptr<ColumnWorkspace> new_column_workspace() const { 
        if constexpr(ROW) {
            return std::shared_ptr<ColumnWorkspace>(new CompressedSparseSecondaryWorkspace(ncols, indices, indptrs));
        } else {
            return nullptr; 
        }
    }

    const T* row(size_t r, T* buffer, RowWorkspace* work) const {
        if constexpr(ROW) {
            primary_dimension_expanded(r, 0, ncols, ncols, buffer);
        } else {
            secondary_dimension_expanded(r, 0, ncols, static_cast<CompressedSparseSecondaryWorkspace*>(work)->core, buffer);
        }
        return buffer;
    }

    const T* column(size_t c, T* buffer, ColumnWorkspace* work) const {
        if constexpr(ROW) {
            secondary_dimension_expanded(c, 0, nrows, static_cast<CompressedSparseSecondaryWorkspace*>(work)->core, buffer);
        } else {
            primary_dimension_expanded(c, 0, nrows, nrows, buffer);
        }
        return buffer;
    }

    SparseRange<T, IDX> sparse_row(size_t r, T* vbuffer, IDX* ibuffer, RowWorkspace* work, bool sorted=true) const {
        // It's always sorted anyway, no need to pass along 'sorted'.
        if constexpr(ROW) {
            return primary_dimension_raw(r, 0, ncols, ncols, vbuffer, ibuffer);
        } else {
            return secondary_dimension_raw(r, 0, ncols, static_cast<CompressedSparseSecondaryWorkspace*>(work)->core, vbuffer, ibuffer); 
        }
    }

    SparseRange<T, IDX> sparse_column(size_t c, T* vbuffer, IDX* ibuffer, ColumnWorkspace* work, bool sorted=true) const {
        // It's always sorted anyway, no need to pass along 'sorted'.
        if constexpr(ROW) {
            return secondary_dimension_raw(c, 0, nrows, static_cast<CompressedSparseSecondaryWorkspace*>(work)->core, vbuffer, ibuffer); 
        } else {
            return primary_dimension_raw(c, 0, nrows, nrows, vbuffer, ibuffer);
        }
    }

public:
    /**
     * @cond
     */
    struct CompressedSparsePrimaryBlockWorkspace : public BlockWorkspace<ROW> {
        CompressedSparsePrimaryBlockWorkspace(size_t start, size_t length) : BlockWorkspace<ROW>(start, length) {}
    };

    struct CompressedSparseSecondaryBlockWorkspace : public BlockWorkspace<!ROW> {
        CompressedSparseSecondaryBlockWorkspace(size_t start, size_t length, size_t max_index, const V& idx, const W& idp) : BlockWorkspace<!ROW>(start, length), core(max_index, idx, idp, start, length) {}
        SecondaryWorkspaceBase core;
    };
    /**
     * @endcond
     */

    std::shared_ptr<RowBlockWorkspace> new_row_workspace(size_t start, size_t length) const { 
        if constexpr(ROW) {
            return std::shared_ptr<RowBlockWorkspace>(new CompressedSparsePrimaryBlockWorkspace(start, length));
        } else {
            return std::shared_ptr<RowBlockWorkspace>(new CompressedSparseSecondaryBlockWorkspace(start, length, nrows, indices, indptrs));
        }
    }

    std::shared_ptr<ColumnBlockWorkspace> new_column_workspace(size_t start, size_t length) const { 
        if constexpr(ROW) {
            return std::shared_ptr<ColumnBlockWorkspace>(new CompressedSparseSecondaryBlockWorkspace(start, length, ncols, indices, indptrs));
        } else {
            return std::shared_ptr<ColumnBlockWorkspace>(new CompressedSparsePrimaryBlockWorkspace(start, length));
        }
    }

    const T* row(size_t r, T* buffer, RowBlockWorkspace* work) const {
        if constexpr(ROW) {
            primary_dimension_expanded(r, work->start, work->length, ncols, buffer);
        } else {
            secondary_dimension_expanded(r, work->start, work->length, static_cast<CompressedSparseSecondaryBlockWorkspace*>(work)->core, buffer);
        }
        return buffer;
    }

    const T* column(size_t c, T* buffer, ColumnBlockWorkspace* work) const {
        if constexpr(ROW) {
            secondary_dimension_expanded(c, work->start, work->length, static_cast<CompressedSparseSecondaryBlockWorkspace*>(work)->core, buffer);
        } else {
            primary_dimension_expanded(c, work->start, work->length, nrows, buffer);
        }
        return buffer;
    }

    SparseRange<T, IDX> sparse_row(size_t r, T* vbuffer, IDX* ibuffer, RowBlockWorkspace* work) const {
        if constexpr(ROW) {
            return primary_dimension_raw(r, work->start, work->length, ncols, vbuffer, ibuffer);
        } else {
            return secondary_dimension_raw(r, work->start, work->length, static_cast<CompressedSparseSecondaryBlockWorkspace*>(work)->core, vbuffer, ibuffer);
        }
    }

    SparseRange<T, IDX> sparse_column(size_t c, T* vbuffer, IDX* ibuffer, ColumnBlockWorkspace* work) const {
        if constexpr(ROW) {
            return secondary_dimension_raw(c, work->start, work->length, static_cast<CompressedSparseSecondaryBlockWorkspace*>(work)->core, vbuffer, ibuffer);
        } else {
            return primary_dimension_raw(c, work->start, work->length, nrows, vbuffer, ibuffer);
        }
    }

private:
    std::pair<size_t, size_t> primary_dimension(size_t i, size_t first, size_t last, size_t otherdim) const {
        const auto pstart = indptrs[i];
        auto iIt = indices.begin() + pstart, 
             eIt = indices.begin() + indptrs[i + 1]; 
        auto xIt = values.begin() + pstart;

        if (first) { // Jumping ahead if non-zero.
            iIt = std::lower_bound(iIt, eIt, first);
        } 

        if (last != otherdim) { // Jumping to last element.
            eIt = std::lower_bound(iIt, eIt, last);
        }

        return std::make_pair(iIt - indices.begin(), eIt - iIt);
    }

    SparseRange<T, IDX> primary_dimension_raw(size_t i, size_t first, size_t last, size_t otherdim, T* out_values, IDX* out_indices) const {
        auto obtained = primary_dimension(i, first, last, otherdim);
        SparseRange<T, IDX> output(obtained.second);

        if constexpr(has_data<T, U>::value) {
            output.value = values.data() + obtained.first;
        } else {
            auto vIt = values.begin() + obtained.first;
            std::copy(vIt, vIt + obtained.second, out_values);
            output.value = out_values;
        }

        if constexpr(has_data<IDX, V>::value) {
            output.index = indices.data() + obtained.first;
        } else {
            auto iIt = indices.begin() + obtained.first;
            std::copy(iIt, iIt + obtained.second, out_indices);
            output.index = out_indices;
        }

        return output;
    }

    void primary_dimension_expanded(size_t i, size_t first, size_t last, size_t otherdim, T* out_values) const {
        std::fill(out_values, out_values + (last - first), static_cast<T>(0));
        auto obtained = primary_dimension(i, first, last, otherdim);
        auto vIt = values.begin() + obtained.first;
        auto iIt = indices.begin() + obtained.first;
        for (size_t x = 0; x < obtained.second; ++x, ++vIt, ++iIt) {
            out_values[*iIt - first] = *vIt;            
        }
        return;
    }

private:
    size_t max_secondary_index() const {
        if constexpr(ROW) {
            return ncols;
        } else {
            return nrows;
        }
    }

    auto define_below_index(size_t pos, size_t curptr) const {
        // We set it to zero if there is no other non-zero element before
        // 'curptr' in 'pos'.  This aligns with the idea of 'below_indices'
        // holding 1-past-the-immediately-below-index; if it is zero,
        // nothing remaings below 'curptr'.
        return (curptr > indptrs[pos] ? indices[curptr - 1] + 1 : 0);
    }

    template<class Store>
    void secondary_dimension_above(IDX secondary, size_t primary, size_t index_primary, SecondaryWorkspaceBase& work, Store& output) const {
        auto& curdex = work.current_indices[index_primary];
        auto& curptr = work.current_indptrs[index_primary];

        // Remember that we already store the index corresponding to the current indptr.
        // So, we only need to do more work if the request is greater than that index.
        if (secondary > curdex) {
            auto max_index = max_secondary_index();
            auto limit = indptrs[primary + 1];

            // Having a peek at the index of the next non-zero
            // element; maybe we're lucky enough that the requested
            // index is below this, as would be the case for
            // consecutive or near-consecutive accesses.
            ++curptr;
            if (curptr < limit) {
                auto candidate = indices[curptr];
                if (candidate >= secondary) {
                    curdex = candidate;

                } else if (secondary + 1 == max_index) {
                    // Special case if the requested index is at the end of the matrix,
                    // in which case we can just jump there directly rather than 
                    // doing an unnecessary binary search.
                    curptr = limit - 1;
                    if (indices[curptr] == secondary) {
                        curdex = indices[curptr];
                    } else {
                        ++curptr;
                        curdex = max_index;
                    }

                } else {
                    // Otherwise we need to search.
                    curptr = std::lower_bound(indices.begin() + curptr, indices.begin() + limit, secondary) - indices.begin();
                    curdex = (curptr < limit ? indices[curptr] : max_index);
                }
            } else {
                curdex = max_index;
            }

            if (work.store_below) {
                work.below_indices[index_primary] = define_below_index(primary, curptr);
            }
        }

        if (secondary == curdex) { // assuming secondary < max_index, of course.
            output.add(primary, values[curptr]);
        }
    }

    template<class Store>
    void secondary_dimension_below(IDX secondary, size_t primary, size_t index_primary, SecondaryWorkspaceBase& work, Store& output) const {
        auto& curdex = work.current_indices[index_primary];
        auto& curptr = work.current_indptrs[index_primary];

        // Remember that below_indices stores 'indices[curptr - 1] + 1'.
        // So, we only need to do more work if the request is less than this;
        // otherwise the curptr remains unchanged. We also skip if curptr is
        // equal to 'indptrs[index_primary]' because decreasing is impossible.
        auto& prevdex = work.below_indices[index_primary];
        auto limit = indptrs[primary];
        if (secondary < prevdex && curptr > limit) {

            // Having a peek at the index of the non-zero element below us.
            // Maybe we're lucky and the requested index is equal to this,
            // in which case we can use it directly. This would be the case
            // for consecutive accesses in the opposite direction.
            auto candidate = indices[curptr - 1];
            if (candidate == secondary) {
                --curptr;
                curdex = candidate;

            } else if (secondary == 0) {
                // Special case if the requested index is at the start of the matrix,
                // in which case we can just jump there directly rather than 
                // doing an unnecessary binary search. 
                curdex = indices[limit];
                curptr = limit;

            } else {
                // Otherwise we need to search. Note that there's no need to
                // handle candidate < secondary, as that would imply that
                // 'secondary >= prevdex', which isn't possible.
                curptr = std::lower_bound(indices.begin() + limit, indices.begin() + curptr - 1, secondary) - indices.begin();
                curdex = indices[curptr]; // guaranteed to be valid as curptr - 1 >= limit.
            }

            prevdex = define_below_index(primary, curptr);
        }

        if (secondary == curdex) { // assuming i < max_index, of course.
            output.add(primary, values[curptr]);
        }
    }

    template<class STORE>
    void secondary_dimension(IDX i, size_t start, size_t length, SecondaryWorkspaceBase& work, STORE& output) const {
        IDX prev_i = work.previous_request;

        if (i >= prev_i) {
            for (size_t current = 0; current < length; ++current) {
                secondary_dimension_above(i, current + start, current, work, output);
            }
        } else {
            // Backfilling everything so that all uses of 'below_indices' are valid.
            if (!work.store_below) {
                work.store_below = true;
                work.below_indices.resize(length);
                for (size_t j = 0; j < length; ++j) {
                    auto secondary = start + j;
                    work.below_indices[j] = define_below_index(secondary, work.current_indptrs[j]);
                }
            }

            for (size_t current = 0; current < length; ++current) {
                secondary_dimension_below(i, current + start, current, work, output);
            }
        }

        work.previous_request = i;
        return;
    }

    struct raw_store {
        T* out_values;
        IDX* out_indices;
        size_t n = 0;
        void add(IDX i, T val) {
            ++n;
            *out_indices = i;
            ++out_indices;
            *out_values = val;
            ++out_values;
            return;
        }
    };

    SparseRange<T, IDX> secondary_dimension_raw(IDX i, size_t start, size_t length, SecondaryWorkspaceBase& work, T* out_values, IDX* out_indices) const {
        raw_store store;
        store.out_values = out_values;
        store.out_indices = out_indices;
        secondary_dimension(i, start, length, work, store);
        return SparseRange<T, IDX>(store.n, out_values, out_indices);
    }

    struct expanded_store {
        T* out_values;
        size_t first;
        void add(size_t i, T val) {
            out_values[i - first] = val;
            return;
        }
    };

    void secondary_dimension_expanded(IDX i, size_t start, size_t length, SecondaryWorkspaceBase& work, T* out_values) const {
        std::fill(out_values, out_values + length, static_cast<T>(0));
        expanded_store store;
        store.out_values = out_values;
        store.first = start;
        secondary_dimension(i, start, length, work, store);
        return;
    }

public:
    /**
     * @cond
     */
    struct CompressedSparsePrimaryIndexWorkspace : public IndexWorkspace<IDX, ROW> {
        CompressedSparsePrimaryIndexWorkspace(size_t length, const IDX* subset) : IndexWorkspace<IDX, ROW>(length, subset) {}
    };

    struct CompressedSparseSecondaryIndexWorkspace : public IndexWorkspace<IDX, !ROW> {
        CompressedSparseSecondaryIndexWorkspace(size_t length, const IDX* subset, size_t max_index, const V& idx, const W& idp) : IndexWorkspace<IDX, !ROW>(length, subset), core(max_index, idx, idp, length, subset) {}

        SecondaryWorkspaceBase core;
    };
    /**
     * @endcond
     */

    std::shared_ptr<RowIndexWorkspace<IDX> > new_row_workspace(size_t length, const IDX* subset) const { 
        if constexpr(ROW) {
            return std::shared_ptr<RowIndexWorkspace<IDX> >(new CompressedSparsePrimaryIndexWorkspace(length, subset));
        } else {
            return std::shared_ptr<RowIndexWorkspace<IDX> >(new CompressedSparseSecondaryIndexWorkspace(length, subset, nrows, indices, indptrs));
        }
    }

    std::shared_ptr<ColumnIndexWorkspace<IDX> > new_column_workspace(size_t length, const IDX* subset) const { 
        if constexpr(ROW) {
            return std::shared_ptr<ColumnIndexWorkspace<IDX> >(new CompressedSparseSecondaryIndexWorkspace(length, subset, ncols, indices, indptrs));
        } else {
            return std::shared_ptr<ColumnIndexWorkspace<IDX> >(new CompressedSparsePrimaryIndexWorkspace(length, subset));
        }
    }

    const T* row(size_t r, T* buffer, RowIndexWorkspace<IDX>* work) const {
        if constexpr(ROW) {
            primary_dimension_expanded(r, work->length, work->indices, ncols, buffer);
        } else {
            secondary_dimension_expanded(r, work->length, work->indices, static_cast<CompressedSparseSecondaryIndexWorkspace*>(work)->core, buffer);
        }
        return buffer;
    }

    const T* column(size_t c, T* buffer, ColumnIndexWorkspace<IDX>* work) const {
        if constexpr(ROW) {
            secondary_dimension_expanded(c, work->length, work->indices, static_cast<CompressedSparseSecondaryIndexWorkspace*>(work)->core, buffer);
        } else {
            primary_dimension_expanded(c, work->length, work->indices, nrows, buffer);
        }
        return buffer;
    }

    SparseRange<T, IDX> sparse_row(size_t r, T* vbuffer, IDX* ibuffer, RowIndexWorkspace<IDX>* work) const {
        if constexpr(ROW) {
            return primary_dimension_raw(r, work->length, work->indices, ncols, vbuffer, ibuffer);
        } else {
            return secondary_dimension_raw(r, work->length, work->indices, static_cast<CompressedSparseSecondaryIndexWorkspace*>(work)->core, vbuffer, ibuffer);
        }
    }

    SparseRange<T, IDX> sparse_column(size_t c, T* vbuffer, IDX* ibuffer, ColumnIndexWorkspace<IDX>* work) const {
        if constexpr(ROW) {
            return secondary_dimension_raw(c, work->length, work->indices, static_cast<CompressedSparseSecondaryIndexWorkspace*>(work)->core, vbuffer, ibuffer);
        } else {
            return primary_dimension_raw(c, work->length, work->indices, nrows, vbuffer, ibuffer);
        }
    }

private:
    template<class Store>
    void primary_dimension(size_t i, size_t length, const IDX* subset, Store& store) const {
        const auto pstart = indptrs[i];
        auto iIt = indices.begin() + pstart, 
             eIt = indices.begin() + indptrs[i + 1]; 
        auto xIt = values.begin() + pstart;

        if (length && subset[0]) { // Jumping ahead if non-zero.
            iIt = std::lower_bound(iIt, eIt, subset[0]);
        } 
        if (iIt == eIt) {
            return;
        }

        size_t counter = 0;
        while (counter < length) {
            auto current = subset[counter];

            while (current > *iIt) {
                ++iIt;
                ++xIt;
            }
            if (iIt == eIt) {
                break;
            }

            if (current == *iIt) {
                store.add(current, *xIt);
            }
            ++counter;
        }

        return;
    }

    SparseRange<T, IDX> primary_dimension_raw(size_t i, size_t length, const IDX* subset, size_t otherdim, T* out_values, IDX* out_indices) const {
        raw_store store;
        store.out_values = out_values;
        store.out_indices = out_indices;
        primary_dimension(i, length, subset, store);
        return SparseRange<T, IDX>(store.n, out_values, out_indices);
    }

    struct expanded_store2 {
        T* out_values;
        void add(size_t i, T val) {
            *out_values = val;
            ++out_values;
            return;
        }
    };

    void primary_dimension_expanded(size_t i, size_t length, const IDX* subset, size_t otherdim, T* out_values) const {
        std::fill(out_values, out_values + length, static_cast<T>(0));
        expanded_store2 store;
        store.out_values = out_values;
        primary_dimension(i, length, subset, store);
        return;
    }

private:
    template<class STORE>
    void secondary_dimension(IDX i, size_t length, const IDX* subset, SecondaryWorkspaceBase& work, STORE& output) const {
        IDX prev_i = work.previous_request;

        for (size_t current = 0; current < length; ++current) {
            if (i > prev_i) {
                secondary_dimension_above(i, subset[current], current, work, output);

            } else if (i < prev_i) {
                // Backfilling everything so that all uses of 'below_indices' are valid.
                if (!work.store_below) {
                    work.store_below = true;
                    work.below_indices.resize(length);
                    for (size_t j = 0; j < length; ++j) {
                        work.below_indices[j] = define_below_index(subset[j], work.current_indptrs[subset[j]]);
                    }
                }

                secondary_dimension_below(i, subset[current], current, work, output);
            }
        }

        work.previous_request = i;
        return;
    }

    SparseRange<T, IDX> secondary_dimension_raw(IDX i, size_t length, const IDX* subset, SecondaryWorkspaceBase& work, T* out_values, IDX* out_indices) const {
        raw_store store;
        store.out_values = out_values;
        store.out_indices = out_indices;
        secondary_dimension(i, length, subset, work, store);
        return SparseRange<T, IDX>(store.n, out_values, out_indices);
    }

    void secondary_dimension_expanded(IDX i, size_t length, const IDX* subset, SecondaryWorkspaceBase& work, T* out_values) const {
        std::fill(out_values, out_values + length, static_cast<T>(0));
        expanded_store2 store;
        store.out_values = out_values;
        secondary_dimension(i, length, subset, work, store);
        return;
    }
};

/**
 * Compressed sparse column matrix.
 * See `tatami::CompressedSparseMatrix` for details on the template parameters.
 */
template<typename T, typename IDX, class U = std::vector<T>, class V = std::vector<IDX>, class W = std::vector<size_t> >
using CompressedSparseColumnMatrix = CompressedSparseMatrix<false, T, IDX, U, V, W>;

/**
 * Compressed sparse row matrix.
 * See `tatami::CompressedSparseMatrix` for details on the template parameters.
 */
template<typename T, typename IDX, class U = std::vector<T>, class V = std::vector<IDX>, class W = std::vector<size_t> >
using CompressedSparseRowMatrix = CompressedSparseMatrix<true, T, IDX, U, V, W>;

}

#endif
