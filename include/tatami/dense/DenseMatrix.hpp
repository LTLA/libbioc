#ifndef TATAMI_DENSE_MATRIX_H
#define TATAMI_DENSE_MATRIX_H

#include "../base/Matrix.hpp"
#include "SparsifiedWrapper.hpp"
#include "../utils/has_data.hpp"
#include "../utils/PseudoOracularExtractor.hpp"

#include <vector>
#include <algorithm>
#include <stdexcept>

/**
 * @file DenseMatrix.hpp
 *
 * @brief Dense matrix representation.
 *
 * `typedef`s are provided for the usual row- and column-major formats.
 */

namespace tatami {

/**
 * @cond
 */
namespace DenseMatrix_internals {

template<typename Value_, typename Index_, class Storage_>
struct PrimaryMyopicFullDense : public MyopicDenseExtractor<Value_, Index_> {
    PrimaryMyopicFullDense(const Storage_& store, Index_ sec) : storage(store), secondary(sec) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        size_t offset = static_cast<size_t>(i) * secondary; // cast to size_t to avoid overflow of 'Index_'.
        if constexpr(has_data<Value_, Storage_>::value) {
            return storage.data() + offset;
        } else {
            auto it = storage.begin() + offset;
            std::copy_n(it, secondary, buffer);
            return buffer;
        }
    }

private:
    const Storage_& storage;
    size_t secondary;
};

template<typename Value_, typename Index_, class Storage_>
struct PrimaryMyopicBlockDense : public MyopicDenseExtractor<Value_, Index_> {
    PrimaryMyopicBlockDense(const Storage_& store, size_t sec, Index_ bs, Index_ bl) : 
        storage(store), secondary(sec), block_start(bs), block_length(bl) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        size_t offset = static_cast<size_t>(i) * secondary + block_start; // cast to size_t to avoid overflow.
        if constexpr(has_data<Value_, Storage_>::value) {
            return storage.data() + offset;
        } else {
            auto it = storage.begin() + offset;
            std::copy_n(it, block_length, buffer);
            return buffer;
        }
    }

private:
    const Storage_& storage;
    size_t secondary;
    size_t block_start, block_length;
};

template<typename Value_, typename Index_, class Storage_>
struct PrimaryMyopicIndexDense : public MyopicDenseExtractor<Value_, Index_> {
    PrimaryMyopicIndexDense(const Storage_& store, size_t sec, VectorPtr<Index_> idx) : 
        storage(store), secondary(sec), indices_ptr(std::move(idx)) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto copy = buffer;
        size_t offset = static_cast<size_t>(i) * secondary; // cast to size_t avoid overflow.
        for (auto x : *indices_ptr) {
            *copy = storage[offset + static_cast<size_t>(x)]; // more casting for overflow protection.
            ++copy;
        }
        return buffer;
    }

private:
    const Storage_& storage;
    size_t secondary;
    VectorPtr<Index_> indices_ptr;
};

template<typename Value_, typename Index_, class Storage_>
struct SecondaryMyopicFullDense : public MyopicDenseExtractor<Value_, Index_> {
    SecondaryMyopicFullDense(const Storage_& store, Index_ sec, Index_ prim) : 
        storage(store), secondary(sec), primary(prim) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        size_t offset = i; // use size_t to avoid overflow.
        auto copy = buffer;
        for (Index_ x = 0; x < primary; ++x, ++copy, offset += secondary) {
            *copy = storage[offset];
        }
        return buffer;
    }

private:
    const Storage_& storage;
    size_t secondary;
    Index_ primary;
};

template<typename Value_, typename Index_, class Storage_>
struct SecondaryMyopicBlockDense : public MyopicDenseExtractor<Value_, Index_> {
    SecondaryMyopicBlockDense(const Storage_& store, Index_ sec, Index_ bs, Index_ bl) : 
        storage(store), secondary(sec), block_start(bs), block_length(bl) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        size_t offset = block_start * secondary + static_cast<size_t>(i); // cast to avoid overflow.
        auto copy = buffer;
        for (Index_ x = 0; x < block_length; ++x, ++copy, offset += secondary) {
            *copy = storage[offset];
        }
        return buffer;
    }

private:
    const Storage_& storage;
    size_t secondary;
    size_t block_start;
    Index_ block_length;
};

template<typename Value_, typename Index_, class Storage_>
struct SecondaryMyopicIndexDense : public MyopicDenseExtractor<Value_, Index_> {
    SecondaryMyopicIndexDense(const Storage_& store, Index_ sec, VectorPtr<Index_> idx) : 
        storage(store), secondary(sec), indices_ptr(std::move(idx)) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto copy = buffer;
        size_t offset = static_cast<size_t>(i); // cast to avoid overflow.
        for (auto x : *indices_ptr) {
            *copy = storage[static_cast<size_t>(x) * secondary + offset]; // more casting to avoid overflow.
            ++copy;
        }
        return buffer;
    }

private:
    const Storage_& storage;
    size_t secondary;
    VectorPtr<Index_> indices_ptr;
};

}
/**
 * @endcond
 */

/**
 * @brief Dense matrix representation.
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam Storage_ Vector class used to store the matrix values internally.
 * This does not necessarily have to contain `Value_`, as long as the type is convertible to `Value_`.
 * Methods should be available for `size()`, `begin()`, `end()` and `[]`.
 * If a method is available for `data()` that returns a `const Value_*`, it will also be used.
 */
template<typename Value_, typename Index_, class Storage_ = std::vector<Value_> >
class DenseMatrix : public Matrix<Value_, Index_> {
public: 
    /**
     * @param nr Number of rows.
     * @param nc Number of columns.
     * @param vals Vector of values of length equal to the product of `nr` and `nc`.
     * @param row Whether `vals` stores the matrix contents in a row-major representation.
     * If `false`, a column-major representation is assumed instead.
     */
    DenseMatrix(Index_ nr, Index_ nc, Storage_ vals, bool row) : nrows(nr), ncols(nc), values(std::move(vals)), row_major(row) {
        check_dimensions(nr, nc, values.size());
        return;
    }

private: 
    Index_ nrows, ncols;
    Storage_ values;
    bool row_major;

    static void check_dimensions(size_t nr, size_t nc, size_t expected) { // cast to size_t is deliberate to avoid overflow on Index_ on product.
        if (nr * nc != expected) {
            throw std::runtime_error("length of 'values' should be equal to product of 'nrows' and 'ncols'");
        }
    }

public:
    Index_ nrow() const { return nrows; }

    Index_ ncol() const { return ncols; }

    bool prefer_rows() const { return row_major; }

    bool uses_oracle(bool) const { return false; }

    bool sparse() const { return false; }

    double sparse_proportion() const { return 0; }

    double prefer_rows_proportion() const { return static_cast<double>(row_major); }

    using Matrix<Value_, Index_>::dense;

    using Matrix<Value_, Index_>::sparse;

private:
    Index_ primary() const {
        if (row_major) {
            return nrows;
        } else {
            return ncols;
        }
    }

    Index_ secondary() const {
        if (row_major) {
            return ncols;
        } else {
            return nrows;
        }
    }

    /*****************************
     ******* Dense myopic ********
     *****************************/
public:
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, const Options&) const {
        if (row_major == row) {
            return std::make_unique<DenseMatrix_internals::PrimaryMyopicFullDense<Value_, Index_, Storage_> >(values, secondary());
        } else {
            return std::make_unique<DenseMatrix_internals::SecondaryMyopicFullDense<Value_, Index_, Storage_> >(values, secondary(), primary()); 
        }
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const Options&) const {
        if (row_major == row) { 
            return std::make_unique<DenseMatrix_internals::PrimaryMyopicBlockDense<Value_, Index_, Storage_> >(values, secondary(), block_start, block_length);
        } else {
            return std::make_unique<DenseMatrix_internals::SecondaryMyopicBlockDense<Value_, Index_, Storage_> >(values, secondary(), block_start, block_length);
        }
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, VectorPtr<Index_> indices_ptr, const Options&) const {
        if (row_major == row) {
            return std::make_unique<DenseMatrix_internals::PrimaryMyopicIndexDense<Value_, Index_, Storage_> >(values, secondary(), std::move(indices_ptr));
        } else {
            return std::make_unique<DenseMatrix_internals::SecondaryMyopicIndexDense<Value_, Index_, Storage_> >(values, secondary(), std::move(indices_ptr));
        }
    }

    /******************************
     ******* Sparse myopic ********
     ******************************/
public:
    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const Options& opt) const {
        return std::make_unique<FullSparsifiedWrapper<false, Value_, Index_> >(dense(row, opt), (row ? ncols : nrows), opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const Options& opt) const {
        return std::make_unique<BlockSparsifiedWrapper<false, Value_, Index_> >(dense(row, block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        auto ptr = dense(row, indices_ptr, opt);
        return std::make_unique<IndexSparsifiedWrapper<false, Value_, Index_> >(std::move(ptr), std::move(indices_ptr), opt);
    }

    /*******************************
     ******* Dense oracular ********
     *******************************/
public:
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, const Options& opt) const {
        return std::make_unique<PseudoOracularDenseExtractor<Value_, Index_> >(std::move(oracle), dense(row, opt));
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, Index_ block_start, Index_ block_end, const Options& opt) const {
        return std::make_unique<PseudoOracularDenseExtractor<Value_, Index_> >(std::move(oracle), dense(row, block_start, block_end, opt));
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return std::make_unique<PseudoOracularDenseExtractor<Value_, Index_> >(std::move(oracle), dense(row, std::move(indices_ptr), opt));
    }

    /********************************
     ******* Sparse oracular ********
     ********************************/
public:
    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, const Options& opt) const {
        return std::make_unique<PseudoOracularSparseExtractor<Value_, Index_> >(std::move(oracle), sparse(row, opt));
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, Index_ block_start, Index_ block_end, const Options& opt) const {
        return std::make_unique<PseudoOracularSparseExtractor<Value_, Index_> >(std::move(oracle), sparse(row, block_start, block_end, opt));
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return std::make_unique<PseudoOracularSparseExtractor<Value_, Index_> >(std::move(oracle), sparse(row, std::move(indices_ptr), opt));
    }
};

/**
 * Column-major matrix.
 * See `tatami::DenseMatrix` for details on the template parameters.
 */
template<typename Value_, typename Index_, class Storage_ = std::vector<Value_> >
class DenseColumnMatrix : public DenseMatrix<Value_, Index_, Storage_> {
public:
    /**
     * @param nr Number of rows.
     * @param nc Number of columns.
     * @param vals Vector of values of length equal to the product of `nr` and `nc`, storing the matrix in column-major format.
     */
    DenseColumnMatrix(Index_ nr, Index_ nc, Storage_ vals) : DenseMatrix<Value_, Index_, Storage_>(nr, nc, std::move(vals), false) {}
};

/**
 * Row-major matrix.
 * See `tatami::DenseMatrix` for details on the template parameters.
 */
template<typename Value_, typename Index_, class Storage_ = std::vector<Value_> >
class DenseRowMatrix : public DenseMatrix<Value_, Index_, Storage_> {
public:
    /**
     * @param nr Number of rows.
     * @param nc Number of columns.
     * @param vals Vector of values of length equal to the product of `nr` and `nc`, storing the matrix in row-major format.
     */
    DenseRowMatrix(Index_ nr, Index_ nc, Storage_ vals) : DenseMatrix<Value_, Index_, Storage_>(nr, nc, std::move(vals), true) {}
};

}

#endif
