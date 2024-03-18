#ifndef TATAMI_DELAYED_UNARY_ISOMETRIC_OP_H
#define TATAMI_DELAYED_UNARY_ISOMETRIC_OP_H

#include "../../base/Matrix.hpp"
#include "../../utils/copy.hpp"
#include "../../utils/new_extractor.hpp"

#include <memory>
#include <algorithm>
#include <vector>

/**
 * @file DelayedUnaryIsometricOp.hpp
 *
 * @brief Delayed unary isometric operations.
 *
 * This is equivalent to the class of the same name in the **DelayedArray** package.
 */

namespace tatami {

/**
 * @cond
 */
namespace DelayedUnaryIsometricOp_internal {

/**
 * DenseBasic is used if:
 *
 * - the underlying matrix is dense.
 *
 * OR
 *
 * - the underlying matrix is sparse
 * - the operation discards sparsity in a variable manner.
 */
template<bool oracle_, typename Value_, typename Index_, class Operation_>
struct DenseBasicFull : public DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    DenseBasicFull(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        const Options& opt) :
        operation(op),
        row(row),
        extent(row ? p->ncol() : p->nrow()),
        internal(new_extractor<false, oracle_>(p, row, std::move(oracle), opt))
    {}

private:
    const Operation_& operation;
    bool row;
    Index_ extent;
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > internal;

public:
    const Value_* fetch(Index_ i, Value_* buffer) {
        auto ptr = internal->fetch(i, buffer);
        copy_n(ptr, extent, buffer);
        operation.dense(row, i, 0, extent, buffer);
        return buffer;
    }
};

template<bool oracle_, typename Value_, typename Index_, class Operation_>
struct DenseBasicBlock : public DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    DenseBasicBlock(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        Index_ bs,
        Index_ bl,
        const Options& opt) :
        operation(op),
        row(row),
        block_start(bs),
        block_length(bl),
        internal(new_extractor<false, oracle_>(p, row, std::move(oracle), bs, bl, opt))
    {}

private:
    const Operation_& operation;
    bool row;
    Index_ block_start, block_length;
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > internal;

public:
    const Value_* fetch(Index_ i, Value_* buffer) {
        auto ptr = internal->fetch(i, buffer);
        copy_n(ptr, block_length, buffer);
        operation.dense(row, i, block_start, block_length, buffer);
        return buffer;
    }
};

template<bool oracle_, typename Value_, typename Index_, class Operation_>
struct DenseBasicIndex : public DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    DenseBasicIndex(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        VectorPtr<Index_> idx_ptr,
        const Options& opt) :
        operation(op),
        row(row),
        indices_ptr(idx_ptr),
        internal(new_extractor<false, oracle_>(p, row, std::move(oracle), std::move(idx_ptr), opt))
    {}

private:
    const Operation_& operation;
    bool row;
    VectorPtr<Index_> indices_ptr;
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > internal;

public:
    const Value_* fetch(Index_ i, Value_* buffer) {
        auto ptr = internal->fetch(i, buffer);
        const auto& indices = *indices_ptr;
        copy_n(ptr, indices.size(), buffer);
        operation.dense(row, i, indices, buffer);
        return buffer;
    }
};

/**
 * DenseExpanded is used if:
 *
 * - the underlying matrix is sparse
 * - the operation preserves sparsity
 * 
 * OR
 *
 * - the underlying matrix is sparse
 * - the operation discards sparsity in a constant manner.
 */
template<bool oracle_, typename Value_, typename Index_, class Operation_> 
struct DenseExpandedFull : public DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    DenseExpandedFull(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op,
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        Options opt) :
        operation(op),
        row(row),
        extent(row ? p->ncol() : p->nrow()),
        internal_vbuffer(extent),
        internal_ibuffer(extent)
    {
        opt.sparse_extract_value = true;
        opt.sparse_extract_index = true;
        internal = new_extractor<true, oracle_>(p, row, std::move(oracle), opt);
    }

private:
    const Operation_& operation;
    bool row;
    Index_ extent;
    std::vector<Value_> internal_vbuffer;
    std::vector<Index_> internal_ibuffer;
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > internal;

public:
    const Value_* fetch(Index_ i, Value_* buffer) {
        auto vbuffer = internal_vbuffer.data();
        auto range = this->internal->fetch(i, vbuffer, internal_ibuffer.data());
        operation.sparse(row, i, range.number, vbuffer, range.index);

        // avoid calling zero() if possible, as this might throw zero-related errors in non-IEEE platforms.
        if (range.number < extent) { 
            std::fill_n(buffer, extent, [&]() {
                if constexpr(Operation_::is_sparse) {
                    return static_cast<Value_>(0);
                } else {
                    return operation.template fill<Value_>(i);
                }
            }());
        }

        for (Index_ i = 0; i < range.number; ++i) {
            buffer[range.index[i]] = vbuffer[i];
        }

        return buffer;
    }
};

template<bool oracle_, typename Value_, typename Index_, class Operation_> 
struct DenseExpandedBlock : public DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    DenseExpandedBlock(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op,
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        Index_ bs,
        Index_ bl,
        Options opt) :
        operation(op),
        row(row),
        block_start(bs),
        block_length(bl),
        internal_vbuffer(block_length),
        internal_ibuffer(block_length)
    {
        opt.sparse_extract_value = true;
        opt.sparse_extract_index = true;
        internal = new_extractor<true, oracle_>(p, row, std::move(oracle), block_start, block_length, opt);
    }

private:
    const Operation_& operation;
    bool row;
    Index_ block_start, block_length;
    std::vector<Value_> internal_vbuffer;
    std::vector<Index_> internal_ibuffer;
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > internal;

public:
    const Value_* fetch(Index_ i, Value_* buffer) {
        auto vbuffer = internal_vbuffer.data();
        auto range = this->internal->fetch(i, vbuffer, internal_ibuffer.data());
        operation.sparse(row, i, range.number, vbuffer, range.index);

        // avoid calling zero() if possible, as this might throw zero-related errors in non-IEEE platforms.
        if (range.number < block_length) { 
            std::fill_n(buffer, block_length, [&]() {
                if constexpr(Operation_::is_sparse) {
                    return static_cast<Value_>(0);
                } else {
                    return operation.template fill<Value_>(i);
                }
            }());
        }

        for (Index_ i = 0; i < range.number; ++i) {
            buffer[range.index[i] - block_start] = vbuffer[i];
        }

        return buffer;
    }
};

template<bool oracle_, typename Value_, typename Index_, class Operation_> 
struct DenseExpandedIndex : public DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    DenseExpandedIndex(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op,
        bool row,
        MaybeOracle<oracle_, Index_> oracle,
        VectorPtr<Index_> indices_ptr,
        Options opt) :
        operation(op),
        row(row)
    {
        opt.sparse_extract_value = true;
        opt.sparse_extract_index = true;

        const auto& indices = *indices_ptr;
        extent = indices.size();
        internal_vbuffer.resize(extent);
        internal_ibuffer.resize(extent);

        if (extent) {
            index_mapping.resize(row ? p->ncol() : p->nrow());
            for (Index_ i = 0; i < extent; ++i) {
                index_mapping[indices[i]] = i;
            }
        }

        internal = new_extractor<true, oracle_>(p, row, std::move(oracle), std::move(indices_ptr), opt);
    }

private:
    const Operation_& operation;
    bool row;
    Index_ extent;
    std::vector<Value_> internal_vbuffer;
    std::vector<Index_> internal_ibuffer;
    std::vector<Index_> index_mapping;
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > internal;

public:
    const Value_* fetch(Index_ i, Value_* buffer) {
        auto vbuffer = internal_vbuffer.data();
        auto range = this->internal->fetch(i, vbuffer, internal_ibuffer.data());
        operation.sparse(row, i, range.number, vbuffer, range.index);

        // avoid calling zero() if possible, as this might throw zero-related errors in non-IEEE platforms.
        if (range.number < extent) { 
            std::fill_n(buffer, extent, [&]() {
                if constexpr(Operation_::is_sparse) {
                    return static_cast<Value_>(0);
                } else {
                    return operation.template fill<Value_>(i);
                }
            }());
        }

        for (Index_ i = 0; i < range.number; ++i) {
            buffer[index_mapping[range.index[i]]] = vbuffer[i];
        }

        return buffer;
    }
};

/**
 * SparseSimple is used if:
 *
 * - the underlying matrix is sparse
 * - the operation preserves sparsity
 * - indices are not necessary to perform the operation 
 */
template<bool oracle_, typename Value_, typename Index_, class Operation_>
struct SparseSimple : public SparseExtractor<oracle_, Value_, Index_> {
    SparseSimple(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        const Options& opt) :
        operation(op),
        row(row),
        internal(new_extractor<true, oracle_>(p, row, std::move(oracle), opt)) 
    {}

    SparseSimple(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        Index_ bs,
        Index_ bl,
        const Options& opt) :
        operation(op),
        row(row),
        internal(new_extractor<true, oracle_>(p, row, std::move(oracle), bs, bl, opt)) 
    {}

    SparseSimple(
        const Matrix<Value_, Index_>* p, 
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        VectorPtr<Index_> indices_ptr,
        const Options& opt) :
        operation(op),
        row(row),
        internal(new_extractor<true, oracle_>(p, row, std::move(oracle), std::move(indices_ptr), opt)) 
    {}

private:
    const Operation_& operation;
    bool row;
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > internal;

public:
    SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto raw = internal->fetch(i, vbuffer, ibuffer);
        if (raw.value) {
            copy_n(raw.value, raw.number, vbuffer);
            operation.sparse(row, i, raw.number, vbuffer, raw.index);
            raw.value = vbuffer;
        }
        return raw;
    }
};

/**
 * SparseNeedsIndices is used if:
 *
 * - the underlying matrix is sparse
 * - the operation preserves sparsity
 * - indices are necessary to perform the operation 
 */
template<bool oracle_, typename Value_, typename Index_, class Operation_>
struct SparseNeedsIndices : public SparseExtractor<oracle_, Value_, Index_> {
    SparseNeedsIndices(
        const Matrix<Value_, Index_>* p,
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        Options opt) :
        operation(op),
        row(row)
    {
        initialize(opt, row ? p->ncol() : p->nrow());
        internal = new_extractor<true, oracle_>(p, row, std::move(oracle), opt);
    }

    SparseNeedsIndices(
        const Matrix<Value_, Index_>* p,
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        Index_ bs,
        Index_ bl,
        Options opt) :
        operation(op),
        row(row)
    {
        initialize(opt, bl);
        internal = new_extractor<true, oracle_>(p, row, std::move(oracle), bs, bl, opt);
    }

    SparseNeedsIndices(
        const Matrix<Value_, Index_>* p,
        const Operation_& op, 
        bool row, 
        MaybeOracle<oracle_, Index_> oracle, 
        VectorPtr<Index_> indices_ptr,
        Options opt) :
        operation(op),
        row(row)
    {
        initialize(opt, indices_ptr->size());
        internal = new_extractor<true, oracle_>(p, row, std::move(oracle), std::move(indices_ptr), opt);
    }

private:
    void initialize(Options& opt, size_t extent) {
        report_value = opt.sparse_extract_value;
        report_index = opt.sparse_extract_index;

        // The index is not required if we don't even want the values,
        // in which case Operation_::sparse() isn't even called.
        if (report_value) {
            opt.sparse_extract_index = true;

            // We only need an internal ibuffer if the user wants the
            // values but didn't provide enough space to store the indices
            // (which we need to pass to Operation_::sparse()).
            if (!report_index) {
                internal_ibuffer.resize(extent);
            }
        }
    }

private:
    const Operation_& operation;
    bool row;
    bool report_value, report_index;
    std::vector<Index_> internal_ibuffer;
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > internal;

public:
    SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto raw = internal->fetch(i, vbuffer, report_index ? ibuffer : internal_ibuffer.data());

        if (report_value) {
            copy_n(raw.value, raw.number, vbuffer);
            operation.sparse(row, i, raw.number, vbuffer, raw.index);
            raw.value = vbuffer;
        }

        if (!report_index) {
            raw.index = NULL;
        } 

        return raw;
    }
};

}
/**
 * @endcond
 */

/**
 * @brief Delayed isometric operations on a single matrix.
 *
 * Implements any operation that preserves the shape of the matrix and operates on each matrix value independently.
 * This operation is "delayed" in that it is only evaluated on request, e.g., with `DenseExtractor::fetch()` or friends.
 * We only consider "unary" operations that involve a single `Matrix` - see `DelayedBinaryIsometricOp` for operations between two `Matrix` instances.
 * 
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 * @tparam Operation_ Class implementing the operation.
 * A non-sparsity-preserving operation should implement the same methods as `DelayedUnaryMockVariableDenseHelper`
 * or `DelayedUnaryMockConstantDenseHelper` (depending on whether the conversion of zeros to non-zero values is constant),
 * while a sparsity-preserving operation should implement the same methods as `DelayedUnaryMockSparseHelper`.
 */
template<typename Value_, typename Index_, class Operation_>
class DelayedUnaryIsometricOp : public Matrix<Value_, Index_> {
public:
    /**
     * @param p Pointer to the underlying matrix.
     * @param op Instance of the functor class.
     */
    DelayedUnaryIsometricOp(std::shared_ptr<const Matrix<Value_, Index_> > p, Operation_ op) : mat(std::move(p)), operation(std::move(op)) {}

private:
    std::shared_ptr<const Matrix<Value_, Index_> > mat;
    Operation_ operation;

public:
    Index_ nrow() const {
        return mat->nrow();
    }
    
    Index_ ncol() const {
        return mat->ncol();
    }

    /**
     * @return `true` if both the underlying (pre-operation) matrix is sparse and the operation preserves sparsity.
     * Otherwise returns `false`.
     */
    bool sparse() const {
        if constexpr(Operation_::is_sparse) {
            return mat->sparse();
        }
        return false;
    }

    double sparse_proportion() const {
        if constexpr(Operation_::is_sparse) {
            return mat->sparse_proportion();
        } else {
            return 0;
        }
    }

    bool prefer_rows() const { 
        return mat->prefer_rows();
    }

    double prefer_rows_proportion() const { 
        return mat->prefer_rows_proportion();
    }

    bool uses_oracle(bool row) const {
        return mat->uses_oracle(row);
    }

    using Matrix<Value_, Index_>::dense_row;

    using Matrix<Value_, Index_>::dense_column;

    using Matrix<Value_, Index_>::sparse_row;

    using Matrix<Value_, Index_>::sparse_column;

    /********************
     *** Myopic dense ***
     ********************/
private:
    template<bool oracle_, typename ... Args_>
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > dense_basic_internal(bool row, MaybeOracle<oracle_, Index_> oracle, const Options& opt) const {
        return std::make_unique<DelayedUnaryIsometricOp_internal::DenseBasicFull<oracle_, Value_, Index_, Operation_> >(mat.get(), operation, row, std::move(oracle), opt);
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > dense_basic_internal(bool row, MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        return std::make_unique<DelayedUnaryIsometricOp_internal::DenseBasicBlock<oracle_, Value_, Index_, Operation_> >(mat.get(), operation, row, std::move(oracle), block_start, block_length, opt);
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > dense_basic_internal(bool row, MaybeOracle<oracle_, Index_> oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return std::make_unique<DelayedUnaryIsometricOp_internal::DenseBasicIndex<oracle_, Value_, Index_, Operation_> >(mat.get(), operation, row, std::move(oracle), std::move(indices_ptr), opt);
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > dense_expanded_internal(bool row, MaybeOracle<oracle_, Index_> oracle, const Options& opt) const {
        return std::make_unique<DelayedUnaryIsometricOp_internal::DenseExpandedFull<oracle_, Value_, Index_, Operation_> >(mat.get(), operation, row, std::move(oracle), opt);
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > dense_expanded_internal(bool row, MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        return std::make_unique<DelayedUnaryIsometricOp_internal::DenseExpandedBlock<oracle_, Value_, Index_, Operation_> >(mat.get(), operation, row, std::move(oracle), block_start, block_length, opt);
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > dense_expanded_internal(bool row, MaybeOracle<oracle_, Index_> oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return std::make_unique<DelayedUnaryIsometricOp_internal::DenseExpandedIndex<oracle_, Value_, Index_, Operation_> >(mat.get(), operation, row, std::move(oracle), std::move(indices_ptr), opt);
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<DenseExtractor<oracle_, Value_, Index_> > dense_internal(bool row, Args_&& ... args) const {
        if (mat->sparse()) {
            // Hide sparse->dense expanded code behind constexpr(), because
            // non-sparse operation classes won't define sparse().
            if constexpr(Operation_::is_sparse) {
                return dense_expanded_internal<oracle_>(row, std::forward<Args_>(args)...);
            } else {
                if constexpr(!Operation_::zero_depends_on_row || !Operation_::zero_depends_on_column) {
                    // If we don't depend on the rows, then we don't need row indices when 'row = false'.
                    // Similarly, if we don't depend on columns, then we don't column row indices when 'row = true'.
                    if ((!Operation_::zero_depends_on_row && !row) || (!Operation_::zero_depends_on_column && row)) {
                        return dense_expanded_internal<oracle_>(row, std::forward<Args_>(args)...);
                    }
                }
            }
        }

        return dense_basic_internal<oracle_>(row, std::forward<Args_>(args)...);
    }

public:
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, const Options& opt) const {
        return dense_internal<false>(row, false, opt);
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const Options& opt) const {
        return dense_internal<false>(row, false, block_start, block_length, opt);
    }
    
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return dense_internal<false>(row, false, std::move(indices_ptr), opt);
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    template<bool oracle_, typename ... Args_>
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > sparse_to_dense_internal(bool row, MaybeOracle<oracle_, Index_> oracle, const Options& opt) const {
        return std::make_unique<FullSparsifiedWrapper<oracle_, Value_, Index_> >(
            dense_internal<oracle_>(row, std::move(oracle), opt),
            (row ? mat->ncol() : mat->nrow()),
            opt
        );
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > sparse_to_dense_internal(bool row, MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        return std::make_unique<BlockSparsifiedWrapper<oracle_, Value_, Index_> >(
            dense_internal<oracle_>(row, std::move(oracle), block_start, block_length, opt),
            block_start,
            block_length,
            opt
        );
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > sparse_to_dense_internal(bool row, MaybeOracle<oracle_, Index_> oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return std::make_unique<IndexSparsifiedWrapper<oracle_, Value_, Index_> >(
            dense_internal<oracle_>(row, std::move(oracle), indices_ptr, opt),
            indices_ptr,
            opt
        );
    }

    template<bool oracle_, typename ... Args_>
    std::unique_ptr<SparseExtractor<oracle_, Value_, Index_> > sparse_internal(bool row, MaybeOracle<oracle_, Index_> oracle, Args_&& ... args) const {
        if (mat->sparse()) { 
            // Hide sparse code behind constexpr(), because non-sparse
            // operation classes won't define sparse().
            if constexpr(Operation_::is_sparse) {
                if constexpr(!Operation_::depends_on_row || !Operation_::depends_on_column) {
                    if ((!Operation_::depends_on_row && !row) || (!Operation_::depends_on_column && row)) {
                        // If we don't depend on the rows, then we don't need row indices when 'row = false'.
                        // Similarly, if we don't depend on columsns, then we don't column row indices when 'row = true'.
                        return std::make_unique<DelayedUnaryIsometricOp_internal::SparseSimple<oracle_, Value_, Index_, Operation_> >(mat.get(), operation, row, std::move(oracle), std::forward<Args_>(args)...);
                    }
                }

                return std::make_unique<DelayedUnaryIsometricOp_internal::SparseNeedsIndices<oracle_, Value_, Index_, Operation_> >(mat.get(), operation, row, std::move(oracle), std::forward<Args_>(args)...);
            }
        }

        return sparse_to_dense_internal<oracle_>(row, std::move(oracle), std::forward<Args_>(args)...);
    }

public:
    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const Options& opt) const {
        return sparse_internal<false>(row, false, opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const Options& opt) const {
        return sparse_internal<false>(row, false, block_start, block_length, opt);
    }
    
    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return sparse_internal<false>(row, false, std::move(indices_ptr), opt);
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, const Options& opt) const {
        return dense_internal<true>(row, std::move(oracle), opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        return dense_internal<true>(row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return dense_internal<true>(row, std::move(oracle), std::move(indices_ptr), opt);
    }

    /***********************
     *** Oracular sparse ***
     ***********************/
public:
    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, const Options& opt) const {
        return sparse_internal<true>(row, std::move(oracle), opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        return sparse_internal<true>(row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return sparse_internal<true>(row, std::move(oracle), std::move(indices_ptr), opt);
    }
};

/**
 * A `make_*` helper function to enable partial template deduction of supplied types.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 * @tparam Operation_ Helper class defining the operation.
 *
 * @param p Pointer to a (possibly `const`) `Matrix`.
 * @param op Instance of the operation helper class.
 *
 * @return Instance of a `DelayedUnaryIsometricOp` clas.
 */
template<typename Value_, typename Index_, class Operation_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedUnaryIsometricOp(std::shared_ptr<const Matrix<Value_, Index_> > p, Operation_ op) {
    typedef typename std::remove_reference<Operation_>::type Op_;
    return std::shared_ptr<Matrix<Value_, Index_> >(new DelayedUnaryIsometricOp<Value_, Index_, Op_>(std::move(p), std::move(op)));
}

/**
 * @cond
 */
// For automatic template deduction with non-const pointers.
template<typename Value_, typename Index_, class Operation_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedUnaryIsometricOp(std::shared_ptr<Matrix<Value_, Index_> > p, Operation_ op) {
    typedef typename std::remove_reference<Operation_>::type Op_;
    return std::shared_ptr<Matrix<Value_, Index_> >(new DelayedUnaryIsometricOp<Value_, Index_, Op_>(std::move(p), std::move(op)));
}

// For back-compatibility.
template<typename ... Args_>
auto make_DelayedUnaryIsometricOp(Args_&&... args) {
    return make_DelayedUnaryIsometricOp(std::forward<Args_>(args)...);
}

template<typename Value_, typename Index_, class Operation_>
using DelayedIsometricOp = DelayedUnaryIsometricOp<Value_, Index_, Operation_>;
/**
 * @endcond
 */

}

//#include "arith_helpers.hpp"
//
//#include "math_helpers.hpp"
//
//#include "compare_helpers.hpp"
//
//#include "boolean_helpers.hpp"

#include "mock_helpers.hpp"

#endif
