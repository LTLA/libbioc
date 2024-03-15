#ifndef TATAMI_DELAYED_BINARY_ISOMETRIC_OP_H
#define TATAMI_DELAYED_BINARY_ISOMETRIC_OP_H

#include "../../base/Matrix.hpp"
#include "../../utils/new_extractor.hpp"
#include "../../dense/SparsifiedWrapper.hpp"

#include <memory>

/**
 * @file DelayedBinaryIsometricOp.hpp
 *
 * @brief Delayed binary isometric operations.
 *
 * This is equivalent to the class of the same name in the **DelayedArray** package.
 */

namespace tatami {

/**
 * @cond
 */
namespace DelayedBinaryIsometricOp_internal {

/********************
 *** Myopic dense ***
 ********************/

template<bool accrow_, typename Value_, typename Index_, class Operation_>
class MyopicDenseFull : public MyopicDenseExtractor<Value_, Index_> {
    MyopicDenseFull(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        const Options& opt) :
        left(new_extractor<accrow_, false>(lmat, opt)),
        right(new_extractor<accrow_, false>(rmat, opt)),
        operation(op),
        extent(accrow_ ? lmat->ncol() : lmat->nrow()),
        holding_buffer(extent) 
    {} 

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto ptr = left->fetch(i, buffer);
        copy_n(ptr, extent, buffer);
        auto rptr = right->fetch(i, holding_buffer.data());
        operation.template dense<accrow_>(i, extent, buffer, rptr);
        return buffer;
    }

public:
    Index_ sparsify_full_length() const {
        return extent;
    }

private:
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > left;
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > right;
    const Operation_& operation;
    Index_ extent;
    std::vector<Value_> holding_buffer;
};

template<bool accrow_, typename Value_, typename Index_, class Operation_>
class MyopicDenseBlock : public MyopicDenseExtractor<Value_, Index_> {
    MyopicDenseBlock(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        Index_ bs,
        Index_ bl,
        const Options& opt) :
        left(new_extractor<accrow_, false>(lmat, bs, bl, opt)),
        right(new_extractor<accrow_, false>(rmat, bs, bl, opt)),
        operation(op),
        block_start(bs),
        block_length(bl),
        holding_buffer(block_length)
    {} 

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto ptr = left->fetch(i, buffer);
        copy_n(ptr, block_length, buffer);
        auto rptr = right->fetch(i, holding_buffer.data());
        operation.template dense<accrow_>(i, block_start, block_length, buffer, rptr);
        return buffer;
    }

public:
    Index_ sparsify_block_start() const {
        return block_start;
    }

    Index_ sparsify_block_length() const {
        return block_length;
    }

private:
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > left;
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > right;
    const Operation_& operation;
    Index_ block_start, block_length;
    std::vector<Value_> holding_buffer;
};

template<bool accrow_, typename Value_, typename Index_, class Operation_>
class MyopicDenseIndex : public MyopicDenseExtractor<Value_, Index_> {
    MyopicDenseIndex(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        std::vector<Index_> idx,
        const Options& opt) :
        left(new_extractor<accrow_, false>(lmat, idx, opt)),
        right(new_extractor<accrow_, false>(rmat, idx, opt)),
        operation(op),
        indices(std::move(idx)),
        holding_buffer(indices.size())
    {} 

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto ptr = left->fetch(i, buffer);
        copy_n<Index_>(ptr, indices.size(), buffer);
        auto rptr = right->fetch(i, holding_buffer.data());
        operation.template dense<accrow_>(i, indices, buffer, rptr);
        return buffer;
    }

public:
    const std::vector<Index_>& sparsify_indices() const {
        return indices;
    }

private:
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > left;
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > right;
    const Operation_& operation;
    std::vector<Index_> indices;
    std::vector<Value_> holding_buffer;
};

/**********************
 *** Oracular dense ***
 **********************/

template<typename Index_>
struct OracleManager {
    OracleManager(std::shared_ptr<Oracle<Index_> > ora) : oracle(std::move(ora)) {}
    Index_ operator()() {
        return oracle->get(used++);
    }
private:
    size_t used = 0;
    std::shared_ptr<Oracle<Index_> > oracle;
};

template<bool accrow_, typename Value_, typename Index_, class Operation_>
class OracularDenseFull : public OracularDenseExtractor<Value_, Index_> {
    OracularDenseFull(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        std::shared_ptr<Oracle<Index_> > ora,
        const Options& opt) :
        left(new_extractor<accrow_, false>(lmat, ora, opt)),
        right(new_extractor<accrow_, false>(rmat, ora, opt)),
        operation(op),
        oracle_copy(std::move(ora)),
        extent(row_ ? lmat->ncol() : lmat->nrow()),
        holding_buffer(extent) 
    {} 

    const Value_* fetch(Value_* buffer) {
        auto ptr = left->fetch(buffer);
        copy_n(ptr, extent, buffer);
        auto rptr = right->fetch(holding_buffer.data());
        operation.template dense<accrow_>(oracle_copy(), 0, extent, buffer, rptr);
        return buffer;
    }

public:
    Index_ sparsify_full_length() const {
        return extent;
    }

private:
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > left;
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > right;
    const Operation_& operation;
    OracleManager<Index_> oracle_copy;
    Index_ extent;
    std::vector<Value_> holding_buffer;
};

template<bool accrow_, typename Value_, typename Index_, class Operation_>
class OracularDenseBlock : public OracularDenseExtractor<Value_, Index_> {
    OracularDenseBlock(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        std::shared_ptr<Oracle<Index_> > ora,
        Index_ bs,
        Index_ bl,
        const Options& opt) :
        left(new_extractor<accrow_, false>(lmat, ora, bs, bl, opt)),
        right(new_extractor<accrow_, false>(rmat, ora, bs, bl, opt)),
        operation(op),
        oracle(std::move(ora)),
        block_start(bs),
        block_length(bl),
        holding_buffer(block_length)
    {} 

    const Value_* fetch(Value_* buffer) {
        auto ptr = left->fetch(buffer);
        copy_n(ptr, extent, buffer);
        auto rptr = right->fetch(holding_buffer.data());
        operation.template dense<accrow_>(oracle_copy(), block_start, block_length, buffer, rptr);
        return buffer;
    }

public:
    Index_ sparsify_block_start() const {
        return block_start;
    }

    Index_ sparsify_block_length() const {
        return block_length;
    }

private:
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > left;
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > right;
    const Operation_& operation;
    OracleManager<Index_> oracle_copy;
    Index_ block_start, block_length;
    std::vector<Value_> holding_buffer;
};

template<typename Value_, typename Index_, class Operation_>
class OracularDenseIndex : public OracularDenseExtractor<Value_, Index_> {
    OracularDenseIndex(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        std::shared_ptr<Oracle<Index_> > ora,
        std::vector<Index_> idx,
        const Options& opt) :
        left(new_extractor<accrow_, false>(lmat, ora, idx, opt)),
        right(new_extractor<accrow_, false>(rmat, ora, idx, opt)),
        operation(op),
        oracle_copy(std::move(ora)),
        indices(std::move(idx)),
        holding_buffer(indices.size())
    {} 

    const Value_* fetch(Value_* buffer) {
        auto ptr = left->fetch(buffer);
        copy_n(ptr, static_cast<Index_>(indices.size()), buffer);
        auto rptr = right->fetch(holding_buffer.data());
        operation.template dense<accrow_>(oracle_copy(), indices, buffer, rptr);
        return buffer;
    }

public:
    Index_ sparsify_full_length() const {
        return extent;
    }

private:
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > left;
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > right;
    const Operation_& operation;
    OracleManager<Index_> oracle_copy;
    std::vector<Index_> indices;
    std::vector<Value_> holding_buffer;
};

/*******************************
 *** Myopic sparse (regular) ***
 *******************************/

template<typename Value_, typename Index_, class Operation_>
class MyopicSparseRegular : public MyopicSparseExtractor<Value_, Index_> {
    MyopicSparseSimple(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        Options opt) :
        operation(op),
        report_value(opts.sparse_extract_value),
        report_index(opts.sparse_extract_index)
    {
        allocate_buffers(accrow_ ? lmat->ncol() : lmat->nrow());
        enable_sorted_index(opt);
        left = new_extractor<accrow_, true>(lmat, opt);
        right = new_extractor<accrow_, true>(rmat, opt);
    }

    MyopicSparseSimple(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        Index_ block_start,
        Index_ block_length,
        Options opt) :
        operation(op),
        report_value(opts.sparse_extract_value),
        report_index(opts.sparse_extract_index)
    {
        allocate_buffers(block_length);
        enable_sorted_index(opt);
        left = new_extractor<accrow_, true>(lmat, block_start, block_length, opt);
        right = new_extractor<accrow_, true>(rmat, block_start, block_length, opt);
    }

    MyopicSparseSimple(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        std::vector<Index_> indices,
        Options opt) :
        operation(op),
        report_value(opts.sparse_extract_value),
        report_index(opts.sparse_extract_index)
    {
        allocate_buffers(indices.size());
        enable_sorted_index(opt);
        left = new_extractor<accrow_, true>(lmat, indices, opt);
        right = new_extractor<accrow_, true>(rmat, std::move(indices), opt);
    }

private:
    void allocate_buffers(size_t n) {
        left_internal_ibuffer.resize(n);
        right_internal_ibuffer.resize(n);
        if (report_value) {
            left_internal_vbuffer.resize(n);
            right_internal_vbuffer.resize(n);
        }
    }

    void enable_sorted_index(Options& opt) {
        opt.sparse_extract_index = true;
        opt.sparse_ordered_index = true; 
    }

public:
    SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto left_ranges = left->fetch(i, left_internal_vbuffer.data(), left_internal_ibuffer.data());
        auto right_ranges = right->fetch(i, right_internal_vbuffer.data(), right_internal_ibuffer.data());
        SparseRange<Value_, Index_> output(0, (report_value ? vbuffer : NULL), (report_index ? ibuffer : NULL));
        output.number = operation.template sparse<accrow_>(i, left_ranges, right_ranges, output.value, output.index, report_value, report_index);
        return buffer;
    }

private:
    const Operation_& operation;
    bool report_value = false;
    bool report_index = false;

    std::vector<Value_> left_internal_vbuffer, right_internal_vbuffer;
    std::vector<Index_> left_internal_ibuffer, right_internal_ibuffer;

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > left;
    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > right;
};

/*********************************
 *** Oracular sparse (regular) ***
 *********************************/

template<bool accrow_, typename Value_, typename Index_, class Operation_>
class OracularSparseRegular : public OracularSparseExtractor<Value_, Index_> {
    OracularSparseRegular(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        std::shared_ptr<Oracle<Index_> > ora,
        Options opt) :
        operation(op),
        report_value(opts.sparse_extract_value),
        report_index(opts.sparse_extract_index),
        oracle_copy(ora)
    {
        allocate_buffers(accrow_ ? lmat->ncol() : lmat->nrow());
        enable_sorted_index(opt);
        left = new_extractor<accrow_, true>(lmat, ora, opt);
        right = new_extractor<accrow_, true>(rmat, std::move(ora), opt);
    }

    OracularSparseRegular(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        std::shared_ptr<Oracle<Index_> > ora,
        Index_ block_start,
        Index_ block_length,
        Options opt) :
        operation(op),
        report_value(opts.sparse_extract_value),
        report_index(opts.sparse_extract_index),
        oracle_copy(ora)
    {
        allocate_buffers(block_length);
        enable_sorted_index(opt);
        left = new_extractor<accrow_, true>(lmat, ora, block_start, block_length, opt);
        right = new_extractor<accrow_, true>(rmat, std::move(ora), block_start, block_length, opt);
    }

    OracularSparseRegular(
        const Matrix<Value_, Index_>* lmat,
        const Matrix<Value_, Index_>* rmat,
        const Operation_& op,
        std::shared_ptr<Oracle<Index_> > ora,
        std::vector<Index_> indices,
        Options opt) :
        operation(op),
        report_value(opts.sparse_extract_value),
        report_index(opts.sparse_extract_index),
        oracle_copy(ora)
    {
        allocate_buffers(indices.size());
        enable_sorted_index(opt);
        left = new_extractor<accrow_, true>(lmat, ora, indices, opt);
        right = new_extractor<accrow_, true>(rmat, std::move(ora), std::move(indices), opt);
    }

private:
    void allocate_buffers(size_t n) {
        left_internal_ibuffer.resize(n);
        right_internal_ibuffer.resize(n);
        if (report_value) {
            left_internal_vbuffer.resize(n);
            right_internal_vbuffer.resize(n);
        }
    }

    void enable_sorted_index(Options& opt) {
        opt.sparse_extract_index = true;
        opt.sparse_ordered_index = true; 
    }

public:
    SparseRange<Value_, Index_> fetch(Value_* vbuffer, Index_* ibuffer) {
        auto left_ranges = left->fetch(left_internal_vbuffer.data(), left_internal_ibuffer.data());
        auto right_ranges = right->fetch(right_internal_vbuffer.data(), right_internal_ibuffer.data());
        SparseRange<Value_, Index_> output(0, (report_value ? vbuffer : NULL), (report_index ? ibuffer : NULL));
        output.number = operation.template sparse<accrow_>(oracle_copy(), left_ranges, right_ranges, output.value, output.index, report_value, report_index);
        return buffer;
    }

private:
    const Operation_& operation;
    bool report_value = false;
    bool report_index = false;

    OracleManager<Index_> oracle;

    std::vector<Value_> left_internal_vbuffer, right_internal_vbuffer;
    std::vector<Index_> left_internal_ibuffer, right_internal_ibuffer;
    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > left;
    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > right;
};

}
/**
 * @endcond
 */

/**
 * @brief Delayed isometric operations on two matrices
 *
 * Implements any operation that takes two matrices of the same shape and returns another matrix of that shape.
 * Each entry of the output matrix is a function of the corresponding values in the two input matrices.
 * This operation is "delayed" in that it is only evaluated on request, e.g., with `DenseExtractor::fetch()` or friends.
 *
 * The `Operation_` class is expected to provide the following static `constexpr` member variables:
 *
 * - `always_sparse`: whether the operation can be optimized to return a sparse result if both input matrices are sparse.
 * 
 * The class should implement the following method:
 *
 * - `void dense<row_>(Index_ i, Index_ start, Index_ length, Value_* left_buffer, const Value_* right_buffer) const`: 
 *   This method should apply the operation to corresponding values of `left_buffer` and `right_buffer`,
 *   each of which contain a contiguous block of elements from row `i` of the left and right matrices, respectively (when `row_ = true`).
 *   The result of the operation should be stored in `left_buffer`.
 *   The block starts at column `start` and is of length `length`.
 *   If `row_ = false`, `i` is instead a column and the block starts at row `start`.
 * - `void dense<row_>(Index_ i, const Index_* indices, Index_ length, Value_* buffer1, const Value_* buffer2) const`: 
 *   This method should apply the operation to corresponding values of `left_buffer` and `right_buffer`,
 *   each of which contain a subset of elements from row `i` of the left and right matrices, respectively (when `row_ = true`).
 *   The result of the operation should be stored in `left_buffer`.
 *   The subset is defined by column indices in the `indices` array of length `length`.
 *   If `row_ = false`, `i` is instead a column and `indices` contains rows.
 * 
 * If `always_sparse = true`, the class should implement:
 *
 * - `Index_ sparse<row_, needs_value, needs_index>(Index_ i, const SparseRange<Value_, Index_>& left, const SparseRange<Value_, Index_>& right, Value_* value_buffer, Index_* index_buffer) const`:
 *   This method should apply the operation to the sparse values in `left` and `right`, 
 *   consisting of the contents of row `i` from the left and right matrices, respectively (when `row_ = true`).
 *   All non-zero values resulting from the operation should be stored in `value_buffer` if `needs_value = true`, otherwise `value_buffer = NULL` and should be ignored.
 *   The corresponding indices of those values should be stored in `index_buffer` if `needs_index = true`, otherwise `index_buffer = NULL` and should be ignored.
 *   The return value should be the number of structural non-zero elements in the output buffers.
 *   If `row_ = false`, the contents of `left` and `right` are taken from column `i` instead.
 *   Note that all values in `left` and `right` are already sorted by increasing index.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 * @tparam Operation_ Class implementing the operation.
 */
template<typename Value_, typename Index_, class Operation_>
class DelayedBinaryIsometricOp : public Matrix<Value_, Index_> {
public:
    /**
     * @param l Pointer to the left matrix.
     * @param r Pointer to the right matrix.
     * @param op Instance of the functor class.
     */
    DelayedBinaryIsometricOp(std::shared_ptr<const Matrix<Value_, Index_> > l, std::shared_ptr<const Matrix<Value_, Index_> > r, Operation_ op) : 
        left(std::move(l)), right(std::move(r)), operation(std::move(op)) 
    {
        if (left->nrow() != right->nrow() || left->ncol() != right->ncol()) {
            throw std::runtime_error("shape of the left and right matrices should be the same");
        }

        prefer_rows_proportion_internal = (left->prefer_rows_proportion() + right->prefer_rows_proportion()) / 2;
    }

private:
    std::shared_ptr<const Matrix<Value_, Index_> > left, right;
    Operation_ operation;
    double prefer_rows_proportion_internal;

public:
    Index_ nrow() const {
        return left->nrow();
    }

    Index_ ncol() const {
        return left->ncol();
    }

    /**
     * @return `true` if both underlying (pre-operation) matrices are sparse and the operation preserves sparsity.
     * Otherwise returns `false`.
     */
    bool sparse() const {
        if constexpr(Operation_::always_sparse) {
            return left->sparse() && right->sparse();
        }
        return false;
    }

    double sparse_proportion() const {
        if constexpr(Operation_::always_sparse) {
            // Well, better than nothing.
            return (left->sparse_proportion() + right->sparse_proportion())/2;
        }
        return 0;
    }

    bool prefer_rows() const { 
        return prefer_rows_proportion_internal > 0.5;
    }

    double prefer_rows_proportion() const { 
        return prefer_rows_proportion_internal;
    }

    bool uses_oracle(bool row) const {
        return left->uses_oracle(row) || right->uses_oracle(row);
    }

    using Matrix<Value_, Index_>::dense_row;

    using Matrix<Value_, Index_>::dense_column;

    using Matrix<Value_, Index_>::sparse_row;

    using Matrix<Value_, Index_>::sparse_column;

    /********************
     *** Myopic dense *** 
     ********************/
private:
    template<bool accrow_>
    auto dense_internal(const Options& opt) const {
        return std::make_unique<DelayedIsometricBinaryOp_internal::MyopicDenseFull<accrow_, Value_, Index_, Storage_> >(left.get(), right.get(), operation, opt);
    }

    template<bool accrow_>
    auto dense_internal(Index_ block_start, Index_ block_length, const Options&) const {
        return std::make_unique<DelayedIsometricBinaryOp_internal::MyopicDenseBlock<accrow_, Value_, Index_, Storage_> >(left.get(), right.get(), operation, block_start, block_length, opt);
    }

    template<bool accrow_>
    auto dense_internal(std::vector<Index_> indices, const Options&) const {
        return std::make_unique<DelayedIsometricBinaryOp_internal::MyopicDenseIndex<accrow_, Value_, Index_, Storage_> >(left.get(), right.get(), operation, std::move(indices), opt);
    }

public:
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense_row(const Options& opt) const {
        return dense_internal<true>(opt);
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense_row(Index_ block_start, Index_ block_length, const Options& opt) const {
        return dense_internal<true>(block_start, block_length, opt);
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense_row(std::vector<Index_> indices, const Options& opt) const {
        return dense_internal<true>(std::move(indices), opt);
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense_column(const Options& opt) const {
        return dense_internal<false>(opt);
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense_column(Index_ block_start, Index_ block_length, const Options& opt) const {
        return dense_internal<false>(block_start, block_length, opt);
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense_column(std::vector<Index_> indices, const Options& opt) const {
        return dense_internal<false>(std::move(indices), opt);
    }

    /**********************
     *** Oracular dense *** 
     **********************/
private:
    template<bool accrow_>
    auto dense_internal(std::shared_ptr<Oracle<Index_> > ora, const Options& opt) const {
        return std::make_unique<DelayedIsometricBinaryOp_internal::OracularDenseFull<accrow_, Value_, Index_, Storage_> >(left.get(), right.get(), operation, std::move(ora), opt);
    }

    template<bool accrow_>
    auto dense_internal(std::shared_ptr<Oracle<Index_> > ora, Index_ block_start, Index_ block_length, const Options&) const {
        return std::make_unique<DelayedIsometricBinaryOp_internal::OracularDenseBlock<accrow_, Value_, Index_, Storage_> >(left.get(), right.get(), operation, std::move(ora), block_start, block_length, opt);
    }

    template<bool accrow_>
    auto dense_internal(std::shared_ptr<Oracle<Index_> > ora, std::vector<Index_> indices, const Options&) const {
        return std::make_unique<DelayedIsometricBinaryOp_internal::OracularDenseIndex<accrow_, Value_, Index_, Storage_> >(left.get(), right.get(), operation, std::move(ora), std::move(indices), opt);
    }

public:
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense_row(std::shared_ptr<Oracle<Index_> > ora, const Options& opt) const {
        return dense_internal<true>(std::move(ora), opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense_row(std::shared_ptr<Oracle<Index_> > ora, Index_ block_start, Index_ block_length, const Options& opt) const {
        return dense_internal<true>(std::move(ora), block_start, block_length, opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense_row(std::shared_ptr<Oracle<Index_> > ora, std::vector<Index_> indices, const Options& opt) const {
        return dense_internal<true>(std::move(ora), std::move(indices), opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense_column(std::shared_ptr<Oracle<Index_> > ora, const Options& opt) const {
        return dense_internal<false>(std::move(ora), opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense_column(std::shared_ptr<Oracle<Index_> > ora, Index_ block_start, Index_ block_length, const Options& opt) const {
        return dense_internal<false>(std::move(ora), block_start, block_length, opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense_column(std::shared_ptr<Oracle<Index_> > ora, std::vector<Index_> indices, const Options& opt) const {
        return dense_internal<false>(std::move(ora), std::move(indices), opt);
    }

    /*********************
     *** Myopic sparse *** 
     *********************/
private:
    template<bool accrow_>
    auto sparse_internal(const Options& opt) const {
        if constexpr(Operation_::always_sparse) {
            return std::make_unique<DelayedIsometricBinaryOp_internal::MyopicSparseFull<Value_, Index_, Storage_> >(left.get(), right.get(), operation, opt);
        } else {
            auto ptr = dense_internal<accrow_>(opt);
            return std::make_unique<MyopicSparsifiedWrapper<DimensionSelectionType::FULL, Value_, Index_, typename decltype(ptr)::element_type> >(std::move(*ptr), opt);
        }
    }

    template<bool accrow_>
    auto sparse_internal(Index_ block_start, Index_ block_length, const Options&) const {
        if constexpr(Operation_::always_sparse) {
            return std::make_unique<DelayedIsometricBinaryOp_internal::MyopicSparseBlock<Value_, Index_, Storage_> >(left.get(), right.get(), operation, block_start, block_length, opt);
        } else {
            auto ptr = dense_internal<accrow_>(block_start, block_length, opt);
            return std::make_unique<MyopicSparsifiedWrapper<DimensionSelectionType::BLOCK, Value_, Index_, typename decltype(ptr)::element_type> >(std::move(*ptr), opt);
        }
    }

    template<bool accrow_>
    auto sparse_internal(std::vector<Index_> indices, const Options&) const {
        if constexpr(Operation_::always_sparse) {
            return std::make_unique<DelayedIsometricBinaryOp_internal::MyopicSparseIndex<Value_, Index_, Storage_> >(left.get(), right.get(), operation, std::move(indices), opt);
        } else {
            auto ptr = dense_internal<accrow_>(std::move(indices), opt);
            return std::make_unique<MyopicSparsifiedWrapper<DimensionSelectionType::INDEX, Value_, Index_, typename decltype(ptr)::element_type> >(std::move(*ptr), opt);
        }
    }

public:
    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse_row(const Options& opt) const {
        return sparse_internal<true>(opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse_row(Index_ block_start, Index_ block_length, const Options& opt) const {
        return sparse_internal<true>(block_start, block_length, opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse_row(std::vector<Index_> indices, const Options& opt) const {
        return sparse_internal<true>(std::move(indices), opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse_column(const Options& opt) const {
        return sparse_internal<false>(opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse_column(Index_ block_start, Index_ block_length, const Options& opt) const {
        return sparse_internal<false>(block_start, block_length, opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse_column(std::vector<Index_> indices, const Options& opt) const {
        return sparse_internal<false>(std::move(indices), opt);
    }

    /***********************
     *** Oracular sparse *** 
     ***********************/
private:
    template<bool accrow_>
    auto sparse_internal(std::shared_ptr<Oracle<Index_> > ora, const Options& opt) const {
        if constexpr(Operation_::always_sparse) {
            return std::make_unique<DelayedIsometricBinaryOp_internal::OracularSparseFull<Value_, Index_, Storage_> >(left.get(), right.get(), operation, std::move(ora), opt);
        } else {
            auto ptr = dense_internal<accrow_>(std::move(ora), opt);
            return std::make_unique<MyopicSparsifiedWrapper<DimensionSelectionType::FULL, Value_, Index_, typename decltype(ptr)::element_type> >(std::move(*ptr), opt);
        }
    }

    template<bool accrow_>
    auto sparse_internal(std::shared_ptr<Oracle<Index_> > ora, Index_ block_start, Index_ block_length, const Options&) const {
        if constexpr(Operation_::always_sparse) {
            return std::make_unique<DelayedIsometricBinaryOp_internal::OracularSparseFull<Value_, Index_, Storage_> >(left.get(), right.get(), operation, std::move(ora), block_start, block_length, opt);
        } else {
            auto ptr = dense_internal<accrow_>(std::move(oracle), block_start, block_length, opt);
            return std::make_unique<MyopicSparsifiedWrapper<DimensionSelectionType::BLOCK, Value_, Index_, typename decltype(ptr)::element_type> >(std::move(*ptr), opt);
        }
    }

    template<bool accrow_>
    auto sparse_internal(std::shared_ptr<Oracle<Index_> > ora, std::vector<Index_> indices, const Options&) const {
        if constexpr(Operation_::always_sparse) {
            return std::make_unique<DelayedIsometricBinaryOp_internal::OracularSparseFull<Value_, Index_, Storage_> >(left.get(), right.get(), operation, std::move(ora), std::move(indices), opt);
        } else {
            auto ptr = dense_internal<accrow_>(std::move(ora), std::move(indices), opt);
            return std::make_unique<MyopicSparsifiedWrapper<DimensionSelectionType::INDEX, Value_, Index_, typename decltype(ptr)::element_type> >(std::move(*ptr), opt);
        }
    }

public:
    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse_row(std::shared_ptr<Oracle<Index_> > ora, const Options& opt) const {
        return sparse_internal<true>(std::move(ora), opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse_row(std::shared_ptr<Oracle<Index_> > ora, Index_ block_start, Index_ block_length, const Options& opt) const {
        return sparse_internal<true>(std::move(ora), block_start, block_length, opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse_row(std::shared_ptr<Oracle<Index_> > ora, std::vector<Index_> indices, const Options& opt) const {
        return sparse_internal<true>(std::move(ora), std::move(indices), opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse_column(std::shared_ptr<Oracle<Index_> > ora, const Options& opt) const {
        return sparse_internal<false>(std::move(ora), opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse_column(std::shared_ptr<Oracle<Index_> > ora, Index_ block_start, Index_ block_length, const Options& opt) const {
        return sparse_internal<false>(std::move(ora), block_start, block_length, opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse_column(std::shared_ptr<Oracle<Index_> > ora, std::vector<Index_> indices, const Options& opt) const {
        return sparse_internal<false>(std::move(ora), std::move(indices), opt);
    }
};

/**
 * A `make_*` helper function to enable partial template deduction of supplied types.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 * @tparam Operation_ Helper class defining the operation.
 *
 * @param left Pointer to a (possibly `const`) `Matrix`.
 * @param right Pointer to a (possibly `const`) `Matrix`.
 * @param op Instance of the operation helper class.
 *
 * @return Instance of a `DelayedBinaryIsometricOp` clas.
 */
template<typename Value_, typename Index_, class Operation_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedBinaryIsometricOp(std::shared_ptr<const Matrix<Value_, Index_> > left, std::shared_ptr<const Matrix<Value_, Index_> > right, Operation_ op) {
    typedef typename std::remove_reference<Operation_>::type Op_;
    return std::shared_ptr<Matrix<Value_, Index_> >(new DelayedBinaryIsometricOp<Value_, Index_, Op_>(std::move(left), std::move(right), std::move(op)));
}

/**
 * @cond
 */
// For automatic template deduction with non-const pointers.
template<typename Value_, typename Index_, class Operation_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedBinaryIsometricOp(std::shared_ptr<Matrix<Value_, Index_> > left, std::shared_ptr<Matrix<Value_, Index_> > right, Operation_ op) {
    typedef typename std::remove_reference<Operation_>::type Op_;
    return std::shared_ptr<Matrix<Value_, Index_> >(new DelayedBinaryIsometricOp<Value_, Index_, Op_>(std::move(left), std::move(right), std::move(op)));
}
/**
 * @endcond
 */

}

#include "arith_helpers.hpp"

#include "compare_helpers.hpp"

#include "boolean_helpers.hpp"

#endif
