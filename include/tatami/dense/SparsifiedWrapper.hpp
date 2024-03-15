#ifndef TATAMI_SPARSIFIED_WRAPPER_HPP
#define TATAMI_SPARSIFIED_WRAPPER_HPP

#include "../base/SparseRange.hpp"
#include "../base/Options.hpp"
#include "../base/Extractor.hpp"

#include <algorithm>

/**
 * @file SparsifiedWrapper.hpp
 *
 * @brief Wrapper class for sparse extraction from a dense `Matrix`.
 */

namespace tatami {

/**
 * @cond
 */
namespace SparsifiedWrapper_internal {

template<DimensionSelectionType selection_, class Extractor_>
auto fetch_number(const Extractor_& raw) {
    if constexpr(selection_ == DimensionSelectionType::FULL) {
        return raw.sparsify_full_length();
    } else if constexpr(selection_ == DimensionSelectionType::BLOCK) {
        return raw.sparsify_block_length();
    } else {
        return raw.sparsify_indices().size();
    }
}

template<DimensionSelectionType selection_, class Extractor_, typename Index_>
void populate_indices(const Extractor_& raw, Index_* ibuffer) {
    if constexpr(selection_ == DimensionSelectionType::FULL) {
        std::iota(ibuffer, ibuffer + raw.sparsify_full_length(), static_cast<Index_>(0));
    } else if constexpr(selection_ == DimensionSelectionType::BLOCK) {
        std::iota(ibuffer, ibuffer + raw.sparsify_block_length(), raw.sparsify_block_start());
    } else {
        const auto& ix = raw.sparsify_indices();
        std::copy(ix.begin(), ix.end(), ibuffer);
    }
}

}
/**
 * @endcond
 */

/**
 * @brief Wrap a dense extractor in the sparse interface without an oracle.
 *
 * This can be used to quickly implement the sparse extraction methods for a dense `Matrix` subclass.
 * The dense extraction is performed as usual and every value is treated as a structural non-zero;
 * this wrapper just adds the associated indices to satisfy the sparse extraction interface.
 *
 * @tparam selection_ Type of selection on the extraction dimension.
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam Extractor_ The underlying dense extractor class.
 * This should be a subclass of a `MyopicDenseExtractor` with some additional methods depending on `selection_`.
 * - For `DimensionSelectionType::FULL`, we expect the `Index_ sparsify_full_length() const` method,
 *   which returns the full extent of the extraction dimension.
 * - For `DimensionSelectionType::BLOCK`, we expect the `Index_ sparsify_block_start() const` and `Index_ sparsify_block_length() const` methods,
 *   which return the start and length of the contiguous block on the extraction dimension, respectively.
 * - For `DimensionSelectionType::INDEX`, we expect the `const std::vector<Index_>& sparsify_indices() const` method,
 *   which returns the vector of subset indices on the extraction dimension, respectively.
 */
template<DimensionSelectionType selection_, typename Value_, typename Index_, class Extractor_>
struct MyopicSparsifiedWrapper : public MyopicSparseExtractor<Value_, Index_> {
    /**
     * @param r Instance of the dense extractor.
     * This should be a `MyopicDenseExtractor` subclass produced by `Matrix::dense_row()` or `Matrix::dense_column()`.
     * @param opt Options for extraction.
     */
    MyopicSparsifiedWrapper(Extractor_ r, const Options& opt) :
        raw(std::move(r)), needs_value(opt.sparse_extract_value), needs_index(opt.sparse_extract_index) {}

    SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        Index_ num = SparsifiedWrapper_internal::fetch_number<selection_>(raw); 
        SparseRange<Value_, Index_> output(num, NULL, NULL);
        if (needs_value) {
            output.value = raw.fetch(i, vbuffer);
        }
        if (needs_index) {
            SparsifiedWrapper_internal::populate_indices<selection_>(raw, ibuffer);
            output.index = ibuffer;
        }
        return output;
    }

private:
    Extractor_ raw;
    bool needs_value;
    bool needs_index;
};

/**
 * @brief Wrap a dense extractor in the sparse interface with an oracle.
 *
 * This is the oracle-aware counterpart to `MyopicSparsifiedWrapper`, for the sparse methods that need an oracle.
 *
 * @tparam selection_ Type of selection on the extraction dimension.
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam Extractor_ The underlying dense extractor class.
 * This should be a subclass of a `OracleDenseExtractor` with some additional methods depending on `selection_`,
 * see the documentation for `MyopicSparsifiedWrapper` for more details.
 */
template<DimensionSelectionType selection_, typename Value_, typename Index_, class Extractor_>
struct OracularSparsifiedWrapper : public OracularSparseExtractor<Value_, Index_> {
    /**
     * @param r Instance of the dense extractor,
     * produced by `Matrix::dense_row()` or `Matrix::dense_column()` with an `Oracle`.
     * @param opt Options for extraction.
     */
    OracularSparsifiedWrapper(Extractor_ r, const Options& opt) :
        raw(std::move(r)), needs_value(opt.sparse_extract_value), needs_index(opt.sparse_extract_index) {}

    SparseRange<Value_, Index_> fetch(Value_* vbuffer, Index_* ibuffer) {
        Index_ num = SparsifiedWrapper_internal::fetch_number<selection_>(raw); 
        SparseRange<Value_, Index_> output(num, NULL, NULL);
        if (needs_value) {
            output.value = raw.fetch(vbuffer);
        }
        if (needs_index) {
            SparsifiedWrapper_internal::populate_indices<selection_>(raw, ibuffer);
            output.index = ibuffer;
        }
        return output;
    }

private:
    Extractor_ raw;
    bool needs_value;
    bool needs_index;
};

}

#endif
