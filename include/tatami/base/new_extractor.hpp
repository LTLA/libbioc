#ifndef TATAMI_NEW_EXTRACTOR_HPP
#define TATAMI_NEW_EXTRACTOR_HPP

#include "Matrix.hpp"

/**
 * @file new_extractor.hpp
 * @brief Templated construction of a new extractor.
 */

namespace tatami {

/**
 * @tparam row_ Whether to iterate over rows.
 * @tparam sparse_ Whether to perform sparse retrieval.
 * @tparam Value_ Data value type, should be numeric.
 * @tparam Index_ Row/column index type, should be integer.
 * @tparam Args_ Further arguments.
 *
 * @param[in] ptr Pointer to a `Matrix` object to iterate over.
 * @param args Zero or more additional arguments to pass to methods like `Matrix::dense_row()`.
 *
 * @return An extractor to access the requested dimension of `ptr`.
 * This may be any of `MyopicDenseExtractor`, `MyopicSparseExtractor`, `OracularDenseExtractor` or `OracularSparseExtractor`,
 * depending on `sparse_` and whether an `Oracle` is supplied in `args`.
 */
template<bool row_, bool sparse_, typename Value_, typename Index_, typename ... Args_>
auto new_extractor(const Matrix<Value_, Index_>* ptr, Args_&&... args) {
    if constexpr(sparse_) {
        if constexpr(row_) {
            return ptr->sparse_row(std::forward<Args_>(args)...);
        } else {
            return ptr->sparse_column(std::forward<Args_>(args)...);
        }
    } else {
        if constexpr(row_) {
            return ptr->dense_row(std::forward<Args_>(args)...);
        } else {
            return ptr->dense_column(std::forward<Args_>(args)...);
        }
    }
}

}

#endif
