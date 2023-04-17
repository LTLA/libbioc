#ifndef TATAMI_DENSE_MATRIX_H
#define TATAMI_DENSE_MATRIX_H

#include "VirtualDenseMatrix.hpp"
#include "../StandardExtractor.hpp"
#include "../utils.hpp"

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
 * @brief Dense matrix representation.
 *
 * @tparam row_ Whether this is a row-major representation.
 * If `false`, a column-major representation is assumed instead.
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam Storage_ Vector class used to store the matrix values internally.
 * This does not necessarily have to contain `Value_`, as long as the type is convertible to `Value_`.
 * Methods should be available for `size()`, `begin()`, `end()` and `[]`.
 * If a method is available for `data()` that returns a `const Value_*`, it will also be used.
 */
template<bool row_, typename Value_, typename Index_ = int, class Storage_ = std::vector<Value_> >
class DenseMatrix : public VirtualDenseMatrix<Value_, Index_> {
public: 
    /**
     * @param nr Number of rows.
     * @param nc Number of columns.
     * @param source Vector of values, or length equal to the product of `nr` and `nc`.
     */
    DenseMatrix(Index_ nr, Index_ nc, const Storage_& source) : nrows(nr), ncols(nc), values(source) {
        check_dimensions(nr, nc, values.size());
        return;
    }

    /**
     * @param nr Number of rows.
     * @param nc Number of columns.
     * @param source Vector of values, or length equal to the product of `nr` and `nc`.
     */
    DenseMatrix(Index_ nr, Index_ nc, Storage_&& source) : nrows(nr), ncols(nc), values(source) {
        check_dimensions(nr, nc, values.size());
        return;
    }

private: 
    Index_ nrows, ncols;
    Storage_ values;

    static void check_dimensions(size_t nr, size_t nc, size_t expected) { // cast to size_t is deliberate to avoid overflow on Index_ on product.
        if (nr * nc != expected) {
            throw std::runtime_error("length of 'values' should be equal to product of 'nrows' and 'ncols'");
        }
    }

public:
    Index_ nrow() const { return nrows; }

    Index_ ncol() const { return ncols; }

    bool prefer_rows() const { return row_; }

    using Matrix<Value_, Index_>::dense_row;

    using Matrix<Value_, Index_>::dense_column;

    using Matrix<Value_, Index_>::sparse_row;

    using Matrix<Value_, Index_>::sparse_column;

private:
    template<DimensionSelectionType selection_, bool use_start_, bool accrow_>
    struct DenseBase : public StandardExtractor<selection_, use_start_, false, Value_, Index_> {
        DenseBase(const DenseMatrix* p, ExtractionOptions<Index_>& options) : 
            StandardExtractor<selection_, use_start_, false, Value_, Index_>(options), // note: moves options.selection.indices
            parent(p) 
        {
            if constexpr(selection_ == DimensionSelectionType::FULL) {
                this->extracted_length = (accrow_ ? parent->ncols : parent->nrows);
            }
        }

    public:
        const Value_* fetch(Index_ position, Value_* buffer) {
            if constexpr(row_ == accrow_) {
                if constexpr(selection_ == DimensionSelectionType::FULL) {
                    return parent->primary<accrow_>(position, buffer, static_cast<Index_>(0), this->extracted_length);
                } else if constexpr(selection_ == DimensionSelectionType::BLOCK) {
                    return parent->primary<accrow_>(position, buffer, this->extracted_block, this->extracted_block + this->extracted_length);
                } else {
                    return parent->primary<accrow_>(position, buffer, this->quick_extracted_index(), this->extracted_length);
                }
            } else {
                if constexpr(selection_ == DimensionSelectionType::FULL) {
                    parent->secondary<accrow_>(position, buffer, static_cast<Index_>(0), this->extracted_length);
                } else if constexpr(selection_ == DimensionSelectionType::BLOCK) {
                    parent->secondary<accrow_>(position, buffer, this->extracted_block, this->extracted_block + this->extracted_length);
                } else {
                    parent->secondary<accrow_>(position, buffer, this->quick_extracted_index(), this->extracted_length);
                }
                return buffer;
            }
        }

    private:
        const DenseMatrix* parent;
    };

    template<bool accrow_>
    std::unique_ptr<DenseExtractor<Value_, Index_> > populate(ExtractionOptions<Index_> eopt) const {
        std::unique_ptr<DenseExtractor<Value_, Index_> > output;

        switch (eopt.selection.type) {
            case DimensionSelectionType::FULL:
                output.reset(new DenseBase<DimensionSelectionType::FULL, true, accrow_>(this, eopt));
                break;
            case DimensionSelectionType::BLOCK:
                output.reset(new DenseBase<DimensionSelectionType::BLOCK, true, accrow_>(this, eopt));
                break;
            case DimensionSelectionType::INDEX:
                if (eopt.selection.index_start) {
                    output.reset(new DenseBase<DimensionSelectionType::INDEX, true, accrow_>(this, eopt));
                } else {
                    output.reset(new DenseBase<DimensionSelectionType::INDEX, false, accrow_>(this, eopt));
                }
                break;
        }

        return output;
    }

private:
    template<bool accrow_> 
    size_t other_dimension() const { // deliberate cast to avoid integer overflow on Index_ when multiplying to compute offsets.
        if constexpr(row_) {
            return ncols;
        } else {
            return nrows;
        }
    }

    template<bool accrow_> 
    const Value_* primary(Index_ x, Value_* buffer, Index_ start, Index_ end) const {
        size_t shift = x * other_dimension<accrow_>();
        if constexpr(has_data<Value_, Storage_>::value) {
            return values.data() + shift + start;
        } else {
            std::copy(values.begin() + shift + start, values.begin() + shift + end, buffer);
            return buffer;
        }
    }

    template<bool accrow_> 
    void secondary(Index_ x, Value_* buffer, Index_ start, Index_ end) const {
        size_t dim_secondary = other_dimension<accrow_>();
        auto it = values.begin() + x + start * dim_secondary;
        for (Index_ i = start; i < end; ++i, ++buffer, it += dim_secondary) {
            *buffer = *it; 
        }
        return;
    }

    template<bool accrow_> 
    const Value_* primary(Index_ x, Value_* buffer, const Index_* indices, Index_ length) const {
        size_t offset = x * other_dimension<accrow_>();
        for (Index_ i = 0; i < length; ++i) {
            buffer[i] = values[indices[i] + offset];
        }
        return buffer;
    }

    template<bool accrow_> 
    void secondary(Index_ x, Value_* buffer, const Index_* indices, Index_ length) const {
        size_t dim_secondary = other_dimension<accrow_>();        
        for (Index_ i = 0; i < length; ++i, ++buffer) {
            *buffer = values[indices[i] * dim_secondary + x]; 
        }
        return;
    }

public:
    std::unique_ptr<DenseExtractor<Value_, Index_> > dense_row(IterationOptions<Index_>, ExtractionOptions<Index_> eopt) const {
        return populate<true>(std::move(eopt));
    }

    std::unique_ptr<DenseExtractor<Value_, Index_> > dense_column(IterationOptions<Index_>, ExtractionOptions<Index_> eopt) const {
        return populate<false>(std::move(eopt));
    }
};

/**
 * Column-major matrix.
 * See `tatami::DenseMatrix` for details on the template parameters.
 */
template<typename Value_, typename Index_ = int, class Storage_ = std::vector<Value_> >
using DenseColumnMatrix = DenseMatrix<false, Value_, Index_, Storage_>;

/**
 * Row-major matrix.
 * See `tatami::DenseMatrix` for details on the template parameters.
 */
template<typename Value_, typename Index_ = int, class Storage_ = std::vector<Value_> >
using DenseRowMatrix = DenseMatrix<true, Value_, Index_, Storage_>;

}

#endif