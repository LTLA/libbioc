#ifndef TATAMI_DELAYED_TRANSPOSE
#define TATAMI_DELAYED_TRANSPOSE

#include "../base/Matrix.hpp"
#include <memory>

/**
 * @file DelayedTranspose.hpp
 *
 * @brief Delayed transposition.
 *
 * This is equivalent to the `DelayedAperm` class in the **DelayedArray** package.
 */

namespace tatami {

/**
 * @brief Delayed transposition of a matrix.
 *
 * Implements delayed transposition of a matrix.
 * This operation is "delayed" in that it is only evaluated during data extraction.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 */
template<typename Value_, typename Index_>
class DelayedTranspose : public Matrix<Value_, Index_> {
public:
    /**
     * @param matrix Pointer to the underlying (pre-transpose) matrix.
     */
    DelayedTranspose(std::shared_ptr<const Matrix<Value_, Index_> > matrix) : my_matrix(std::move(matrix)) {}

private:
    std::shared_ptr<const Matrix<Value_, Index_> > my_matrix;

public:
    Index_ nrow() const {
        return my_matrix->ncol();
    }

    Index_ ncol() const {
        return my_matrix->nrow();
    }

    bool is_sparse() const {
        return my_matrix->is_sparse();
    }

    double is_sparse_proportion() const {
        return my_matrix->is_sparse_proportion();
    }

    bool prefer_rows() const {
        return !my_matrix->prefer_rows();
    }

    double prefer_rows_proportion() const {
        return 1 - my_matrix->prefer_rows_proportion();
    }

    bool uses_oracle(bool row) const {
        return my_matrix->uses_oracle(!row);
    }

    using Matrix<Value_, Index_>::dense;

    using Matrix<Value_, Index_>::sparse;

    /**
     * @cond
     */
    const std::shared_ptr<const Matrix<Value_, Index_> >& seed() const {
        return my_matrix;
    }
    /**
     * @endcond
     */

    /********************
     *** Myopic dense ***
     ********************/
public:
    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, const Options& opt) const {
        return my_matrix->dense(!row, opt);
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const Options& opt) const {
        return my_matrix->dense(!row, block_start, block_length, opt);
    }

    std::unique_ptr<MyopicDenseExtractor<Value_, Index_> > dense(bool row, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return my_matrix->dense(!row, std::move(indices_ptr), opt);
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const Options& opt) const {
        return my_matrix->sparse(!row, opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const Options& opt) const {
        return my_matrix->sparse(!row, block_start, block_length, opt);
    }

    std::unique_ptr<MyopicSparseExtractor<Value_, Index_> > sparse(bool row, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return my_matrix->sparse(!row, std::move(indices_ptr), opt);
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, const Options& opt) const {
        return my_matrix->dense(!row, std::move(oracle), opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        return my_matrix->dense(!row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const Oracle<Index_> > oracle, VectorPtr<Index_> indices_ptr, const Options& opt) const {
        return my_matrix->dense(!row, std::move(oracle), std::move(indices_ptr), opt);
    }

    /***********************
     *** Oracular sparse ***
     ***********************/
public:
    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, const Options& opt) const {
        return my_matrix->sparse(!row, std::move(oracle), opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const Options& opt) const {
        return my_matrix->sparse(!row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const Oracle<Index_> > oracle, VectorPtr<Index_> indices, const Options& opt) const {
        return my_matrix->sparse(!row, std::move(oracle), std::move(indices), opt);
    }
};

/**
 * A `make_*` helper function for delayed transposition.
 * This will typically return a `DelayedTranspose` instance wrapping `matrix`.
 * However, if `matrix` was already a `DelayedTranspose`, then the two delayed transpositions cancel out;
 * the underlying seed is returned directly, eliminating two unnecessary operations.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of index value.
 *
 * @param matrix Pointer to a (possibly `const`) `Matrix` instance.
 *
 * @return A pointer to a `Matrix` representing the transposition of `matrix`.
 */
template<typename Value_, typename Index_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedTranspose(std::shared_ptr<const Matrix<Value_, Index_> > matrix) {
    auto tptr = dynamic_cast<const DelayedTranspose<Value_, Index_>*>(matrix.get());
    if (tptr == NULL) {
        return std::shared_ptr<Matrix<Value_, Index_> >(new DelayedTranspose<Value_, Index_>(std::move(matrix)));
    } else {
        return std::const_pointer_cast<Matrix<Value_, Index_> >(tptr->seed()); // TODO: jesus christ, get rid of this.
    }
}

/**
 * @cond
 */
template<typename Value_, typename Index_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedTranspose(std::shared_ptr<Matrix<Value_, Index_> > matrix) {
    return make_DelayedTranspose(std::shared_ptr<const Matrix<Value_, Index_> >(std::move(matrix)));
}
/**
 * @endcond
 */

}

#endif
