#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "tatami/dense/DenseMatrix.hpp"
#include "tatami/isometric/unary/DelayedUnaryIsometricOperation.hpp"
#include "tatami/isometric/unary/boolean_helpers.hpp"
#include "tatami/sparse/convert_to_compressed_sparse.hpp"

#include "tatami_test/tatami_test.hpp"
#include "../utils.h"

class DelayedUnaryIsometricBooleanVectorTest : public ::testing::TestWithParam<std::tuple<bool, bool> > {
protected:
    inline static size_t nrow = 191, ncol = 88;
    inline static std::shared_ptr<tatami::NumericMatrix> dense, sparse;
    inline static std::vector<double> simulated;

    static void SetUpTestSuite() {
        simulated = tatami_test::simulate_sparse_vector<double>(nrow * ncol, 0.1, -3, 3);
        dense = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int>(nrow, ncol, simulated));
        sparse = tatami::convert_to_compressed_sparse<false, double, int>(dense.get()); // column major.
    }

    static void fill_default_vector(std::vector<char>& vec) {
        bool val = true;
        for (auto& x : vec) {
            x = val;
            val = !val;
        }
    }
};

TEST_P(DelayedUnaryIsometricBooleanVectorTest, AND) {
    auto param = GetParam();
    bool row = std::get<0>(param);
    bool is_sparse = std::get<1>(param);

    std::vector<char> vec(row ? nrow : ncol);
    if (is_sparse) {
        std::fill(vec.begin(), vec.end(), 1);
    } else {
        fill_default_vector(vec);
    }

    auto op = tatami::make_DelayedUnaryIsometricBooleanAndVector(vec, row);
    auto dense_mod = tatami::make_DelayedUnaryIsometricOperation(dense, op);
    auto sparse_mod = tatami::make_DelayedUnaryIsometricOperation(sparse, op);

    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_EQ(dense->nrow(), dense_mod->nrow());
    EXPECT_EQ(dense->ncol(), dense_mod->ncol());
    EXPECT_TRUE(sparse_mod->is_sparse());

    // Toughest tests are handled by 'arith_vector.hpp'; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the operation behaves as expected. 
    auto refvec = simulated;
    for (size_t r = 0; r < nrow; ++r) {
        for (size_t c = 0; c < ncol; ++c) {
            auto& x = refvec[r * ncol + c];
            x = (x && vec[row ? r : c]);
        }
    }
    
    tatami::DenseRowMatrix<double, int> ref(nrow, ncol, std::move(refvec));
    quick_test_all(dense_mod.get(), &ref);
    quick_test_all(sparse_mod.get(), &ref);
}

TEST_P(DelayedUnaryIsometricBooleanVectorTest, OR) {
    auto param = GetParam();
    bool row = std::get<0>(param);
    bool is_sparse = std::get<1>(param);

    std::vector<char> vec(row ? nrow : ncol);
    if (!is_sparse) {
        fill_default_vector(vec);
    }

    auto op = tatami::make_DelayedUnaryIsometricBooleanOrVector(vec, row);
    auto dense_mod = tatami::make_DelayedUnaryIsometricOperation(dense, op);
    auto sparse_mod = tatami::make_DelayedUnaryIsometricOperation(sparse, op);

    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_EQ(dense->nrow(), dense_mod->nrow());
    EXPECT_EQ(dense->ncol(), dense_mod->ncol());
    if (is_sparse) {
        EXPECT_TRUE(sparse_mod->is_sparse());
    } else {
        EXPECT_FALSE(sparse_mod->is_sparse());
    }

    // Toughest tests are handled by 'arith_vector.hpp'; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the operation behaves as expected. 
    auto refvec = simulated;
    for (size_t r = 0; r < nrow; ++r) {
        for (size_t c = 0; c < ncol; ++c) {
            auto& x = refvec[r * ncol + c];
            x = (x || vec[row ? r : c]);
        }
    }
    
    tatami::DenseRowMatrix<double, int> ref(nrow, ncol, std::move(refvec));
    quick_test_all(dense_mod.get(), &ref);
    quick_test_all(sparse_mod.get(), &ref);
}

TEST_P(DelayedUnaryIsometricBooleanVectorTest, XOR) {
    auto param = GetParam();
    bool row = std::get<0>(param);
    bool is_sparse = std::get<1>(param);

    std::vector<char> vec(row ? nrow : ncol);
    if (!is_sparse) {
        fill_default_vector(vec);
    }

    auto op = tatami::make_DelayedUnaryIsometricBooleanXorVector(vec, row);
    auto dense_mod = tatami::make_DelayedUnaryIsometricOperation(dense, op);
    auto sparse_mod = tatami::make_DelayedUnaryIsometricOperation(sparse, op);

    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_EQ(dense->nrow(), dense_mod->nrow());
    EXPECT_EQ(dense->ncol(), dense_mod->ncol());
    if (is_sparse) {
        EXPECT_TRUE(sparse_mod->is_sparse());
    } else {
        EXPECT_FALSE(sparse_mod->is_sparse());
    }

    // Toughest tests are handled by 'arith_vector.hpp'; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the operation behaves as expected. 
    auto refvec = simulated;
    for (size_t r = 0; r < nrow; ++r) {
        for (size_t c = 0; c < ncol; ++c) {
            auto& x = refvec[r * ncol + c];
            x = (static_cast<bool>(x) != static_cast<bool>(vec[row ? r : c]));
        }
    }
    
    tatami::DenseRowMatrix<double, int> ref(nrow, ncol, std::move(refvec));
    quick_test_all(dense_mod.get(), &ref);
    quick_test_all(sparse_mod.get(), &ref);
}

TEST_P(DelayedUnaryIsometricBooleanVectorTest, EQUAL) {
    auto param = GetParam();
    bool row = std::get<0>(param);
    bool is_sparse = std::get<1>(param);

    std::vector<char> vec(row ? nrow : ncol);
    if (is_sparse) {
        std::fill(vec.begin(), vec.end(), 1);
    } else {
        fill_default_vector(vec);
    }

    auto op = tatami::make_DelayedUnaryIsometricBooleanEqualVector(vec, row);
    auto dense_mod = tatami::make_DelayedUnaryIsometricOperation(dense, op);
    auto sparse_mod = tatami::make_DelayedUnaryIsometricOperation(sparse, op);

    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_EQ(dense->nrow(), dense_mod->nrow());
    EXPECT_EQ(dense->ncol(), dense_mod->ncol());
    if (is_sparse) {
        EXPECT_TRUE(sparse_mod->is_sparse());
    } else {
        EXPECT_FALSE(sparse_mod->is_sparse());
    }

    // Toughest tests are handled by 'arith_vector.hpp'; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the operation behaves as expected. 
    auto refvec = simulated;
    for (size_t r = 0; r < nrow; ++r) {
        for (size_t c = 0; c < ncol; ++c) {
            auto& x = refvec[r * ncol + c];
            x = (static_cast<bool>(x) == static_cast<bool>(vec[row ? r : c]));
        }
    }
    
    tatami::DenseRowMatrix<double, int> ref(nrow, ncol, std::move(refvec));
    quick_test_all(dense_mod.get(), &ref);
    quick_test_all(sparse_mod.get(), &ref);
}

INSTANTIATE_TEST_SUITE_P(
    DelayedUnaryIsometricBooleanVector,
    DelayedUnaryIsometricBooleanVectorTest,
    ::testing::Combine(
        ::testing::Values(true, false), // add by row, or by column
        ::testing::Values(true, false) // check sparse case
    )
);
