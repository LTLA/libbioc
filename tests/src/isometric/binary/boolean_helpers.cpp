#include <gtest/gtest.h>

#include <vector>
#include <memory>
#include <tuple>

#include "tatami/dense/DenseMatrix.hpp"
#include "tatami/isometric/binary/DelayedBinaryIsometricOp.hpp"
#include "tatami/sparse/convert_to_compressed_sparse.hpp"

#include "tatami_test/tatami_test.hpp"
#include "../utils.h"

class BinaryBooleanTest : public ::testing::Test {
protected:
    size_t nrow = 123, ncol = 155;
    std::shared_ptr<tatami::NumericMatrix> dense_left, sparse_left, dense_right, sparse_right;
    std::vector<double> simulated_left, simulated_right;
protected:
    void SetUp() {
        simulated_left = tatami_test::simulate_sparse_vector<double>(nrow * ncol, 0.2, /* lower = */ -10, /* upper = */ 10, /* seed */ 12345);
        dense_left = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(nrow, ncol, simulated_left));
        sparse_left = tatami::convert_to_compressed_sparse<false>(dense_left.get()); // column major.

        simulated_right = tatami_test::simulate_sparse_vector<double>(nrow * ncol, 0.2, /* lower = */ -10, /* upper = */ 10, /* seed */ 67890);
        dense_right = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(nrow, ncol, simulated_right));
        sparse_right = tatami::convert_to_compressed_sparse<false>(dense_right.get()); // column major.
        return;
    }
};

TEST_F(BinaryBooleanTest, EQUAL) {
    auto op = tatami::make_DelayedBinaryBooleanEqualHelper();
    auto dense_mod = tatami::make_DelayedBinaryIsometricOp(dense_left, dense_right, op);
    auto sparse_mod = tatami::make_DelayedBinaryIsometricOp(sparse_left, sparse_right, op);

    EXPECT_FALSE(dense_mod->sparse());
    EXPECT_FALSE(sparse_mod->sparse());

    // Toughest tests are handled by 'arith_helpers.cpp'; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the operation behaves as expected. 
    std::vector<double> refvec(nrow * ncol);
    for (size_t i = 0; i < refvec.size(); ++i) {
        refvec[i] = static_cast<bool>(simulated_left[i]) == static_cast<bool>(simulated_right[i]);
    }
    tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

    quick_test_all(dense_mod.get(), &ref);
    quick_test_all(sparse_mod.get(), &ref);
}

TEST_F(BinaryBooleanTest, AND) {
    auto op = tatami::make_DelayedBinaryBooleanAndHelper();
    auto dense_mod = tatami::make_DelayedBinaryIsometricOp(dense_left, dense_right, op);
    auto sparse_mod = tatami::make_DelayedBinaryIsometricOp(sparse_left, sparse_right, op);

    EXPECT_FALSE(dense_mod->sparse());
    EXPECT_TRUE(sparse_mod->sparse());

    // Toughest tests are handled by 'arith_helpers.cpp'; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the operation behaves as expected. 
    std::vector<double> refvec(nrow * ncol);
    for (size_t i = 0; i < refvec.size(); ++i) {
        refvec[i] = static_cast<bool>(simulated_left[i]) && static_cast<bool>(simulated_right[i]);
    }
    tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

    quick_test_all(dense_mod.get(), &ref);
    quick_test_all(sparse_mod.get(), &ref);
}

TEST_F(BinaryBooleanTest, OR) {
    auto op = tatami::make_DelayedBinaryBooleanOrHelper();
    auto dense_mod = tatami::make_DelayedBinaryIsometricOp(dense_left, dense_right, op);
    auto sparse_mod = tatami::make_DelayedBinaryIsometricOp(sparse_left, sparse_right, op);

    EXPECT_FALSE(dense_mod->sparse());
    EXPECT_TRUE(sparse_mod->sparse());

    // Toughest tests are handled by 'arith_helpers.cpp'; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the operation behaves as expected. 
    std::vector<double> refvec(nrow * ncol);
    for (size_t i = 0; i < refvec.size(); ++i) {
        refvec[i] = static_cast<bool>(simulated_left[i]) || static_cast<bool>(simulated_right[i]);
    }
    tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

    quick_test_all(dense_mod.get(), &ref);
    quick_test_all(sparse_mod.get(), &ref);
}

TEST_F(BinaryBooleanTest, XOR) {
    auto op = tatami::make_DelayedBinaryBooleanXorHelper();
    auto dense_mod = tatami::make_DelayedBinaryIsometricOp(dense_left, dense_right, op);
    auto sparse_mod = tatami::make_DelayedBinaryIsometricOp(sparse_left, sparse_right, op);

    EXPECT_FALSE(dense_mod->sparse());
    EXPECT_TRUE(sparse_mod->sparse());

    // Toughest tests are handled by 'arith_helpers.cpp'; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the operation behaves as expected. 
    std::vector<double> refvec(nrow * ncol);
    for (size_t i = 0; i < refvec.size(); ++i) {
        refvec[i] = static_cast<bool>(simulated_left[i]) != static_cast<bool>(simulated_right[i]);
    }
    tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

    quick_test_all(dense_mod.get(), &ref);
    quick_test_all(sparse_mod.get(), &ref);
}
