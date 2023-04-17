#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "tatami/base/dense/DenseMatrix.hpp"
#include "tatami/base/isometric/DelayedIsometricOp.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "../_tests/test_column_access.h"
#include "../_tests/test_row_access.h"
#include "../_tests/simulate_vector.h"

class MathTest : public ::testing::Test {
protected:
    size_t nrow = 82, ncol = 51;
    std::shared_ptr<tatami::NumericMatrix> dense, sparse;
    std::vector<double> simulated;
protected:
    void SetUp() {
        simulated = simulate_sparse_vector<double>(nrow * ncol, 0.1);
        dense = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(nrow, ncol, simulated));
        sparse = tatami::convert_to_sparse<false>(dense.get()); // column major.
        return;
    }
};

TEST_F(MathTest, Abs) {
    tatami::DelayedAbsHelper op;
    auto dense_mod = tatami::make_DelayedIsometricOp(dense, op);
    auto sparse_mod = tatami::make_DelayedIsometricOp(sparse, op);

    EXPECT_FALSE(dense_mod->sparse());
    EXPECT_TRUE(sparse_mod->sparse());
    EXPECT_EQ(dense->nrow(), dense_mod->nrow());
    EXPECT_EQ(dense->ncol(), dense_mod->ncol());

    auto refvec = simulated;
    for (auto& r : refvec) {
        r = std::abs(r);
    }
    tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

    // Toughest tests are handled by the Vector case; they would
    // be kind of redundant here, so we'll just do something simple
    // to check that the scalar operation behaves as expected. 
    bool FORWARD = true;
    int JUMP = 1;

    test_simple_column_access(dense_mod.get(), &ref, FORWARD, JUMP);
    test_simple_column_access(sparse_mod.get(), &ref, FORWARD, JUMP);

    test_simple_row_access(dense_mod.get(), &ref, FORWARD, JUMP);
    test_simple_row_access(sparse_mod.get(), &ref, FORWARD, JUMP);
}

TEST_F(MathTest, SqrtByColumn) {
    tatami::DelayedAbsHelper op0;
    auto dense_mod0 = tatami::make_DelayedIsometricOp(dense, op0);
    auto sparse_mod0 = tatami::make_DelayedIsometricOp(sparse, op0);

    tatami::DelayedSqrtHelper op;
    auto dense_mod = tatami::make_DelayedIsometricOp(dense_mod0, op);
    auto sparse_mod = tatami::make_DelayedIsometricOp(sparse_mod0, op);

    EXPECT_FALSE(dense_mod->sparse());
    EXPECT_TRUE(sparse_mod->sparse());
    EXPECT_EQ(dense->nrow(), dense_mod->nrow());
    EXPECT_EQ(dense->ncol(), dense_mod->ncol());

    auto refvec = simulated;
    for (auto& r : refvec) {
        r = std::sqrt(std::abs(r));
    }
    tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

    // Again, doing some light tests.
    bool FORWARD = true;
    int JUMP = 1;

    test_simple_column_access(dense_mod.get(), &ref, FORWARD, JUMP);
    test_simple_column_access(sparse_mod.get(), &ref, FORWARD, JUMP);

    test_simple_row_access(dense_mod.get(), &ref, FORWARD, JUMP);
    test_simple_row_access(sparse_mod.get(), &ref, FORWARD, JUMP);
}

TEST_F(MathTest, LogByColumn) {
    tatami::DelayedAbsHelper op0;
    auto dense_mod0 = tatami::make_DelayedIsometricOp(dense, op0);
    auto sparse_mod0 = tatami::make_DelayedIsometricOp(sparse, op0);
    
    double CONSTANT = 5;
    tatami::DelayedAddScalarHelper<double> op1(CONSTANT);
    auto dense_mod1 = tatami::make_DelayedIsometricOp(dense_mod0, op1);
    auto sparse_mod1 = tatami::make_DelayedIsometricOp(sparse_mod0, op1);

    // Trying with the natural base.
    {
        tatami::DelayedLogHelper op;
        auto dense_mod = tatami::make_DelayedIsometricOp(dense_mod1, op);
        auto sparse_mod = tatami::make_DelayedIsometricOp(sparse_mod1, op);

        EXPECT_FALSE(dense_mod->sparse());
        EXPECT_FALSE(sparse_mod->sparse());
        EXPECT_EQ(dense->nrow(), dense_mod->nrow());
        EXPECT_EQ(dense->ncol(), dense_mod->ncol());

        auto refvec = simulated;
        for (auto& r : refvec) {
            r = std::log(std::abs(r) + CONSTANT);
        }
        tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

        // Again, doing some light tests.
        bool FORWARD = true;
        int JUMP = 1;

        test_simple_column_access(dense_mod.get(), &ref, FORWARD, JUMP);
        test_simple_column_access(sparse_mod.get(), &ref, FORWARD, JUMP);

        test_simple_row_access(dense_mod.get(), &ref, FORWARD, JUMP);
        test_simple_row_access(sparse_mod.get(), &ref, FORWARD, JUMP);
    }

    // Trying with another base.
    {
        tatami::DelayedLogHelper op(2);
        auto dense_mod = tatami::make_DelayedIsometricOp(dense_mod1, op);
        auto sparse_mod = tatami::make_DelayedIsometricOp(sparse_mod1, op);

        EXPECT_FALSE(dense_mod->sparse());
        EXPECT_FALSE(sparse_mod->sparse());
        EXPECT_EQ(dense->nrow(), dense_mod->nrow());
        EXPECT_EQ(dense->ncol(), dense_mod->ncol());

        auto refvec = simulated;
        for (auto& r : refvec) {
            r = std::log(std::abs(r) + CONSTANT) / std::log(2);
        }
        tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

        // Again, doing some light tests.
        bool FORWARD = true;
        int JUMP = 1;

        test_simple_column_access(dense_mod.get(), &ref, FORWARD, JUMP);
        test_simple_column_access(sparse_mod.get(), &ref, FORWARD, JUMP);

        test_simple_row_access(dense_mod.get(), &ref, FORWARD, JUMP);
        test_simple_row_access(sparse_mod.get(), &ref, FORWARD, JUMP);
    }
}

TEST_F(MathTest, Log1pByColumn) {
    tatami::DelayedAbsHelper op0;
    auto dense_mod0 = tatami::make_DelayedIsometricOp(dense, op0);
    auto sparse_mod0 = tatami::make_DelayedIsometricOp(sparse, op0);

    // Trying with the natural base.
    {
        tatami::DelayedLog1pHelper op;
        auto dense_mod = tatami::make_DelayedIsometricOp(dense_mod0, op);
        auto sparse_mod = tatami::make_DelayedIsometricOp(sparse_mod0, op);

        EXPECT_FALSE(dense_mod->sparse());
        EXPECT_TRUE(sparse_mod->sparse());
        EXPECT_EQ(dense->nrow(), dense_mod->nrow());
        EXPECT_EQ(dense->ncol(), dense_mod->ncol());

        auto refvec = simulated;
        for (auto& r : refvec) {
            r = std::log1p(std::abs(r));
        }
        tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

        // Again, doing some light tests.
        bool FORWARD = true;
        int JUMP = 1;

        test_simple_column_access(dense_mod.get(), &ref, FORWARD, JUMP);
        test_simple_column_access(sparse_mod.get(), &ref, FORWARD, JUMP);

        test_simple_row_access(dense_mod.get(), &ref, FORWARD, JUMP);
        test_simple_row_access(sparse_mod.get(), &ref, FORWARD, JUMP);
    }

    // Trying with another base.
    {
        tatami::DelayedLog1pHelper op(2);
        auto dense_mod = tatami::make_DelayedIsometricOp(dense_mod0, op);
        auto sparse_mod = tatami::make_DelayedIsometricOp(sparse_mod0, op);

        EXPECT_FALSE(dense_mod->sparse());
        EXPECT_TRUE(sparse_mod->sparse());
        EXPECT_EQ(dense->nrow(), dense_mod->nrow());
        EXPECT_EQ(dense->ncol(), dense_mod->ncol());

        auto refvec = simulated;
        for (auto& r : refvec) {
            r = std::log1p(std::abs(r)) / std::log(2);
        }
        tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

        // Again, doing some light tests.
        bool FORWARD = true;
        int JUMP = 1;

        test_simple_column_access(dense_mod.get(), &ref, FORWARD, JUMP);
        test_simple_column_access(sparse_mod.get(), &ref, FORWARD, JUMP);

        test_simple_row_access(dense_mod.get(), &ref, FORWARD, JUMP);
        test_simple_row_access(sparse_mod.get(), &ref, FORWARD, JUMP);
    }
}

TEST_F(MathTest, ExpByColumn) {
    tatami::DelayedExpHelper op;
    auto dense_mod = tatami::make_DelayedIsometricOp(dense, op);
    auto sparse_mod = tatami::make_DelayedIsometricOp(sparse, op);

    EXPECT_FALSE(dense_mod->sparse());
    EXPECT_FALSE(sparse_mod->sparse());
    EXPECT_EQ(dense->nrow(), dense_mod->nrow());
    EXPECT_EQ(dense->ncol(), dense_mod->ncol());

    auto refvec = simulated;
    for (auto& r : refvec) {
        r = std::exp(r);
    }
    tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

    // Again, doing some light tests.
    bool FORWARD = true;
    int JUMP = 1;

    test_simple_column_access(dense_mod.get(), &ref, FORWARD, JUMP);
    test_simple_column_access(sparse_mod.get(), &ref, FORWARD, JUMP);

    test_simple_row_access(dense_mod.get(), &ref, FORWARD, JUMP);
    test_simple_row_access(sparse_mod.get(), &ref, FORWARD, JUMP);
}

TEST_F(MathTest, RoundByColumn) {
    tatami::DelayedRoundHelper op;
    auto dense_mod = tatami::make_DelayedIsometricOp(dense, op);
    auto sparse_mod = tatami::make_DelayedIsometricOp(sparse, op);

    EXPECT_FALSE(dense_mod->sparse());
    EXPECT_TRUE(sparse_mod->sparse());
    EXPECT_EQ(dense->nrow(), dense_mod->nrow());
    EXPECT_EQ(dense->ncol(), dense_mod->ncol());

    auto refvec = simulated;
    for (auto& r : refvec) {
        r = std::round(r);
    }
    tatami::DenseRowMatrix<double> ref(nrow, ncol, std::move(refvec));

    // Again, doing some light tests.
    bool FORWARD = true;
    int JUMP = 1;

    test_simple_column_access(dense_mod.get(), &ref, FORWARD, JUMP);
    test_simple_column_access(sparse_mod.get(), &ref, FORWARD, JUMP);

    test_simple_row_access(dense_mod.get(), &ref, FORWARD, JUMP);
    test_simple_row_access(sparse_mod.get(), &ref, FORWARD, JUMP);
}
