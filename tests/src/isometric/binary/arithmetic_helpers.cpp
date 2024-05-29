#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "tatami/dense/DenseMatrix.hpp"
#include "tatami/isometric/binary/DelayedBinaryIsometricOperation.hpp"
#include "tatami/isometric/binary/arithmetic_helpers.hpp"
#include "tatami/isometric/unary/DelayedUnaryIsometricOperation.hpp"
#include "tatami/isometric/unary/math_helpers.hpp"
#include "tatami/sparse/convert_to_compressed_sparse.hpp"

#include "tatami_test/tatami_test.hpp"
#include "../utils.h"

class DelayedBinaryIsometricArithmeticUtils {
protected:
    inline static size_t nrow = 91, ncol = 121;
    inline static std::shared_ptr<tatami::NumericMatrix> dense_left, sparse_left, dense_right, sparse_right;
    inline static std::vector<double> simulated_left, simulated_right;
    
    static void assemble() {
        if (dense_left) {
            return;
        }

        simulated_left = tatami_test::simulate_sparse_vector<double>(nrow * ncol, 0.1, /* lower = */ -5, /* upper = */ 5, /* seed */ 12345);
        dense_left = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int>(nrow, ncol, simulated_left));
        sparse_left = tatami::convert_to_compressed_sparse<false, double, int>(dense_left.get()); // column major.

        simulated_right = tatami_test::simulate_sparse_vector<double>(nrow * ncol, 0.1, /* lower = */ -5, /* upper = */ 5, /* seed */ 67890);
        dense_right = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int>(nrow, ncol, simulated_right));
        sparse_right = tatami::convert_to_compressed_sparse<false, double, int>(dense_right.get()); // column major.
        return;
    }
};

#define BINARY_ARITH_BASIC_SETUP(name, base) \
    class name : \
        public ::testing::Test, \
        public base { \
    protected: \
        static void SetUpTestSuite() { \
            assemble(); \
        } \
    };

#define BINARY_ARITH_FULL_TEST(name, base) \
    class name : \
        public ::testing::TestWithParam<tatami_test::StandardTestAccessParameters>, \
        public base { \
    protected: \
        static void SetUpTestSuite() { \
            assemble(); \
        } \
    }; \
    \
    TEST_P(name, Basic) { \
        auto params = tatami_test::convert_access_parameters(GetParam()); \
        tatami_test::test_full_access(params, dense_mod.get(), ref.get()); \
        tatami_test::test_full_access(params, sparse_mod.get(), ref.get()); \
        tatami_test::test_full_access(params, sparse_uns.get(), ref.get()); \
    } \
    \
    INSTANTIATE_TEST_SUITE_P( \
        DelayedBinaryIsometricArithmetic, \
        name, \
        tatami_test::standard_test_access_parameter_combinations() \
    );

#define BINARY_ARITH_BLOCK_TEST(name, base) \
    class name : \
        public ::testing::TestWithParam<std::tuple<tatami_test::StandardTestAccessParameters, std::pair<double, double> > >, \
        public base { \
    protected: \
        static void SetUpTestSuite() { \
            assemble(); \
        } \
    }; \
    \
    TEST_P(name, Basic) { \
        auto tparam = GetParam(); \
        auto params = tatami_test::convert_access_parameters(std::get<0>(tparam)); \
        auto interval_info = std::get<1>(tparam); \
        auto len = (params.use_row ? ref->ncol() : ref->nrow()); \
        size_t FIRST = interval_info.first * len, LAST = interval_info.second * len; \
        tatami_test::test_block_access(params, dense_mod.get(), ref.get(), FIRST, LAST); \
        tatami_test::test_block_access(params, sparse_mod.get(), ref.get(), FIRST, LAST); \
        tatami_test::test_block_access(params, sparse_uns.get(), ref.get(), FIRST, LAST); \
    } \
    \
    INSTANTIATE_TEST_SUITE_P( \
        DelayedBinaryIsometricArithmetic, \
        name, \
        ::testing::Combine( \
            tatami_test::standard_test_access_parameter_combinations(), \
            ::testing::Values( \
                std::make_pair(0, 0.35), \
                std::make_pair(0.27, 0.87), \
                std::make_pair(0.67, 1.0) \
            ) \
        ) \
    );

#define BINARY_ARITH_INDEX_TEST(name, base) \
    class name : \
        public ::testing::TestWithParam<std::tuple<tatami_test::StandardTestAccessParameters, std::pair<double, int> > >, \
        public base { \
    protected: \
        static void SetUpTestSuite() { \
            assemble(); \
        } \
    }; \
    \
    TEST_P(name, Basic) { \
        auto tparam = GetParam(); \
        auto params = tatami_test::convert_access_parameters(std::get<0>(tparam)); \
        auto interval_info = std::get<1>(tparam); \
        auto len = (params.use_row ? ref->ncol() : ref->nrow()); \
        size_t FIRST = interval_info.first * len, STEP = interval_info.second; \
        tatami_test::test_indexed_access(params, dense_mod.get(), ref.get(), FIRST, STEP); \
        tatami_test::test_indexed_access(params, sparse_mod.get(), ref.get(), FIRST, STEP); \
        tatami_test::test_indexed_access(params, sparse_uns.get(), ref.get(), FIRST, STEP); \
    } \
    \
    INSTANTIATE_TEST_SUITE_P( \
        DelayedBinaryIsometricArithmetic, \
        name, \
        ::testing::Combine( \
            tatami_test::standard_test_access_parameter_combinations(), \
            ::testing::Values( \
                std::make_pair(0.0, 4), \
                std::make_pair(0.21, 9), \
                std::make_pair(0.56, 10) \
            ) \
        ) \
    );

/****************************
 ********* ADDITION *********
 ****************************/

class DelayedBinaryIsometricAddUtils : public DelayedBinaryIsometricArithmeticUtils {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_mod, sparse_mod, sparse_uns, ref;

    static void assemble() {
        if (ref) {
            return;
        }

        DelayedBinaryIsometricArithmeticUtils::assemble();
        auto op = tatami::make_DelayedBinaryIsometricAdd();
        dense_mod = tatami::make_DelayedBinaryIsometricOperation(dense_left, dense_right, op);
        sparse_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left, sparse_right, op);
        sparse_uns = tatami::make_DelayedBinaryIsometricOperation(
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_left)),
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_right)),
            op
        );

        auto refvec = simulated_left;
        for (size_t i = 0; i < refvec.size(); ++i) {
            refvec[i] += simulated_right[i];
        }
        ref.reset(new tatami::DenseRowMatrix<double, int>(nrow, ncol, std::move(refvec)));
    }
};

BINARY_ARITH_BASIC_SETUP(DelayedBinaryIsometricAddTest, DelayedBinaryIsometricAddUtils)
TEST_F(DelayedBinaryIsometricAddTest, Basic) {
    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_EQ(dense_mod->is_sparse_proportion(), 0);
    EXPECT_TRUE(sparse_mod->is_sparse());
    EXPECT_EQ(sparse_mod->is_sparse_proportion(), 1);
    EXPECT_EQ(nrow, dense_mod->nrow());
    EXPECT_EQ(ncol, dense_mod->ncol());

    EXPECT_TRUE(dense_mod->prefer_rows());
    EXPECT_EQ(dense_mod->prefer_rows_proportion(), 1);
    EXPECT_FALSE(sparse_mod->prefer_rows());
    EXPECT_EQ(sparse_mod->prefer_rows_proportion(), 0);

    auto mixed_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left, dense_right, tatami::make_DelayedBinaryIsometricAdd());
    EXPECT_FALSE(mixed_mod->is_sparse());
    EXPECT_EQ(mixed_mod->prefer_rows_proportion(), 0.5);
    EXPECT_EQ(mixed_mod->is_sparse_proportion(), 0.5);
}

BINARY_ARITH_FULL_TEST(DelayedBinaryIsometricAddFullTest, DelayedBinaryIsometricAddUtils)
BINARY_ARITH_BLOCK_TEST(DelayedBinaryIsometricAddBlockTest, DelayedBinaryIsometricAddUtils)
BINARY_ARITH_INDEX_TEST(DelayedBinaryIsometricAddIndexTest, DelayedBinaryIsometricAddUtils)

TEST_F(DelayedBinaryIsometricAddTest, NewType) {
    auto op = tatami::make_DelayedBinaryIsometricAdd();
    auto dense_fmod = tatami::make_DelayedBinaryIsometricOperation<float>(dense_left, dense_right, op);
    auto sparse_fmod = tatami::make_DelayedBinaryIsometricOperation<float>(sparse_left, sparse_right, op);

    std::vector<float> frefvec(simulated_left.size());
    for (size_t i = 0; i < frefvec.size(); ++i) {
        frefvec[i] = simulated_left[i] + simulated_right[i];
    }
    tatami::DenseRowMatrix<float, int> fref(nrow, ncol, std::move(frefvec));

    quick_test_all(dense_fmod.get(), &fref);
    quick_test_all(sparse_fmod.get(), &fref);
}

/*******************************
 ********* SUBTRACTION *********
 *******************************/

class DelayedBinaryIsometricSubtractUtils : public DelayedBinaryIsometricArithmeticUtils {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_mod, sparse_mod, sparse_uns, ref;

    static void assemble() {
        if (ref) {
            return;
        }

        DelayedBinaryIsometricArithmeticUtils::assemble();
        auto op = tatami::make_DelayedBinaryIsometricSubtract();
        dense_mod = tatami::make_DelayedBinaryIsometricOperation(dense_left, dense_right, op);
        sparse_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left, sparse_right, op);
        sparse_uns = tatami::make_DelayedBinaryIsometricOperation(
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_left)),
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_right)), 
            op
        );

        auto refvec = simulated_left;
        for (size_t i = 0; i < refvec.size(); ++i) {
            refvec[i] -= simulated_right[i];
        }
        ref.reset(new tatami::DenseRowMatrix<double, int>(nrow, ncol, std::move(refvec)));
    }
};

BINARY_ARITH_BASIC_SETUP(DelayedBinaryIsometricSubtractTest, DelayedBinaryIsometricSubtractUtils)
TEST_F(DelayedBinaryIsometricSubtractTest, Basic) {
    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_TRUE(sparse_mod->is_sparse());

    auto mixed_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left, dense_right, tatami::make_DelayedBinaryIsometricSubtract());
    EXPECT_FALSE(mixed_mod->is_sparse());
}

BINARY_ARITH_FULL_TEST(DelayedBinaryIsometricSubtractFullTest, DelayedBinaryIsometricSubtractUtils)
BINARY_ARITH_BLOCK_TEST(DelayedBinaryIsometricSubtractBlockTest, DelayedBinaryIsometricSubtractUtils)
BINARY_ARITH_INDEX_TEST(DelayedBinaryIsometricSubtractIndexTest, DelayedBinaryIsometricSubtractUtils)

/**********************************
 ********* MULTIPLICATION *********
 **********************************/

class DelayedBinaryIsometricMultiplyUtils : public DelayedBinaryIsometricArithmeticUtils {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_mod, sparse_mod, sparse_uns, ref;

    static void assemble() {
        if (ref) {
            return;
        }

        DelayedBinaryIsometricArithmeticUtils::assemble();

        auto op = tatami::make_DelayedBinaryIsometricMultiply();
        dense_mod = tatami::make_DelayedBinaryIsometricOperation(dense_left, dense_right, op);
        sparse_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left, sparse_right, op);
        sparse_uns = tatami::make_DelayedBinaryIsometricOperation(
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_left)),
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_right)), 
            op
        );

        auto refvec = simulated_left;
        for (size_t i = 0; i < refvec.size(); ++i) {
            refvec[i] *= simulated_right[i];
        }
        ref.reset(new tatami::DenseRowMatrix<double, int>(nrow, ncol, std::move(refvec)));
    }
};

BINARY_ARITH_BASIC_SETUP(DelayedBinaryIsometricMultiplyTest, DelayedBinaryIsometricMultiplyUtils)
TEST_F(DelayedBinaryIsometricMultiplyTest, Basic) {
    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_TRUE(sparse_mod->is_sparse());
}

BINARY_ARITH_FULL_TEST(DelayedBinaryIsometricMultiplyFullTest, DelayedBinaryIsometricMultiplyUtils)
BINARY_ARITH_BLOCK_TEST(DelayedBinaryIsometricMultiplyBlockTest, DelayedBinaryIsometricMultiplyUtils)
BINARY_ARITH_INDEX_TEST(DelayedBinaryIsometricMultiplyIndexTest, DelayedBinaryIsometricMultiplyUtils)

/****************************
 ********* DIVISION *********
 ****************************/

#define BINARY_ARITH_FULL_TEST_WITH_NAN(name, base) \
    class name : \
        public ::testing::TestWithParam<tatami_test::StandardTestAccessParameters>, \
        public base { \
    protected: \
        static void SetUpTestSuite() { \
            assemble(); \
        } \
    }; \
    \
    TEST_P(name, Basic) { \
        auto params = tatami_test::convert_access_parameters(GetParam()); \
        params.has_nan = true; \
        tatami_test::test_full_access(params, dense_mod.get(), ref.get()); \
        tatami_test::test_full_access(params, sparse_mod.get(), ref.get()); \
        tatami_test::test_full_access(params, sparse_uns.get(), ref.get()); \
    } \
    \
    INSTANTIATE_TEST_SUITE_P( \
        DelayedBinaryIsometricArithmetic, \
        name, \
        tatami_test::standard_test_access_parameter_combinations() \
    );

#define BINARY_ARITH_BLOCK_TEST_WITH_NAN(name, base) \
    class name : \
        public ::testing::TestWithParam<std::tuple<tatami_test::StandardTestAccessParameters, std::pair<double, double> > >, \
        public base { \
    protected: \
        static void SetUpTestSuite() { \
            assemble(); \
        } \
    }; \
    \
    TEST_P(name, Basic) { \
        auto tparam = GetParam(); \
        auto params = tatami_test::convert_access_parameters(std::get<0>(tparam)); \
        params.has_nan = true; \
        auto interval_info = std::get<1>(tparam); \
        auto len = (params.use_row ? ref->ncol() : ref->nrow()); \
        size_t FIRST = interval_info.first * len, LAST = interval_info.second * len; \
        tatami_test::test_block_access(params, dense_mod.get(), ref.get(), FIRST, LAST); \
        tatami_test::test_block_access(params, sparse_mod.get(), ref.get(), FIRST, LAST); \
        tatami_test::test_block_access(params, sparse_uns.get(), ref.get(), FIRST, LAST); \
    } \
    \
    INSTANTIATE_TEST_SUITE_P( \
        DelayedBinaryIsometricArithmetic, \
        name, \
        ::testing::Combine( \
            tatami_test::standard_test_access_parameter_combinations(), \
            ::testing::Values( \
                std::make_pair(0, 0.35), \
                std::make_pair(0.27, 0.87), \
                std::make_pair(0.67, 1.0) \
            ) \
        ) \
    );

#define BINARY_ARITH_INDEX_TEST_WITH_NAN(name, base) \
    class name : \
        public ::testing::TestWithParam<std::tuple<tatami_test::StandardTestAccessParameters, std::pair<double, int> > >, \
        public base { \
    protected: \
        static void SetUpTestSuite() { \
            assemble(); \
        } \
    }; \
    \
    TEST_P(name, Basic) { \
        auto tparam = GetParam(); \
        auto params = tatami_test::convert_access_parameters(std::get<0>(tparam)); \
        params.has_nan = true; \
        auto interval_info = std::get<1>(tparam); \
        auto len = (params.use_row ? ref->ncol() : ref->nrow()); \
        size_t FIRST = interval_info.first * len, STEP = interval_info.second; \
        tatami_test::test_indexed_access(params, dense_mod.get(), ref.get(), FIRST, STEP); \
        tatami_test::test_indexed_access(params, sparse_mod.get(), ref.get(), FIRST, STEP); \
        tatami_test::test_indexed_access(params, sparse_uns.get(), ref.get(), FIRST, STEP); \
    } \
    \
    INSTANTIATE_TEST_SUITE_P( \
        DelayedBinaryIsometricArithmetic, \
        name, \
        ::testing::Combine( \
            tatami_test::standard_test_access_parameter_combinations(), \
            ::testing::Values( \
                std::make_pair(0.0, 4), \
                std::make_pair(0.21, 9), \
                std::make_pair(0.56, 10) \
            ) \
        ) \
    );


class DelayedBinaryIsometricDivideUtils : public DelayedBinaryIsometricArithmeticUtils {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_mod, sparse_mod, sparse_uns, ref;

    static void assemble() {
        if (ref) {
            return;
        }

        DelayedBinaryIsometricArithmeticUtils::assemble();

        auto op = tatami::make_DelayedBinaryIsometricDivide();
        dense_mod = tatami::make_DelayedBinaryIsometricOperation(dense_left, dense_right, op);
        sparse_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left, sparse_right, op);
        sparse_uns = tatami::make_DelayedBinaryIsometricOperation(
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_left)),
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_right)), 
            op
        );

        auto refvec = simulated_left;
        for (size_t i = 0; i < refvec.size(); ++i) {
            refvec[i] = careful_division(refvec[i], simulated_right[i]);
        }
        ref.reset(new tatami::DenseRowMatrix<double, int>(nrow, ncol, std::move(refvec)));
    }
};

BINARY_ARITH_BASIC_SETUP(DelayedBinaryIsometricDivideTest, DelayedBinaryIsometricDivideUtils)
TEST_F(DelayedBinaryIsometricDivideTest, Basic) {
    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_FALSE(sparse_mod->is_sparse());
    EXPECT_TRUE(dense_mod->prefer_rows());
    EXPECT_FALSE(sparse_mod->prefer_rows());
}

BINARY_ARITH_FULL_TEST_WITH_NAN(DelayedBinaryIsometricDivideFullTest, DelayedBinaryIsometricDivideUtils)
BINARY_ARITH_BLOCK_TEST_WITH_NAN(DelayedBinaryIsometricDivideBlockTest, DelayedBinaryIsometricDivideUtils)
BINARY_ARITH_INDEX_TEST_WITH_NAN(DelayedBinaryIsometricDivideIndexTest, DelayedBinaryIsometricDivideUtils)

/*******************************
 ************ POWER ************
 *******************************/

class DelayedBinaryIsometricPowerUtils : public DelayedBinaryIsometricArithmeticUtils {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_mod, sparse_mod, sparse_uns, ref;

    static void assemble() {
        if (ref) {
            return;
        }

        DelayedBinaryIsometricArithmeticUtils::assemble();

        tatami::DelayedUnaryIsometricAbs op0;
        auto dense_left0 = tatami::make_DelayedUnaryIsometricOperation(dense_left, op0);
        auto sparse_left0 = tatami::make_DelayedUnaryIsometricOperation(sparse_left, op0);
        auto dense_right0 = tatami::make_DelayedUnaryIsometricOperation(dense_right, op0);
        auto sparse_right0 = tatami::make_DelayedUnaryIsometricOperation(sparse_right, op0);

        auto op = tatami::make_DelayedBinaryIsometricPower();
        dense_mod = tatami::make_DelayedBinaryIsometricOperation(dense_left0, dense_right0, op);
        sparse_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left0, sparse_right0, op);
        sparse_uns = tatami::make_DelayedBinaryIsometricOperation(
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_left0)),
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_right0)), 
            op
        );

        auto refvec = simulated_left;
        for (size_t i = 0; i < refvec.size(); ++i) {
            refvec[i] = std::pow(std::abs(refvec[i]), std::abs(simulated_right[i]));
        }
        ref.reset(new tatami::DenseRowMatrix<double, int>(nrow, ncol, std::move(refvec)));
    }
};

BINARY_ARITH_BASIC_SETUP(DelayedBinaryIsometricPowerTest, DelayedBinaryIsometricPowerUtils)
TEST_F(DelayedBinaryIsometricPowerTest, Basic) {
    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_FALSE(sparse_mod->is_sparse());
}

BINARY_ARITH_FULL_TEST(DelayedBinaryIsometricPowerFullTest, DelayedBinaryIsometricPowerUtils)
BINARY_ARITH_BLOCK_TEST(DelayedBinaryIsometricPowerBlockTest, DelayedBinaryIsometricPowerUtils)
BINARY_ARITH_INDEX_TEST(DelayedBinaryIsometricPowerIndexTest, DelayedBinaryIsometricPowerUtils)

/****************************
 ********** MODULO **********
 ****************************/

class DelayedBinaryIsometricModuloUtils : public DelayedBinaryIsometricArithmeticUtils {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_mod, sparse_mod, sparse_uns, ref;

    static void assemble() {
        if (ref) {
            return;
        }

        DelayedBinaryIsometricArithmeticUtils::assemble();

        auto op = tatami::make_DelayedBinaryIsometricModulo();
        dense_mod = tatami::make_DelayedBinaryIsometricOperation(dense_left, dense_right, op);
        sparse_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left, sparse_right, op);
        sparse_uns = tatami::make_DelayedBinaryIsometricOperation(
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_left)),
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_right)), 
            op
        );

        auto refvec = simulated_left;
        for (size_t i = 0; i < refvec.size(); ++i) {
            refvec[i] = std::fmod(refvec[i], simulated_right[i]);
        }
        ref.reset(new tatami::DenseRowMatrix<double, int>(nrow, ncol, std::move(refvec)));
    }
};

BINARY_ARITH_BASIC_SETUP(DelayedBinaryIsometricModuloTest, DelayedBinaryIsometricModuloUtils)
TEST_F(DelayedBinaryIsometricModuloTest, Basic) {
    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_FALSE(sparse_mod->is_sparse());
    EXPECT_TRUE(dense_mod->prefer_rows());
    EXPECT_FALSE(sparse_mod->prefer_rows());
}

BINARY_ARITH_FULL_TEST_WITH_NAN(DelayedBinaryIsometricModuloFullTest, DelayedBinaryIsometricModuloUtils)
BINARY_ARITH_BLOCK_TEST_WITH_NAN(DelayedBinaryIsometricModuloBlockTest, DelayedBinaryIsometricModuloUtils)
BINARY_ARITH_INDEX_TEST_WITH_NAN(DelayedBinaryIsometricModuloIndexTest, DelayedBinaryIsometricModuloUtils)

/****************************
 ***** INTEGER DIVISION *****
 ****************************/

class DelayedBinaryIsometricIntegerDivideUtils : public DelayedBinaryIsometricArithmeticUtils {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_mod, sparse_mod, sparse_uns, ref;

    static void assemble() {
        if (ref) {
            return;
        }

        DelayedBinaryIsometricArithmeticUtils::assemble();

        auto op = tatami::make_DelayedBinaryIsometricIntegerDivide();
        dense_mod = tatami::make_DelayedBinaryIsometricOperation(dense_left, dense_right, op);
        sparse_mod = tatami::make_DelayedBinaryIsometricOperation(sparse_left, sparse_right, op);
        sparse_uns = tatami::make_DelayedBinaryIsometricOperation(
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_left)),
            std::shared_ptr<tatami::NumericMatrix>(new tatami_test::UnsortedWrapper<double, int>(sparse_right)), 
            op
        );

        auto refvec = simulated_left;
        for (size_t i = 0; i < refvec.size(); ++i) {
            // x == (x %% y) + y * (x %/% y)
            refvec[i] = std::floor(refvec[i] / simulated_right[i]);
        }
        ref.reset(new tatami::DenseRowMatrix<double, int>(nrow, ncol, std::move(refvec)));
    }
};

BINARY_ARITH_BASIC_SETUP(DelayedBinaryIsometricIntegerDivideTest, DelayedBinaryIsometricIntegerDivideUtils)
TEST_F(DelayedBinaryIsometricIntegerDivideTest, Basic) {
    EXPECT_FALSE(dense_mod->is_sparse());
    EXPECT_FALSE(sparse_mod->is_sparse());
    EXPECT_TRUE(dense_mod->prefer_rows());
    EXPECT_FALSE(sparse_mod->prefer_rows());
}

BINARY_ARITH_FULL_TEST_WITH_NAN(DelayedBinaryIsometricIntegerDivideFullTest, DelayedBinaryIsometricIntegerDivideUtils)
BINARY_ARITH_BLOCK_TEST_WITH_NAN(DelayedBinaryIsometricIntegerDivideBlockTest, DelayedBinaryIsometricIntegerDivideUtils)
BINARY_ARITH_INDEX_TEST_WITH_NAN(DelayedBinaryIsometricIntegerDivideIndexTest, DelayedBinaryIsometricIntegerDivideUtils)
