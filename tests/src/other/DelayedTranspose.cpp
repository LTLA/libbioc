#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "tatami/dense/DenseMatrix.hpp"
#include "tatami/other/DelayedTranspose.hpp"
#include "tatami/sparse/convert_to_compressed_sparse.hpp"

#include "tatami_test/tatami_test.hpp"

class TransposeUtils {
protected:
    inline static size_t nrow = 199, ncol = 201;
    inline static std::shared_ptr<tatami::NumericMatrix> dense, sparse, tdense, tsparse, uns_tsparse, ref;

    static void assemble() {
        if (ref) {
            return;
        }

        auto simulated = tatami_test::simulate_vector<double>(nrow * ncol, []{
            tatami_test::SimulateVectorOptions opt;
            opt.density = 0.06;
            opt.seed = 12301231;
            return opt;
        }());

        dense = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int>(nrow, ncol, simulated));
        sparse = tatami::convert_to_compressed_sparse<false, double, int>(dense.get()); // column-major.
        tdense = tatami::make_DelayedTranspose(dense);
        tsparse = tatami::make_DelayedTranspose(sparse);
        uns_tsparse = tatami::make_DelayedTranspose<double, int>(std::make_shared<const tatami_test::ReversedIndicesWrapper<double, int> >(sparse));

        std::vector<double> refvec(nrow * ncol);
        for (size_t r = 0; r < nrow; ++r) {
            for (size_t c = 0; c < ncol; ++c) {
                refvec[c * nrow + r] = simulated[r * ncol + c];
            }
        }
        ref.reset(new tatami::DenseRowMatrix<double, int>(ncol, nrow, refvec));
    }
};

class TransposeTest : public ::testing::Test, public TransposeUtils {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_F(TransposeTest, Basic) {
    EXPECT_EQ(tdense->ncol(), nrow);
    EXPECT_EQ(tdense->nrow(), ncol);
    EXPECT_FALSE(tdense->is_sparse());
    EXPECT_EQ(tdense->is_sparse_proportion(), 0);
    EXPECT_FALSE(tdense->prefer_rows());
    EXPECT_EQ(tdense->prefer_rows_proportion(), 0);

    EXPECT_EQ(tsparse->ncol(), nrow);
    EXPECT_EQ(tsparse->nrow(), ncol);
    EXPECT_TRUE(tsparse->is_sparse());
    EXPECT_EQ(tsparse->is_sparse_proportion(), 1);
    EXPECT_TRUE(tsparse->prefer_rows());
    EXPECT_EQ(tsparse->prefer_rows_proportion(), 1);

    EXPECT_FALSE(tsparse->uses_oracle(false));
}

TEST_F(TransposeTest, ConstOverload) {
    std::shared_ptr<const tatami::NumericMatrix> const_dense(dense);
    auto tdense = tatami::make_DelayedTranspose(const_dense);

    // Cursory checks.
    EXPECT_EQ(dense->nrow(), tdense->ncol());
    EXPECT_EQ(dense->ncol(), tdense->nrow());
}

TEST_F(TransposeTest, Simplified) {
    auto stacked = tatami::make_DelayedTranspose(tdense);
    EXPECT_EQ(stacked->nrow(), nrow);
    EXPECT_EQ(stacked->ncol(), ncol);

    // This should collapse to the seed class.
    auto ptr = dynamic_cast<tatami::DelayedTranspose<double, int>*>(stacked.get());
    EXPECT_TRUE(ptr == NULL);
    auto ref = dynamic_cast<tatami::DelayedTranspose<double, int>*>(tdense.get());
    EXPECT_TRUE(ref != NULL); // as a control.

    tatami_test::test_simple_row_access(*stacked, *dense);
    tatami_test::test_simple_column_access(*stacked, *dense);
}

class TransposeFullTest : 
    public ::testing::TestWithParam<tatami_test::StandardTestAccessOptions>,
    public TransposeUtils {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(TransposeFullTest, Basic) {
    auto tparam = GetParam(); 
    auto options = tatami_test::convert_test_access_options(tparam);
    tatami_test::test_full_access(*tdense, *ref, options);
    tatami_test::test_full_access(*tsparse, *ref, options);
    tatami_test::test_unsorted_full_access(*uns_tsparse, options);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeTest,
    TransposeFullTest,
    tatami_test::standard_test_access_options_combinations()
);

class TransposeBlockTest : 
    public ::testing::TestWithParam<std::tuple<tatami_test::StandardTestAccessOptions, std::pair<double, double> > >, 
    public TransposeUtils {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(TransposeBlockTest, Basic) {
    auto tparam = GetParam(); 
    auto options = tatami_test::convert_test_access_options(std::get<0>(tparam));
    auto interval_info = std::get<1>(tparam);
    tatami_test::test_block_access(*tdense, *ref, interval_info.first, interval_info.second, options);
    tatami_test::test_block_access(*tsparse, *ref, interval_info.first, interval_info.second, options);
    tatami_test::test_unsorted_block_access(*uns_tsparse, interval_info.first, interval_info.second, options);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeTest,
    TransposeBlockTest,
    ::testing::Combine(
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.44),
            std::make_pair(0.21, 0.68), 
            std::make_pair(0.33, 0.67)
        )
    )
);

class TransposeIndexTest : 
    public ::testing::TestWithParam<std::tuple<tatami_test::StandardTestAccessOptions, std::pair<double, double> > >, 
    public TransposeUtils {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(TransposeIndexTest, Basic) {
    auto tparam = GetParam(); 
    auto options = tatami_test::convert_test_access_options(std::get<0>(tparam));
    auto interval_info = std::get<1>(tparam);
    tatami_test::test_indexed_access(*tdense, *ref, interval_info.first, interval_info.second, options);
    tatami_test::test_indexed_access(*tsparse, *ref, interval_info.first, interval_info.second, options);
    tatami_test::test_unsorted_indexed_access(*uns_tsparse, interval_info.first, interval_info.second, options);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeTest,
    TransposeIndexTest,
    ::testing::Combine(
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.1),
            std::make_pair(0.2, 0.2), 
            std::make_pair(0.7, 0.3)
        )
    )
);
