#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "tatami/dense/DenseMatrix.hpp"
#include "tatami/sparse/FragmentedSparseMatrix.hpp"
#include "tatami/utils/ArrayView.hpp"

#include "tatami_test/tatami_test.hpp"

TEST(FragmentedSparseMatrix, ConstructionEmpty) {
    std::vector<std::vector<double> > values(20);
    std::vector<std::vector<int> > indices(20);

    tatami::FragmentedSparseColumnMatrix<double, int> mat(10, 20, values, indices);
    EXPECT_TRUE(mat.is_sparse());
    EXPECT_FALSE(mat.prefer_rows());
    EXPECT_EQ(mat.nrow(), 10);
    EXPECT_EQ(mat.ncol(), 20);

    // Comparing access for an empty matrix.
    tatami::DenseColumnMatrix<double, int> dense(10, 20, std::vector<double>(200));
    tatami_test::test_simple_column_access(&mat, &dense);
    tatami_test::test_simple_row_access(&mat, &dense);

    // Same for row-major.
    values.resize(10);
    indices.resize(10);
    tatami::FragmentedSparseRowMatrix<double, int> rmat(10, 20, values, indices);
    EXPECT_TRUE(rmat.is_sparse());
    EXPECT_EQ(rmat.nrow(), 10);
    EXPECT_EQ(rmat.ncol(), 20);
    tatami_test::test_simple_column_access(&rmat, &dense);
    tatami_test::test_simple_row_access(&rmat, &dense);
}

TEST(FragmentedSparseMatrix, ConstructionFail) {
    std::vector<std::vector<double> > values(20);
    std::vector<std::vector<int> > indices(19);
    tatami_test::throws_error([&]() { tatami::FragmentedSparseColumnMatrix<double, int> mat(10, 20, values, indices); }, "same length");

    indices.resize(20);
    tatami_test::throws_error([&]() { tatami::FragmentedSparseColumnMatrix<double, int> mat(10, 10, values, indices); }, "should be equal to number of columns");
    tatami_test::throws_error([&]() { tatami::FragmentedSparseRowMatrix<double, int> mat(10, 10, values, indices); }, "should be equal to number of rows");

    indices.front().resize(10);
    tatami_test::throws_error([&]() { tatami::FragmentedSparseColumnMatrix<double, int> mat(10, 20, values, indices); }, "same length");

    values.front().resize(10);
    tatami_test::throws_error([&]() { tatami::FragmentedSparseColumnMatrix<double, int> mat(10, 20, values, indices); }, "strictly increasing");
    tatami_test::throws_error([&]() { tatami::FragmentedSparseRowMatrix<double, int> mat(20, 10, values, indices); }, "strictly increasing");

    values.front().resize(1);
    indices.front().resize(1);
    indices.front()[0] = -1;
    tatami_test::throws_error([&]() { tatami::FragmentedSparseColumnMatrix<double, int> mat(10, 20, values, indices); }, "non-negative");

    indices.front()[0] = 10001;
    tatami_test::throws_error([&]() { tatami::FragmentedSparseColumnMatrix<double, int> mat(10, 20, values, indices); }, "non-negative");
}

/*************************************
 *************************************/

class FragmentedSparseUtils {
protected:
    inline static size_t nrow = 200, ncol = 100;
    inline static std::shared_ptr<tatami::NumericMatrix> dense, sparse_row, sparse_column;

    static void assemble() {
        if (dense) {
            return;
        }

        dense.reset(new tatami::DenseRowMatrix<double, int>(nrow, ncol, tatami_test::simulate_sparse_vector<double>(nrow * ncol, 0.05)));

        {
            std::vector<std::vector<double> > values(nrow);
            std::vector<std::vector<int> > indices(nrow);
            std::vector<double> buffer(ncol);

            auto wrk = dense->dense_row();
            for (size_t r = 0; r < nrow; ++r) {
                auto content = wrk->fetch(r, buffer.data());
                for (size_t c = 0; c < ncol; ++c) {
                    if (content[c]) {
                        values[r].push_back(content[c]);
                        indices[r].push_back(c);
                    }
                }
            }

            sparse_row.reset(new tatami::FragmentedSparseRowMatrix<double, int>(nrow, ncol, std::move(values), std::move(indices)));
        }

        {
            std::vector<std::vector<double> > values(ncol);
            std::vector<std::vector<int> > indices(ncol);
            std::vector<double> buffer(nrow);

            auto wrk = dense->dense_column();
            for (size_t c = 0; c < ncol; ++c) {
                auto content = wrk->fetch(c, buffer.data());
                for (size_t r = 0; r < nrow; ++r) {
                    if (content[r]) {
                        values[c].push_back(content[r]);
                        indices[c].push_back(r);
                    }
                }
            }

            sparse_column.reset(new tatami::FragmentedSparseColumnMatrix<double, int>(nrow, ncol, std::move(values), std::move(indices)));
        }

        return;
    }
};

class FragmentedSparseTest : public ::testing::Test, public FragmentedSparseUtils {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_F(FragmentedSparseTest, Basic) {
    size_t NC = sparse_column->ncol(), NR = sparse_column->nrow();
    EXPECT_EQ(NC, ncol);
    EXPECT_EQ(NR, nrow);
    EXPECT_EQ(sparse_row->ncol(), ncol);
    EXPECT_EQ(sparse_row->nrow(), nrow);

    EXPECT_FALSE(dense->is_sparse());
    EXPECT_TRUE(sparse_row->is_sparse());
    EXPECT_EQ(sparse_row->is_sparse_proportion(), 1);
    EXPECT_TRUE(sparse_column->is_sparse());
    EXPECT_EQ(sparse_column->is_sparse_proportion(), 1);

    EXPECT_TRUE(sparse_row->prefer_rows());
    EXPECT_EQ(sparse_row->prefer_rows_proportion(), 1);
    EXPECT_FALSE(sparse_column->prefer_rows());
    EXPECT_EQ(sparse_column->prefer_rows_proportion(), 0);

    EXPECT_FALSE(sparse_row->uses_oracle(true));
}

/*************************************
 *************************************/

class FragmentedSparseFullAccessTest : public ::testing::TestWithParam<tatami_test::StandardTestAccessParameters>, public FragmentedSparseUtils {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(FragmentedSparseFullAccessTest, Full) {
    auto tparam = GetParam(); 
    auto params = tatami_test::convert_access_parameters(tparam);
    tatami_test::test_full_access(params, sparse_column.get(), dense.get());
    tatami_test::test_full_access(params, sparse_row.get(), dense.get());
}

INSTANTIATE_TEST_SUITE_P(
    FragmentedSparseMatrix,
    FragmentedSparseFullAccessTest,
    ::testing::Combine(
        ::testing::Values(true, false), // row extraction.
        ::testing::Values(true, false), // an oracle.
        ::testing::Values(tatami_test::FORWARD, tatami_test::REVERSE, tatami_test::RANDOM),
        ::testing::Values(1, 4, 10, 20) // jump, to test the workspace's memory; trying out longer jumps to test secondaries.
    )
);

/*************************************
 *************************************/

class FragmentedSparseSlicedAccessTest : public ::testing::TestWithParam<std::tuple<tatami_test::StandardTestAccessParameters, std::pair<double, double> > >, public FragmentedSparseUtils {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(FragmentedSparseSlicedAccessTest, Sliced) {
    auto tparam = GetParam(); 
    auto params = tatami_test::convert_access_parameters(std::get<0>(tparam));

    auto interval_info = std::get<1>(tparam);
    auto len = (params.use_row ? ncol : nrow);
    size_t FIRST = interval_info.first * len, LAST = interval_info.second * len;

    tatami_test::test_block_access(params, sparse_column.get(), dense.get(), FIRST, LAST);
    tatami_test::test_block_access(params, sparse_row.get(), dense.get(), FIRST, LAST);
}

INSTANTIATE_TEST_SUITE_P(
    FragmentedSparseMatrix,
    FragmentedSparseSlicedAccessTest,
    ::testing::Combine(
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.51),
            std::make_pair(0.25, 0.9), 
            std::make_pair(0.63, 1.0)
        )
    )
);

/*************************************
 *************************************/

class FragmentedSparseIndexedAccessTest : public ::testing::TestWithParam<std::tuple<tatami_test::StandardTestAccessParameters, std::pair<double, int> > >, public FragmentedSparseUtils {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(FragmentedSparseIndexedAccessTest, Indexed) {
    auto tparam = GetParam(); 
    auto params = tatami_test::convert_access_parameters(std::get<0>(tparam));

    auto interval_info = std::get<1>(tparam);
    auto len = (params.use_row ? ncol : nrow);
    size_t FIRST = interval_info.first * len, STEP = interval_info.second;

    tatami_test::test_indexed_access(params, sparse_column.get(), dense.get(), FIRST, STEP);
    tatami_test::test_indexed_access(params, sparse_row.get(), dense.get(), FIRST, STEP);
}

INSTANTIATE_TEST_SUITE_P(
    FragmentedSparseMatrix,
    FragmentedSparseIndexedAccessTest,
    ::testing::Combine(
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 5),
            std::make_pair(0.2, 10), 
            std::make_pair(0.7, 3)
        )
    )
);

/*************************************
 *************************************/

TEST(FragmentedSparseMatrix, ArrayView) {
    std::vector<tatami::ArrayView<double> > values(20, tatami::ArrayView<double>(NULL, 0));
    std::vector<tatami::ArrayView<int> > indices(20, tatami::ArrayView<int>(NULL, 0));

    tatami::FragmentedSparseColumnMatrix<double, int, decltype(values), decltype(indices)> mat(10, 20, values, indices);
    EXPECT_TRUE(mat.is_sparse());
    EXPECT_FALSE(mat.prefer_rows());
    EXPECT_EQ(mat.nrow(), 10);
    EXPECT_EQ(mat.ncol(), 20);
}
