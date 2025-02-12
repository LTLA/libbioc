#include <gtest/gtest.h>
#include "../custom_parallel.h"

#include "tatami/dense/DenseMatrix.hpp"
#include "tatami/sparse/CompressedSparseMatrix.hpp"
#include "tatami/dense/convert_to_dense.hpp"

#include "tatami_test/tatami_test.hpp"

class ConvertToDenseTest : public ::testing::TestWithParam<std::tuple<int, int, bool, bool, int> > {
protected:
    size_t NR, NC;
    bool from_row, to_row;
    int threads;

    template<class Param>
    void assemble(const Param& param) {
        NR = std::get<0>(param);
        NC = std::get<1>(param);
        from_row = std::get<2>(param);
        to_row = std::get<3>(param);
        threads = std::get<4>(param);
    }
};

TEST_P(ConvertToDenseTest, FromDense) {
    assemble(GetParam());

    auto vec = tatami_test::simulate_vector<double>(NR * NC, []{
        tatami_test::SimulateVectorOptions opt;
        opt.seed = 142857;
        return opt;
    }());
    auto mat = std::make_shared<tatami::DenseMatrix<double, int> >(NR, NC, vec, from_row);

    auto converted = tatami::convert_to_dense(mat.get(), to_row, threads);
    EXPECT_EQ(converted->prefer_rows(), to_row);
    EXPECT_FALSE(converted->is_sparse());

    tatami_test::test_simple_row_access(*converted, *mat);
    tatami_test::test_simple_column_access(*converted, *mat);

    auto converted2 = tatami::convert_to_dense<int, size_t>(mat.get(), to_row, threads); // works for a different type.
    EXPECT_EQ(converted2->prefer_rows(), to_row);
    EXPECT_FALSE(converted2->is_sparse());

    auto old = mat->dense_row();
    std::vector<double> buffer(NC);
    auto wrk2 = converted2->dense_row();
    for (size_t i = 0; i < NR; ++i) {
        auto ptr = old->fetch(i, buffer.data());
        std::vector<int> expected(ptr, ptr + NC);
        EXPECT_EQ(tatami_test::fetch(*wrk2, i, NC), expected);
    }
}

TEST_P(ConvertToDenseTest, FromSparse) {
    assemble(GetParam());

    auto vec = tatami_test::simulate_compressed_sparse<double, int>((from_row ? NR : NC), (from_row ? NC : NR), [&]{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.2;
        opt.seed = 3498761;
        return opt;
    }());
    tatami::CompressedSparseMatrix<double, int> smat(NR, NC, std::move(vec.data), std::move(vec.index), std::move(vec.indptr), from_row);

    auto converted = tatami::convert_to_dense(&smat, to_row, threads);
    EXPECT_EQ(converted->prefer_rows(), to_row);
    EXPECT_FALSE(converted->is_sparse());
    tatami_test::test_simple_row_access(*converted, smat);
    tatami_test::test_simple_column_access(*converted, smat);
}

INSTANTIATE_TEST_SUITE_P(
    ConvertToDense,
    ConvertToDenseTest,
    ::testing::Combine(
        ::testing::Values(10, 50, 100), // number of rows
        ::testing::Values(10, 50, 100), // number of columns
        ::testing::Values(true, false), // from row major?
        ::testing::Values(true, false), // to row major?
        ::testing::Values(1, 3)         // number of threads
    )
);
