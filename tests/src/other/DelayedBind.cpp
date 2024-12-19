#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "tatami/dense/DenseMatrix.hpp"
#include "tatami/other/DelayedBind.hpp"
#include "tatami/sparse/convert_to_compressed_sparse.hpp"

#include "tatami_test/tatami_test.hpp"

class DelayedBindUtils {
public:
    typedef std::tuple<std::vector<int>, bool> SimulationParameters;

    static auto spawn_bind_scenarios () {
        return ::testing::Combine(
            ::testing::Values(
                std::vector<int>{ 10 },
                std::vector<int>{ 10, 20 },
                std::vector<int>{ 20, 10 },
                std::vector<int>{ 5, 2, 5 },
                std::vector<int>{ 5, 10, 20 },
                std::vector<int>{ 20, 10, 5 },
                std::vector<int>{ 5, 0, 5 }
            ),
            ::testing::Values(true, false) // bind by row or by column
        );
    }

protected:
    inline static size_t otherdim = 97;
    inline static std::shared_ptr<tatami::NumericMatrix> bound_dense, bound_sparse, manual;
    inline static std::shared_ptr<tatami::NumericMatrix> forced_bound_dense, forced_bound_sparse;
    inline static std::shared_ptr<tatami::NumericMatrix> uns_bound_dense, uns_bound_sparse;
    inline static SimulationParameters last_params;

    static void assemble(SimulationParameters sim_params) {
        if (bound_dense && sim_params == last_params) {
            return;
        }
        last_params = sim_params;

        const auto& lengths = std::get<0>(sim_params);
        bool row = std::get<1>(sim_params);

        std::vector<double> concat;
        size_t n_total = 0;
        std::vector<std::shared_ptr<tatami::NumericMatrix> > collected_dense, collected_sparse;
        std::vector<std::shared_ptr<tatami::NumericMatrix> > forced_collected_dense, forced_collected_sparse;
        std::vector<std::shared_ptr<tatami::NumericMatrix> > uns_collected_sparse;

        for (size_t i = 0; i < lengths.size(); ++i) {
            auto to_add = tatami_test::simulate_vector<double>(lengths[i] * otherdim, [&]{
                tatami_test::SimulateVectorOptions opt;
                opt.density = 0.2;
                opt.lower = -10;
                opt.upper = 10;
                opt.seed = i * 17 + 59 * lengths[i];
                return opt;
            }());

            concat.insert(concat.end(), to_add.begin(), to_add.end());
            n_total += lengths[i];

            if (row) {
                collected_dense.emplace_back(new tatami::DenseRowMatrix<double, int>(lengths[i], otherdim, to_add));
            } else {
                collected_dense.emplace_back(new tatami::DenseColumnMatrix<double, int>(otherdim, lengths[i], to_add));
            }
            collected_sparse.push_back(tatami::convert_to_compressed_sparse<false, double, int>(collected_dense.back().get())); // always CSC

            forced_collected_dense.emplace_back(std::make_shared<tatami_test::ForcedOracleWrapper<double, int> >(collected_dense.back()));
            forced_collected_sparse.emplace_back(std::make_shared<tatami_test::ForcedOracleWrapper<double, int> >(collected_sparse.back()));
            uns_collected_sparse.emplace_back(std::make_shared<tatami_test::ReversedIndicesWrapper<double, int> >(collected_sparse.back()));
        }

        if (row) {
            bound_dense.reset(new tatami::DelayedBind(std::move(collected_dense), true));
            bound_sparse.reset(new tatami::DelayedBind(std::move(collected_sparse), true));
            manual.reset(new tatami::DenseRowMatrix<double, int>(n_total, otherdim, std::move(concat)));
            forced_bound_dense.reset(new tatami::DelayedBind(std::move(forced_collected_dense), true));
            forced_bound_sparse.reset(new tatami::DelayedBind(std::move(forced_collected_sparse), true));
            uns_bound_sparse.reset(new tatami::DelayedBind(std::move(uns_collected_sparse), true));
        } else {
            bound_dense.reset(new tatami::DelayedBind(std::move(collected_dense), false));
            bound_sparse.reset(new tatami::DelayedBind(std::move(collected_sparse), false));
            manual.reset(new tatami::DenseColumnMatrix<double, int>(otherdim, n_total, std::move(concat)));
            forced_bound_dense.reset(new tatami::DelayedBind(std::move(forced_collected_dense), false));
            forced_bound_sparse.reset(new tatami::DelayedBind(std::move(forced_collected_sparse), false));
            uns_bound_sparse.reset(new tatami::DelayedBind(std::move(uns_collected_sparse), false));
        }
    }
};

class DelayedBindUtilsTest : public ::testing::Test, public DelayedBindUtils {};

TEST_F(DelayedBindUtilsTest, ByRow) {
    assemble(SimulationParameters({ 10, 20, 5 }, true));

    EXPECT_EQ(bound_dense->nrow(), 35);
    EXPECT_EQ(bound_dense->ncol(), otherdim);
    EXPECT_FALSE(bound_dense->is_sparse());
    EXPECT_EQ(bound_dense->is_sparse_proportion(), 0);
    EXPECT_TRUE(bound_dense->prefer_rows());
    EXPECT_EQ(bound_dense->prefer_rows_proportion(), 1);

    EXPECT_EQ(bound_sparse->nrow(), 35);
    EXPECT_EQ(bound_sparse->ncol(), otherdim);
    EXPECT_TRUE(bound_sparse->is_sparse());
    EXPECT_EQ(bound_sparse->is_sparse_proportion(), 1);
    EXPECT_FALSE(bound_sparse->prefer_rows());
    EXPECT_EQ(bound_sparse->prefer_rows_proportion(), 0);

    EXPECT_FALSE(bound_sparse->uses_oracle(true));
    EXPECT_FALSE(bound_sparse->uses_oracle(true));
}

TEST_F(DelayedBindUtilsTest, ByColumn) {
    assemble(SimulationParameters({ 10, 20, 5 }, false));

    EXPECT_EQ(bound_dense->nrow(), otherdim);
    EXPECT_EQ(bound_dense->ncol(), 35);
    EXPECT_FALSE(bound_dense->is_sparse());
    EXPECT_FALSE(bound_dense->prefer_rows());

    EXPECT_EQ(bound_sparse->nrow(), otherdim);
    EXPECT_EQ(bound_sparse->ncol(), 35);
    EXPECT_TRUE(bound_sparse->is_sparse());
    EXPECT_FALSE(bound_sparse->prefer_rows());

    EXPECT_FALSE(bound_dense->uses_oracle(false));
    EXPECT_FALSE(bound_sparse->uses_oracle(false));
}

TEST_F(DelayedBindUtilsTest, InconsistentBinds) {
    assemble(SimulationParameters({ 10, 20, 5 }, true));

    // Bound_sparse is CSC, bound_dense is row-major.
    tatami::DelayedBind combined(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ bound_sparse, bound_dense }, false);

    EXPECT_FLOAT_EQ(combined.is_sparse_proportion(), 0.5);
    EXPECT_FALSE(combined.is_sparse());

    EXPECT_FLOAT_EQ(combined.prefer_rows_proportion(), 0.5);
    EXPECT_FALSE(combined.prefer_rows());
}

TEST_F(DelayedBindUtilsTest, MakeHelper) {
    assemble(SimulationParameters({ 10, 20, 5 }, true));

    {
        std::vector<std::shared_ptr<tatami::NumericMatrix> > collected{ manual };

        auto direct = tatami::make_DelayedBind(collected, true);
        EXPECT_EQ(direct->nrow(), manual->nrow());
        EXPECT_EQ(direct->ncol(), manual->ncol());
        auto ptr = dynamic_cast<tatami::DelayedBind<double, int>*>(direct.get()); // No DelayedBind if there's only one element.
        EXPECT_TRUE(ptr == NULL);

        tatami_test::test_simple_row_access(*direct, *manual);
        tatami_test::test_simple_column_access(*direct, *manual);
    }

    {
        std::vector<std::shared_ptr<tatami::NumericMatrix> > collected{ manual, manual };

        auto direct = tatami::make_DelayedBind(collected, true);
        EXPECT_EQ(direct->nrow(), manual->nrow() * 2);
        EXPECT_EQ(direct->ncol(), manual->ncol());
        auto ptr = dynamic_cast<tatami::DelayedBind<double, int>*>(direct.get()); // straightforward wrapping with a DelayedBind object.
        EXPECT_TRUE(ptr != NULL);
        EXPECT_EQ(ptr->seeds().size(), 2);

        tatami::DelayedBind ref(std::move(collected), true);
        tatami_test::test_simple_row_access(*direct, ref);
        tatami_test::test_simple_column_access(*direct, ref);
    }

    {
        std::vector<std::shared_ptr<tatami::NumericMatrix> > collected{ bound_dense, bound_dense };

        auto stacked = tatami::make_DelayedBind(collected, true);
        EXPECT_EQ(stacked->nrow(), bound_dense->nrow() * 2);
        EXPECT_EQ(stacked->ncol(), bound_dense->ncol());
        auto ptr = dynamic_cast<tatami::DelayedBind<double, int>*>(stacked.get()); // Internal DelayedBinds should be collapsed together.
        EXPECT_TRUE(ptr != NULL);
        EXPECT_EQ(ptr->seeds().size(), 6);

        tatami::DelayedBind ref(std::move(collected), true);
        tatami_test::test_simple_row_access(*stacked, ref);
        tatami_test::test_simple_column_access(*stacked, ref);
    }

    {
        std::vector<std::shared_ptr<tatami::NumericMatrix> > collected{ bound_dense, bound_dense };

        auto stacked = tatami::make_DelayedBind(collected, false);
        EXPECT_EQ(stacked->nrow(), bound_dense->nrow());
        EXPECT_EQ(stacked->ncol(), bound_dense->ncol() * 2);
        auto ptr = dynamic_cast<tatami::DelayedBind<double, int>*>(stacked.get()); // No collapsing of operations if the bind dimension is not the same.
        EXPECT_TRUE(ptr != NULL);
        EXPECT_EQ(ptr->seeds().size(), 2);

        tatami::DelayedBind ref(std::move(collected), false);
        tatami_test::test_simple_row_access(*stacked, ref);
        tatami_test::test_simple_column_access(*stacked, ref);
    }
}

TEST_F(DelayedBindUtilsTest, ConstOverloads) {
    assemble(SimulationParameters({ 10, 50 }, true));

    // Direct construction.
    {
        std::vector<std::shared_ptr<const tatami::NumericMatrix> > const_collected({ bound_dense, bound_sparse });
        tatami::DelayedBind const_combined(std::move(const_collected), true);

        // Some cursory checks.
        EXPECT_EQ(const_combined.nrow(), 120); // i.e., (10 + 50) * 2 
        EXPECT_EQ(const_combined.ncol(), otherdim);
    }

    // Via the helper.
    {
        std::vector<std::shared_ptr<const tatami::NumericMatrix> > const_collected({ bound_dense, bound_sparse });
        auto const_combined = tatami::make_DelayedBind(std::move(const_collected), true);

        // Some cursory checks.
        EXPECT_EQ(const_combined->nrow(), 120); // i.e., (10 + 50) * 2 
        EXPECT_EQ(const_combined->ncol(), otherdim);
    }
}

TEST(DelayedBindMisc, ErrorCheck) {
    std::vector<std::shared_ptr<tatami::NumericMatrix> > collected;
    collected.emplace_back(new tatami::DenseRowMatrix<double, int>(10, 20, std::vector<double>(200)));
    collected.emplace_back(new tatami::DenseRowMatrix<double, int>(20, 10, std::vector<double>(200)));
    tatami_test::throws_error([&]() { tatami::DelayedBind(collected, true); }, "same number of columns");
    tatami_test::throws_error([&]() { tatami::DelayedBind(collected, false); }, "same number of rows");
}

TEST(DelayedBindMisc, PartialOracleUsage) {
    std::vector<std::shared_ptr<tatami::NumericMatrix> > collected;
    collected.emplace_back(new tatami::DenseRowMatrix<double, int>(10, 20, std::vector<double>(200)));
    collected.emplace_back(new tatami::DenseRowMatrix<double, int>(10, 20, std::vector<double>(200)));

    {
        tatami::DelayedBind combined(collected, true); 
        EXPECT_FALSE(combined.uses_oracle(true));
        EXPECT_FALSE(combined.uses_oracle(false));
    }

    {
        auto p = std::make_shared<tatami_test::ForcedOracleWrapper<double, int> >(std::move(collected.back()));
        collected.back() = std::move(p);
        tatami::DelayedBind combined(collected, true); 
        EXPECT_TRUE(combined.uses_oracle(true));
        EXPECT_TRUE(combined.uses_oracle(false));
    }

    {
        auto p = std::make_shared<tatami_test::ForcedOracleWrapper<double, int> >(std::move(collected.front()));
        collected.front() = std::move(p);
        tatami::DelayedBind combined(collected, true); 
        EXPECT_TRUE(combined.uses_oracle(true));
        EXPECT_TRUE(combined.uses_oracle(false));
    }
}

TEST(DelayedBindMisc, AllEmpty) {
    tatami::DelayedBind empty(std::vector<std::shared_ptr<tatami::Matrix<double, int> > >{}, true);
    EXPECT_EQ(empty.nrow(), 0);
    EXPECT_EQ(empty.ncol(), 0);
}

/****************************
 ****************************/

class DelayedBindEmptyAccessTest : 
    public ::testing::TestWithParam<std::tuple<bool, bool, bool> >,
    public DelayedBindUtils {};

TEST_P(DelayedBindEmptyAccessTest, Empty) {
    auto tparam = GetParam();
    auto bind_rows = std::get<0>(tparam);
    assemble(SimulationParameters({0, 0}, bind_rows));

    if (bind_rows) {
        EXPECT_EQ(bound_dense->nrow(), 0);
        EXPECT_EQ(bound_dense->ncol(), otherdim);
    } else {
        EXPECT_EQ(bound_dense->nrow(), otherdim);
        EXPECT_EQ(bound_dense->ncol(), 0);
    }

    tatami_test::TestAccessOptions options;
    options.use_row = std::get<1>(tparam);
    options.use_oracle = std::get<2>(tparam);

    // Check that we can perform all of the accesses.
    tatami_test::test_full_access(*bound_dense, *manual, options);
    tatami_test::test_block_access(*bound_dense, *manual, 0, 0, options);
    tatami_test::test_indexed_access(*bound_dense, *manual, 0, 1, options);

    tatami_test::test_full_access(*bound_sparse, *manual, options);
    tatami_test::test_block_access(*bound_sparse, *manual, 0, 0, options);
    tatami_test::test_indexed_access(*bound_sparse, *manual, 0, 1, options);

    if (options.use_oracle) {
        tatami_test::test_full_access(*forced_bound_dense, *manual, options);
        tatami_test::test_block_access(*forced_bound_dense, *manual, 0, 0, options);
        tatami_test::test_indexed_access(*forced_bound_dense, *manual, 0, 1, options);

        tatami_test::test_full_access(*forced_bound_sparse, *manual, options);
        tatami_test::test_block_access(*forced_bound_sparse, *manual, 0, 0, options);
        tatami_test::test_indexed_access(*forced_bound_sparse, *manual, 0, 1, options);
    }

    tatami_test::test_unsorted_full_access(*uns_bound_sparse, options);
    tatami_test::test_unsorted_block_access(*uns_bound_sparse, 0, 0, options);
    tatami_test::test_unsorted_indexed_access(*uns_bound_sparse, 0, 1, options);
}

INSTANTIATE_TEST_SUITE_P(
    DelayedBind,
    DelayedBindEmptyAccessTest,
    ::testing::Combine(
        ::testing::Values(true, false), // bind by row or by column
        ::testing::Values(true, false), // access by row or column
        ::testing::Values(true, false)  // whether to use an oracle
    )
);

/****************************
 ****************************/

class DelayedBindFullAccessTest : 
    public ::testing::TestWithParam<std::tuple<typename DelayedBindUtils::SimulationParameters, tatami_test::StandardTestAccessOptions> >,
    public DelayedBindUtils {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DelayedBindFullAccessTest, Basic) {
    auto tparam = GetParam();
    auto options = tatami_test::convert_test_access_options(std::get<1>(tparam));

    tatami_test::test_full_access(*bound_dense, *manual, options);
    tatami_test::test_full_access(*bound_sparse, *manual, options);

    if (options.use_oracle) {
        tatami_test::test_full_access(*forced_bound_dense, *manual, options);
        tatami_test::test_full_access(*forced_bound_sparse, *manual, options);
    }

    tatami_test::test_unsorted_full_access(*uns_bound_sparse, options);
}

INSTANTIATE_TEST_SUITE_P(
    DelayedBind,
    DelayedBindFullAccessTest,
    ::testing::Combine(
        DelayedBindUtils::spawn_bind_scenarios(),
        tatami_test::standard_test_access_options_combinations()
    )
);

/****************************
 ****************************/

class DelayedBindBlockAccessTest : 
    public ::testing::TestWithParam<std::tuple<typename DelayedBindUtils::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >,
    public DelayedBindUtils {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DelayedBindBlockAccessTest, Basic) {
    auto tparam = GetParam();
    auto options = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto interval_info = std::get<2>(tparam);

    tatami_test::test_block_access(*bound_dense, *manual, interval_info.first, interval_info.second, options);
    tatami_test::test_block_access(*bound_sparse, *manual, interval_info.first, interval_info.second, options);

    if (options.use_oracle) {
        tatami_test::test_block_access(*forced_bound_sparse, *manual, interval_info.first, interval_info.second, options);
        tatami_test::test_block_access(*forced_bound_dense, *manual, interval_info.first, interval_info.second, options);
    }

    tatami_test::test_unsorted_block_access(*uns_bound_sparse, interval_info.first, interval_info.second, options);
}

INSTANTIATE_TEST_SUITE_P(
    DelayedBind,
    DelayedBindBlockAccessTest,
    ::testing::Combine(
        DelayedBindUtils::spawn_bind_scenarios(),
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.6), 
            std::make_pair(0.25, 0.51), 
            std::make_pair(0.55, 0.45)
        )
    )
);

/****************************
 ****************************/

class DelayedBindIndexedAccessTest : 
    public ::testing::TestWithParam<std::tuple<typename DelayedBindUtils::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >,
    public DelayedBindUtils {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DelayedBindIndexedAccessTest, Basic) {
    auto tparam = GetParam();
    auto options = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto interval_info = std::get<2>(tparam);

    tatami_test::test_indexed_access(*bound_dense, *manual, interval_info.first, interval_info.second, options);
    tatami_test::test_indexed_access(*bound_sparse, *manual, interval_info.first, interval_info.second, options);

    if (options.use_oracle) {
        tatami_test::test_indexed_access(*forced_bound_sparse, *manual, interval_info.first, interval_info.second, options);
        tatami_test::test_indexed_access(*forced_bound_dense, *manual, interval_info.first, interval_info.second, options);
    }

    tatami_test::test_unsorted_indexed_access(*uns_bound_sparse, interval_info.first, interval_info.second, options);
}

INSTANTIATE_TEST_SUITE_P(
    DelayedBind,
    DelayedBindIndexedAccessTest,
    ::testing::Combine(
        DelayedBindUtils::spawn_bind_scenarios(),
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.2), 
            std::make_pair(0.33, 0.3),
            std::make_pair(0.5, 0.5)
        )
    )
);
