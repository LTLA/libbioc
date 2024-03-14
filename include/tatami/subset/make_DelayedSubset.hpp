#ifndef TATAMI_MAKE_DELAYED_SUBSET_HPP
#define TATAMI_MAKE_DELAYED_SUBSET_HPP

#include "DelayedSubsetSortedUnique.hpp"
#include "DelayedSubsetSorted.hpp"
#include "DelayedSubsetUnique.hpp"
#include "DelayedSubset.hpp"
#include "DelayedSubsetBlock.hpp"
#include "../utils/ArrayView.hpp"

#include <algorithm>
#include <memory>

/**
 * @file make_DelayedSubset.hpp
 *
 * @brief Make a delayed subset wrapper based on row/column indices.
 */

namespace tatami {

/**
 * A `make_*` helper function to enable partial template deduction of supplied types.
 * This will automatically dispatch to `DelayedSubsetSortedUnique`, `DelayedSubsetUnique`, `DelayedSubsetSorted` or `DelayedSubset`, depending on the values in `idx`.
 *
 * @tparam margin_ Dimension along which the subsetting is to occur.
 * If 0, the subset is applied to the rows; if 1, the subset is applied to the columns.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam IndexStorage_ Vector containing the subset indices, to be automatically deduced.
 * Any class implementing `[`, `size()`, `begin()` and `end()` can be used here.
 *
 * @param p Pointer to a (possibly `const`) `Matrix`.
 * @param idx Instance of the index vector.
 *
 * @return A pointer to a `DelayedSubset` instance.
 */
template<int margin_, typename Value_, typename Index_, class IndexStorage_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedSubset(std::shared_ptr<const Matrix<Value_, Index_> > p, IndexStorage_ idx) {
    bool is_unsorted = false;
    for (Index_ i = 0, end = idx.size(); i < end; ++i) {
        if (i) {
            if (idx[i] < idx[i-1]) {
                is_unsorted = true;
                break;
            }
        }
    }

    if (!is_unsorted) {
        bool has_duplicates = false;
        for (Index_ i = 0, end = idx.size(); i < end; ++i) {
            if (i) {
                if (idx[i] == idx[i-1]) {
                    has_duplicates = true;
                    break;
                }
            }
        }

        if (!has_duplicates) {
            bool consecutive = true;
            for (Index_ i = 0, end = idx.size(); i < end; ++i) {
                if (idx[i] > idx[i-1] + 1) {
                    consecutive = false;
                    break;
                }
            }

            if (consecutive) {
                auto start = (idx.size() ? idx[0] : 0);
                return std::shared_ptr<Matrix<Value_, Index_> >(
                    new DelayedSubsetBlock<margin_, Value_, Index_>(std::move(p), start, idx.size())
                );
            } else {
                return std::shared_ptr<Matrix<Value_, Index_> >(
                    new DelayedSubsetSortedUnique<margin_, Value_, Index_, IndexStorage_>(std::move(p), std::move(idx), false)
                );
            }
        } else {
            return std::shared_ptr<Matrix<Value_, Index_> >(
                new DelayedSubsetSorted<margin_, Value_, Index_, IndexStorage_>(std::move(p), std::move(idx), false)
            );
        }
    }

    bool has_duplicates = false;
    std::vector<unsigned char> accumulated(margin_ == 0 ? p->nrow() : p->ncol());
    for (Index_ i = 0, end = idx.size(); i < end; ++i) {
        auto& found = accumulated[idx[i]];
        if (found) {
            has_duplicates = true;
            break;
        } else {
            found = 1;
        }
    }

    if (!has_duplicates) {
        return std::shared_ptr<Matrix<Value_, Index_> >(
            new DelayedSubsetUnique<margin_, Value_, Index_, IndexStorage_>(std::move(p), std::move(idx), false)
        );
    } else {
        return std::shared_ptr<Matrix<Value_, Index_> >(
            new DelayedSubset<margin_, Value_, Index_, IndexStorage_>(std::move(p), std::move(idx))
        );
    }
}

/**
 * @cond
 */
template<int margin_, typename Value_, typename Index_, class IndexStorage_>
std::shared_ptr<Matrix<Value_, Index_> > make_DelayedSubset(std::shared_ptr<Matrix<Value_, Index_> > p, IndexStorage_ idx) {
    return make_DelayedSubset<margin_, Value_, Index_, IndexStorage_>(std::shared_ptr<const Matrix<Value_, Index_> >(std::move(p)), std::move(idx));
}
/**
 * @endcond
 */

}

#endif
