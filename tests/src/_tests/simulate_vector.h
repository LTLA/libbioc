#ifndef SIMULATE_VECTOR_H
#define SIMULATE_VECTOR_H

#include <random>
#include <vector>
#include <memory>

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DenseMatrix.hpp"

template<typename T>
std::vector<T> simulate_dense_vector(size_t length, double lower = 0, double upper = 100, size_t seed = 1234567890) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<> unif(lower, upper);
    std::vector<T> values(length);
    for (auto& v : values) {
        v = unif(rng);
    }
    return values;
}

template<typename T>
std::vector<T> simulate_sparse_vector(size_t length, double density, double lower = -10, double upper = 10, size_t seed = 1234567890) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<> nonzero(0.0, 1.0);
    std::uniform_real_distribution<> unif(lower, upper);
    std::vector<T> values(length);
    for (auto& v : values) {
        if (nonzero(rng) < density) {
            v = unif(rng);
        }
    }
    return values;
}

#endif
