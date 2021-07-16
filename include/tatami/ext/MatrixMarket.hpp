#ifndef SCRAN_LOAD_MATRIX_MARKET_HPP
#define SCRAN_LOAD_MATRIX_MARKET_HPP

#include <limits>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cctype>

#include "../base/CompressedSparseMatrix.hpp"
#include "../base/DelayedBind.hpp"
#include "../utils/compress_sparse_triplets.hpp"

#ifdef TATAMI_USE_ZLIB
#include "zlib.h"
#include <array>
#endif

namespace tatami {

namespace MatrixMarket {

template<typename T>
size_t process_triplet_line(const char* buffer, T& arg1, T& arg2, T& arg3, size_t n) {
    auto read = [=](size_t& i, T& arg) -> bool {
        // These had better be positive integers, 
        // otherwise this will throw a bunch of errors.
        while (i < n && std::isspace(buffer[i])) { 
            ++i;
        }
        if (i == n) {
            return false;
        }

        auto prev = i;
        arg = 0;
        while (i < n && std::isdigit(buffer[i])) {
            arg *= 10;
            arg += (buffer[i] - '0');
            ++i;
        }
        if (i == n || i == prev) {
            return false;
        }

        if (i < n && !std::isspace(buffer[i]) && buffer[i] != '\0') {
            throw std::runtime_error("values should be non-negative integers");
        }
        return true;
    };

    size_t i = 0;
    if (!read(i, arg1)) {
        return 0;
    }
    if (!read(i, arg2)) {
        return 0;
    }
    if (!read(i, arg3)) {
        return 0;
    }

    return i + 1;
}

inline size_t read_to_eol(const char * buffer, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (buffer[i] == '\n' || buffer[i] == '\0') {
            return i + 1;
        }
    }
    return 0; // out of range
}

struct LineAssignments {
    LineAssignments() : rows_per_category(3), lines_per_category(3) {}

    std::vector<uint8_t> category;
    std::vector<size_t> index;
    std::vector<size_t> rows_per_category;
    std::vector<size_t> lines_per_category;
    std::vector<size_t> lines_per_row;
    std::vector<size_t> permutation;

    size_t nrows, ncols, nlines;
    bool passed_preamble = false;

    size_t add(const char* buffer, size_t n) {
        if (buffer[0] == '%') {
            // TODO: should probably check for 'coordinate integer'.
            return read_to_eol(buffer, n);

        } else if (!passed_preamble) {
            passed_preamble = true;
            auto read = process_triplet_line(buffer, nrows, ncols, nlines, n);
            if (read == 0) {
                return 0;
            }

            category.resize(nrows);
            index.resize(nrows);
            permutation.resize(nrows);
            lines_per_row.resize(nrows);
            return read;
        }

        // Assigning each line to a block based on its integer size.
        constexpr int max8 = std::numeric_limits<uint8_t>::max();
        constexpr int max16 = std::numeric_limits<uint16_t>::max();

        int row, col, data;
        auto read = process_triplet_line(buffer, row, col, data, n);
        if (read == 0){
            return 0;
        }
       
        --row; // 1-based.
        if (data > max16) {
            category[row] = std::max(category[row], static_cast<uint8_t>(2));
        } else if (data > max8) {
            category[row] = std::max(category[row], static_cast<uint8_t>(1));
        }
        ++lines_per_row[row];

        return read;
    }

    void finish() {
        // Computing the number of features and lines in each block.
        auto iIt = index.begin();
        auto cIt = lines_per_row.begin();
        for (auto f : category) {
            auto& current = rows_per_category[f];

            (*iIt) = current;
            ++current;
            ++iIt;

            lines_per_category[f] += *cIt;
            ++cIt;
        }

        // Computing the permutation.
        auto cumsum = rows_per_category;
        size_t last = 0;
        for (auto& x : cumsum) {
            std::swap(x, last);
            last += x;
        }
        for (size_t i = 0; i < nrows; ++i) {
            permutation[i] = cumsum[category[i]] + index[i];
        }

        return;
    }
};

template<typename ROW>
struct LayeredBuilder {
private:
    LineAssignments assign;

    std::vector<ROW>      row8;
    std::vector<uint32_t> col8;
    std::vector<uint8_t>  dat8;
    size_t counter8 = 0;

    std::vector<ROW>      row16;
    std::vector<uint32_t> col16;
    std::vector<uint16_t> dat16;
    size_t counter16 = 0;

    std::vector<ROW>      row32;
    std::vector<uint32_t> col32;
    std::vector<uint32_t> dat32;
    size_t counter32 = 0;

    bool passed_preamble = false;
public:
    LayeredBuilder(LineAssignments ass) : 
        assign(std::move(ass)),
        row8(assign.lines_per_category[0]),
        col8(assign.lines_per_category[0]),
        dat8(assign.lines_per_category[0]),
        row16(assign.lines_per_category[1]),
        col16(assign.lines_per_category[1]),
        dat16(assign.lines_per_category[1]),
        row32(assign.lines_per_category[2]),
        col32(assign.lines_per_category[2]),
        dat32(assign.lines_per_category[2]) {}

public:
    size_t add(const char* buffer, size_t n) {
        if (buffer[0] == '%') {
            return read_to_eol(buffer, n);
        } else if (!passed_preamble) {
            passed_preamble = true;
            return read_to_eol(buffer, n);
        }

        int row, col, data;
        auto read = process_triplet_line(buffer, row, col, data, n);
        if (read == 0){
            return 0;
        }

        --row; // 1-based.
        --col; 

        auto idx = assign.index[row];
        switch (assign.category[row]) {
        case 0:
            row8[counter8] = idx;
            col8[counter8] = col;
            dat8[counter8] = data;
            ++counter8;
            break;
        case 1:
            row16[counter16] = idx;
            col16[counter16] = col;
            dat16[counter16] = data;
            ++counter16;
            break;
        case 2:
            row32[counter32] = idx;
            col32[counter32] = col;
            dat32[counter32] = data;
            ++counter32;
            break;
        }
        
        return read;
    }

private:
    template<typename T, typename IDX, typename U, typename V, typename W>
    std::shared_ptr<Matrix<T, IDX> > create_sparse_matrix(size_t nr, size_t nc, U& values, V& rows, W& cols) {
        auto indptrs = compress_sparse_triplets<false>(nr, nc, values, rows, cols);
        return std::shared_ptr<Matrix<T, IDX> >(new CompressedSparseColumnMatrix<T, IDX, U, V, decltype(indptrs)>(nr, nc, std::move(values), std::move(rows), std::move(indptrs)));
    }

public:
    template<typename T, typename IDX>
    std::shared_ptr<Matrix<T, IDX> > finish() {
        std::vector<std::shared_ptr<Matrix<T, IDX> > > collated;
        if (assign.rows_per_category[0]) {
            collated.push_back(create_sparse_matrix<T, IDX>(assign.rows_per_category[0], assign.ncols, dat8, row8, col8));
        }
        if (assign.rows_per_category[1]) {
            collated.push_back(create_sparse_matrix<T, IDX>(assign.rows_per_category[1], assign.ncols, dat16, row16, col16));
        }
        if (assign.rows_per_category[2]) {
            collated.push_back(create_sparse_matrix<T, IDX>(assign.rows_per_category[2], assign.ncols, dat32, row32, col32));
        }

        if (collated.size() == 0) {
            return create_sparse_matrix<T, IDX>(0, assign.ncols, dat8, row8, col8);
        } else if (collated.size() == 1) { 
            return collated[0];
        } else {
            return make_DelayedBind<0>(std::move(collated));
        }
    }
};

template<typename T = double, typename IDX = int>
struct LayeredMatrixData {
    std::shared_ptr<Matrix<T, IDX> > matrix;
    std::vector<size_t> permutation;
};

template<typename T = double, typename IDX = int>
LayeredMatrixData<T, IDX> load_layered_sparse_matrix(const char * filepath) {
    auto process = [=](auto& obj) -> void {
        std::ifstream in(filepath);
        if (!in) {
            throw std::runtime_error("failed to open file");
        }
        std::string line;
        int counter = 0;
        while (std::getline(in, line)) {
            auto status = obj.add(line.c_str(), line.size() + 1); // for the null terminator
            if (!status) {
                throw std::runtime_error(std::string("failed to process line ") + std::to_string(counter + 1));
            }
            ++counter;
        }
        return;
    };

    LineAssignments ass;
    process(ass);
    ass.finish();

    LayeredMatrixData<T, IDX> output;
    output.permutation = ass.permutation;

    constexpr size_t max16 = std::numeric_limits<uint16_t>::max();
    if (ass.nrows > max16) {
        LayeredBuilder<uint16_t> builder(std::move(ass));
        process(builder);
        output.matrix = builder.template finish<T, IDX>();
    } else {
        LayeredBuilder<IDX> builder(std::move(ass));
        process(builder);
        output.matrix = builder.template finish<T, IDX>();
    }

    return output;
}

#ifdef TATAMI_USE_ZLIB

// Stolen from 'inf()' at http://www.zlib.net/zpipe.c,
// with some shuffling of code to make it a bit more C++-like.
struct Unzlibber {
    struct ZStream {
        ZStream() {
            /* allocate inflate state */
            strm.zalloc = Z_NULL;
            strm.zfree = Z_NULL;
            strm.opaque = Z_NULL;
            strm.avail_in = 0;
            strm.next_in = Z_NULL;

            // https://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib
            int ret = inflateInit2(&strm, 16+MAX_WBITS); 
            if (ret != Z_OK) {
                throw 1;
            }
        }

        ~ZStream() {
            (void)inflateEnd(&strm);
            return;
        }

        // Delete the remaining constructors.
        ZStream(const ZStream&) = delete;
        ZStream(ZStream&&) = delete;
        ZStream& operator=(const ZStream&) = delete;
        ZStream& operator=(ZStream&&) = delete;

        z_stream strm;
    };

    template<class OBJECT>
    void operator()(const char* filepath, OBJECT& obj) {
        std::ifstream source(filepath, std::ios::binary);
        if (!source) {
            throw std::runtime_error("failed to open file");
        }

        constexpr int bufsize = 16384; // 262144;
        std::array<unsigned char, bufsize> input;
        std::array<unsigned char, bufsize + 100> output; // give enough space for leftovers and a safety NULL at EOF, see below.

        ZStream zstr;
        size_t leftovers = 0;
        int ret = 0;

        /* decompress until deflate stream ends or end of file */
        do {
            source.read((char*)input.data(), bufsize);
            zstr.strm.avail_in = source.gcount();
            if (!source && zstr.strm.avail_in == 0) {
                break;
            }
            zstr.strm.next_in = input.data(); // this needs to be done every loop. Guess it modifies it in place.

            /* run inflate() on input until output buffer not full */
            do {
                zstr.strm.avail_out = bufsize;
                zstr.strm.next_out = output.data() + leftovers;
                ret = inflate(&(zstr.strm), Z_NO_FLUSH);
                assert(ret != Z_STREAM_ERROR);  /* state not clobbered */

                switch (ret) {
                case Z_NEED_DICT:
                    ret = Z_DATA_ERROR; /* and fall through */
                case Z_DATA_ERROR:
                case Z_MEM_ERROR:
                    throw 1;
                }

                size_t current_stored = bufsize + leftovers - zstr.strm.avail_out;

                // Making sure we have a terminating newline.
                if (ret == Z_STREAM_END && current_stored && output[current_stored-1]!='\n') {
                    output[current_stored] = '\n';
                    ++current_stored;
                }

                // Adding whole lines.
                size_t last_processed = 0, total_processed = 0;
                do {
                    last_processed = obj.add((char*)output.data() + total_processed, current_stored - total_processed);
                    total_processed += last_processed;
                } while (last_processed);

                // Rotating what's left to the front for the next cycle.
                leftovers = current_stored - total_processed;
                for (size_t i = 0; i < leftovers; ++i) {
                    output[i] = output[total_processed + i];
                }

            } while (zstr.strm.avail_out == 0);
        } while (ret != Z_STREAM_END); /* done when inflate() says it's done */

        /* clean up and return */
        if (ret != Z_STREAM_END) {
            throw 1;
        }
        return;
    }
};

template<typename T = double, typename IDX = int>
LayeredMatrixData<T, IDX> load_layered_sparse_matrix_gzip(const char * filepath) {
    Unzlibber unz;

    LineAssignments ass;
    unz(filepath, ass);
    ass.finish();

    LayeredMatrixData<T, IDX> output;
    output.permutation = ass.permutation;

    constexpr size_t max16 = std::numeric_limits<uint16_t>::max();
    if (ass.nrows > max16) {
        LayeredBuilder<uint16_t> builder(std::move(ass));
        unz(filepath, builder);
        output.matrix = builder.template finish<T, IDX>();
    } else {
        LayeredBuilder<IDX> builder(std::move(ass));
        unz(filepath, builder);
        output.matrix = builder.template finish<T, IDX>();
    }

    return output;
}

#endif

}

}

#endif
