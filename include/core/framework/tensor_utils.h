//
// Created by hacht on 2/24/17.
//

#pragma once

#include <algorithm>  // std::fill, std::generate
#include <cmath>      // sqrt
#include <numeric>    // std::accumulate

#include "tensor.h"
#include "cblas.h"
#include "../../util/util.h"

namespace simpleCNN {

    template<typename T>
    void print(T t)
    {
        coloredPrint(Color::GREEN, std::string("[INFO] ")); // fancy
        std::cout << t << std::endl;
    }

    template<typename T>
    void print_pack(std::ostream& out, T t)
    {  // TODO: C++17 allows easier printing
        out << t;
    }

    template<typename T, typename U, typename... Args>
    void print_pack(std::ostream& out, T t, U u, Args... args)
    {
        out << t << ',';
        print_pack(out, u, args...);
    }

    template<typename T, size_t kDim, typename... Args>
    inline std::ostream& print_last_two_dimesions(std::ostream& os,
            const Tensor<T, kDim>& tensor,
            const Args... args
    )
    {
        const std::array<size_t, kDim>& shape = tensor.shape();
        for (size_t k = 0; k<shape[kDim-2]; ++k)
        {
            for (size_t l = 0; l<shape[kDim-1]; ++l)
            {
                os << "\t" << tensor.host_at(args..., k, l);
            }
            os << "\n";
        }
        return os;
    }

    template<typename T,
            size_t kDim,
            typename... Args,
            typename std::enable_if<sizeof...(Args)==kDim-3, int>::type = 0>
    inline std::ostream& print_last_n_dimesions(std::ostream& os,
            const Tensor<T, kDim>& tensor,
            const int d,
            const Args... args
    )
    {
        // const std::array<size_t, kDim>& shape = tensor.shape();
        // const size_t n_dim = sizeof...(args);
        os << "Tensor(";
        print_pack(os, d, args...);
        os << ",:,:):\n";
        print_last_two_dimesions(os, tensor, d, args...);
        return os;
    }

    template<typename T,
            size_t kDim,
            typename... Args,
            typename std::enable_if<(sizeof...(Args)<kDim-3), int>::type = 0>
    inline std::ostream& print_last_n_dimesions(std::ostream& os,
            const Tensor<T, kDim>& tensor,
            const int d,
            const Args... args
    )
    {
        const std::array<size_t, kDim>& shape = tensor.shape();
        const size_t n_dim = sizeof...(args);
        for (size_t k = 0; k<shape[n_dim+1]; ++k)
        {
            print_last_n_dimesions(os, tensor, d, args..., k);
        }
        return os;
    }

/**
 * Overloaded method to print the Tensor class to the standard output
 * @param os
 * @param tensor
 * @return
 */
    template<typename T, size_t kDim>
    inline std::ostream& operator<<(std::ostream& os,
            const Tensor<T, kDim>& tensor)
    {
        const std::array<size_t, kDim>& shape = tensor.shape();
        for (size_t i = 0; i<shape[0]; ++i) print_last_n_dimesions(os, tensor, i);
        return os;
    }

    /**
     * OpenBLAS efficient matrix multiplication for Tensors of dimension 2. This version supports
     * floating point precision only, double precision also exists but is not going to be used here.
     * The wrapper calling function is right below this function. The calling function takes
     * @param A Matrix with floats (pointed to as address of first element)
     * @param B Matrix with floats
     * @param AB (The result) Resulting matrix with floats
     * @param transpose_A If A needs to be transposed such that AB exists.
     * @param transpose_B If B needs to be transposed such that AB exists.
     *
     **/ //--------------------------------------------------------//
    void matrix_multiplcation(float* A, int A_width, int A_height,
            float* B, int B_width, int B_height,
            float* AB, bool tA, bool tB, float beta)
    {
        int A_height_new = tA ? A_width : A_height;
        int A_width_new = tA ? A_height : A_width;
        int B_height_new = tB ? B_width : B_height;
        int B_width_new = tB ? B_height : B_width;
        int m = A_height_new;
        int n = B_width_new;
        int k = A_width_new;
        // Error, width and height should match!
        assert(A_width_new==B_height_new);
        int lda = tA ? m : k;
        int ldb = tB ? k : n;
#define TRANSPOSE(X) ((X) ? CblasTrans : CblasNoTrans)
        // http://www.netlib.org/lapack/explore-html/d7/d2b/sgemm_8f.html
        cblas_sgemm(CblasRowMajor, // heey, I know you
                TRANSPOSE(tA), TRANSPOSE(tB),
                m, n, k, 1.0f,
                A, lda,
                B, ldb,
                beta,
                AB, n);

#undef TRANSPOSE(X)
    }

    void multiply_2_dim_tensors_float(
            matrix_t& A,
            matrix_t& B,
            matrix_t& AB,
            bool transpose_A,
            bool transpose_B)
    {
        matrix_multiplcation(
                &A.host_at(0, 0),
                A.cols(),
                A.rows(),
                &B.host_at(0, 0),
                B.cols(),
                B.rows(),
                &AB.host_at(0, 0), // A * B
                transpose_A,
                transpose_B,
                0);
    }
} // namespace simpleCNN
