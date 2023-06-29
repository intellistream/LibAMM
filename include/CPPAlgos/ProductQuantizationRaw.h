/*! \file ProductQuantizationRaw.h*/
//
// Created by haolan on 22/6/23.
//

#ifndef AMMBENCH_PRODUCTQUANTIZATIONRAW_H
#define AMMBENCH_PRODUCTQUANTIZATIONRAW_H
#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms written in c++
 * @{
 */
/**
 * @class ProductQuantizationRaw CPPAlgos/ProductQuantizationRaw.h
 * @brief The Product Quantization AMM class of c++ algos, using Euclidean distance
 *
 */
    class ProductQuantizationRaw : public AMMBench::AbstractCPPAlgo {
    public:
        ProductQuantizationRaw() {

        }

        ~ProductQuantizationRaw() {

        }

        /**
         * @brief the virtual function provided for outside callers, rewrite in children classes
         * @param A the A matrix
         * @param B the B matrix
         * @param sketchSize the size of sketc or sampling
         * @return the output c matrix
         */
        virtual torch::Tensor amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize);

    };

/**
 * @ingroup AMMBENCH_CppAlgos
 * @typedef AbstractMatrixCppAlgoPtr
 * @brief The class to describe a shared pointer to @ref ProductQuantizationRawAlgo

 */
    typedef std::shared_ptr<class AMMBench::ProductQuantizationRaw> ProductQuantizationRawPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newProductQuantizationRawAlgo
 * @brief (Macro) To creat a new @ref  ProductQuantizationRawAlgounder shared pointer.
 */
#define newProductQuantizationRawAlgo std::make_shared<AMMBench::ProductQuantizationRaw>
}
/**
 * @}
 */
#endif //AMMBENCH_PRODUCTQUANTIZATIONRAW_H
