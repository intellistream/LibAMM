/*! \file ProductQuantizationHash.h*/
//
// Created by haolan on 25/6/23.
//

#ifndef INTELLISTREAM_PRODUCTQUANTIZATIONHASH_H
#define INTELLISTREAM_PRODUCTQUANTIZATIONHASH_H
#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms written in c++
 * @{
 */
/**
 * @class ProductQuantizationHash CPPAlgos/ProductQuantizationHash.h
 * @brief The Product Quantization AMM class of c++ algos, using hash function to find matching prototypes
 *
 */
    class ProductQuantizationHash : public AMMBench::AbstractCPPAlgo {

    protected:
        string prototypesLoadPath;
        string hashLoadPath;

    public:
        ProductQuantizationHash() {

        }

        ~ProductQuantizationHash() {

        }

        /**
         * @brief set the alo-specfic config related to one algorithm
         * @param prototypesLoadPath where to load prototypes
         * @param hashLoadPath where to load hash
         */
        virtual void setConfig(INTELLI::ConfigMapPtr cfg);

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
 * @brief The class to describe a shared pointer to @ref ProductQuantizationHashAlgo

 */
    typedef std::shared_ptr<class AMMBench::ProductQuantizationHash> ProductQuantizationHashPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newProductQuantizationHashAlgo
 * @brief (Macro) To creat a new @ref  ProductQuantizationHashAlgounder shared pointer.
 */
#define newProductQuantizationHashAlgo std::make_shared<AMMBench::ProductQuantizationHash>
}
/**
 * @}
 */
#endif //INTELLISTREAM_PRODUCTQUANTIZATIONHASH_H
