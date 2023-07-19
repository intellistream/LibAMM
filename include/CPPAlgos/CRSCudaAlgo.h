/*! \file CRSCudaAlgo.h*/
//
// Created by tony on 25/05/23.
//

#ifndef INTELLISTREAM_INCLUDE_CPPALGOS_CRSCudaAlgo_H_
#define INTELLISTREAM_INCLUDE_CPPALGOS_CRSCudaAlgo_H_

#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms written in c++
 * @{
 */
/**
 * @class CRSCudaAlgo CPPAlgos/CRSCudaAlgo.h
 * @brief The column row sampling (CRS) class of cuda-based algos
 *
 */
    class CRSCudaAlgo : public AMMBench::AbstractCPPAlgo {
    public:
        CRSCudaAlgo() {

        }

        ~CRSCudaAlgo() {

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
 * @brief The class to describe a shared pointer to @ref CRSCppAlgo

 */
    typedef std::shared_ptr<class AMMBench::CRSCudaAlgo> CRSCudaAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newCRSCppAlgo
 * @brief (Macro) To creat a new @ref  CRSCppAlgounder shared pointer.
 */
#define newCRSCudaAlgo std::make_shared<AMMBench::CRSCudaAlgo>
}
/**
 * @}
 */

#endif //INTELLISTREAM_INCLUDE_CPPALGOS_CRSCppAlgo_H_

