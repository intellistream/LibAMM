//
// Created by luv on 5/28/23.
//

#ifndef INTELLISTREAM_COUNTSKETCHCPPALGO_H
#define INTELLISTREAM_COUNTSKETCHCPPALGO_H

#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms writtrn in c++
 * @{
 */
/**
 * @class CountSketchCPPAlgo CPPAlgos/CountSketchCPPAlgo.h
 * @brief The cloumn row sampling (CRS) class of c++ algos
 *
 */
    class CountSketchCPPAlgo : public AMMBench::AbstractCPPAlgo {
    public:
        CountSketchCPPAlgo() {

        }

        ~CountSketchCPPAlgo() {

        }

        /**
         * @brief the virtual function provided for outside callers, rewrite in children classes
         * @param A the A matrix
         * @param B the B matrix
         * @param sketchSize the size of sketc or sampling
         * @return the output c matrix
         */
        virtual torch::Tensor amm(torch::Tensor A, torch::Tensor B, int sketchSize);

    };

/**
 * @ingroup AMMBENCH_CppAlgos
 * @typedef AbstractMatrixCppAlgoPtr
 * @brief The class to describe a shared pointer to @ref CountSketchCPPAlgo

 */
    typedef std::shared_ptr<class AMMBench::CountSketchCPPAlgo> CountSketchCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newCRSV2CppAlgo
 * @brief (Macro) To creat a new @ref  CRSV2CppAlgounder shared pointer.
 */
#define newCountSketchCPPAlgo std::make_shared<AMMBench::CountSketchCPPAlgo>
}
/**
 * @}
 */
#endif //INTELLISTREAM_COUNTSKETCHCPPALGO_H