//
// Created by haolan on 5/29/23.
//

#ifndef INTELLISTREAM_BCRSCPPALGO_H
#define INTELLISTREAM_BCRSCPPALGO_H
#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms writtrn in c++
 * @{
 */
/**
 * @class CRSCPPlgo CPPAlgos/BCRSCPPAlgo.h
 * @brief The cloumn row sampling (CRS) class of c++ algos
 *
 */
    class BCRSCPPAlgo : public AMMBench::AbstractCPPAlgo {
    public:
        BCRSCPPAlgo() {

        }

        ~BCRSCPPAlgo() {

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
 * @brief The class to describe a shared pointer to @ref BCRSCppAlgo

 */
    typedef std::shared_ptr<class AMMBench::BCRSCPPAlgo> BCRSCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newBCRSCppAlgo
 * @brief (Macro) To creat a new @ref  BCRSCppAlgounder shared pointer.
 */
#define newBCRSCPPAlgo std::make_shared<AMMBench::BCRSCPPAlgo>
}
/**
 * @}
 */
#endif //INTELLISTREAM_BCRSCPPALGO_H
