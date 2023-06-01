//
// Created by luv on 5/30/23.
//

#ifndef INTELLISTREAM_TUGOFWARCPPALGO_H
#define INTELLISTREAM_TUGOFWARCPPALGO_H

#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms writtrn in c++
 * @{
 */
/**
 * @class CRSCPPlgo CPPAlgos/TugOfWarCPPAlgo.h
 * @brief The cloumn row sampling (CRS) class of c++ algos
 *
 */
    class TugOfWarCPPAlgo : public AMMBench::AbstractCPPAlgo {
    public:
        TugOfWarCPPAlgo() {

        }

        ~TugOfWarCPPAlgo() {

        }

        /**
         * @brief the virtual function provided for outside callers, rewrite in children classes
         * @param A the A matrix
         * @param B the B matrix
         * @param sketchSize the size of sketc or sampling
         * @return the output c matrix
         */
        virtual torch::Tensor amm(torch::Tensor A, torch::Tensor B, int sketchSize);

      private:
        torch::Tensor generateTugOfWarMatrix(int64_t m, int64_t n);

    };

/**
 * @ingroup AMMBENCH_CppAlgos
 * @typedef AbstractMatrixCppAlgoPtr
 * @brief The class to describe a shared pointer to @ref TugOfWarCppAlgo

 */
    typedef std::shared_ptr<class AMMBench::TugOfWarCPPAlgo> TugOfWarCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newTugOfWarCppAlgo
 * @brief (Macro) To creat a new @ref  TugOfWarCppAlgounder shared pointer.
 */
#define newTugOfWarCPPAlgo std::make_shared<AMMBench::TugOfWarCPPAlgo>
}
/**
 * @}
 */
#endif //INTELLISTREAM_TUGOFWARCPPALGO_H
