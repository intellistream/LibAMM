//
// Created by haolan on 5/30/23.
//

#ifndef INTELLISTREAM_BETACOOFDCPPALGO_H
#define INTELLISTREAM_BETACOOFDCPPALGO_H
#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms written in c++
 * @{
 */
/**
 * @class CPPAlgos/BetaCoOFDCPPAlgo.h
 * @brief The Beta Co-Occurring FD AMM class of c++ algos
 *
 */
    class BetaCoOFDCPPAlgo : public AMMBench::AbstractCPPAlgo {
    public:
        BetaCoOFDCPPAlgo() {

        }

        ~BetaCoOFDCPPAlgo() {

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
 * @brief The class to describe a shared pointer to @ref BetaCoOFDCppAlgo

 */
    typedef std::shared_ptr<class AMMBench::BetaCoOFDCPPAlgo> BetaCoOFDCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newBetaCoOFDCppAlgo
 * @brief (Macro) To creat a new @ref  BetaCoOFDCppAlgounder shared pointer.
 */
#define newBetaCoOFDCPPAlgo std::make_shared<AMMBench::BetaCoOFDCPPAlgo>
}
/**
 * @}
 */
#endif //INTELLISTREAM_BETACOOFDCPPALGO_H
