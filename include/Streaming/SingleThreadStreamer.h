/*! \file SingleThreadStamper.h*/
//
// Created by tony on 23/06/23.
//

#ifndef INTELLISTREAM_SINGLETHREADSTREAMER_H
#define INTELLISTREAM_SINGLETHREADSTREAMER_H

#include <Streaming/TimeStamper.h>
#include <CPPAlgos/CPPAlgoTable.h>
#include <Parallelization/BlockPartitionRunner.h>

namespace AMMBench {

/**
 * @ingroup AMMBENCH_STREAMING
 * @{
 *
 */
    /**
      * @class SingleThreadStreamer Streaming/SingleThreadStreamer.h
      * @brief The class to run streaming amm under single thread, let each row of A coming in a streaming manner
      * @ingroup AMMBENCH_STREAMING
      * @note  Default behavior
      */
    class SingleThreadStreamer {
    protected:
        INTELLI::ConfigMapPtr cfgGlobal;
        AMMBench::CPPAlgoTable cppAlgoTable;
        uint64_t batchSize = 1;
        AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = nullptr;
        AMMBench::TensorPtr matC = nullptr;
        double throughput = 0.0;
    public:
        SingleThreadStreamer() {}

        ~SingleThreadStreamer() {}

        /**
         * @brief the timestamps to trace the streaming process
         */
        std::vector<AMMBench::AMMTimeStampPtr> myTs;

        /**
     * @brief Set the GLOBAL config map related to this TimerStamper
     * @param cfg The config map
      * @return bool whether the config is successfully set
     */
        virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

        /**
     * @brief To run a streaming Amm, assuming the rows of A coming in a streaming manner and B is fixed
     *  @param A The A matrix
    * @param B The B matrix
      * @return bool whether the config is successfully set
     */
        virtual torch::Tensor streamingAmm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize = 1);

        /**
         * @brief to get the throughput of last streaming process, the unit is rows/second
         * @return the throughput
         */
        double getThroughput() {
            return throughput;
        }

        /**
         * @brief to get the latency within some fraction, such as 0.95
         * @param fraction the 0~1 fraction
         * @return the latency in us
         */
        double getLatencyPercentage(double fraction);

    };
/**
 * @}
 */
} // AMMBench

#endif //INTELLISTREAM_SINGLETHREADSTREAMER_H
