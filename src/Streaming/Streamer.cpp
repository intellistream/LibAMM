//
// Created by haolan on 8/7/23.
//

#include <Streaming/Streamer.h>
#include <Utils/UtilityFunctions.h>
#include "Utils/ConfigMap.hpp"
#include "Utils/ThreadPerf.hpp"
#include <Streaming/SingleThreadStreamer.h>
#include <Streaming/BlockPartitionStreamer.h>

using namespace INTELLI;

torch::Tensor AMMBench::Streamer::run(INTELLI::ConfigMapPtr cfg, torch::Tensor A, torch::Tensor B, uint64_t sketchSize = 1) {
    metrics = newConfigMap();
    matC = newTensor(torch::zeros({A.size(0), B.size(1)}));
    torch::Tensor realC = torch::matmul(A, B);
    uint64_t isStreaming = cfg->tryU64("isStreaming", 0, true);
    uint64_t threads = cfg->tryU64("threads", 1, true);
    sketchSize = cfg->tryU64("sketchDimension", sketchSize, true);
    uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
    UtilityFunctions::bind2Core((int) coreBind);

    if (isStreaming) {
        uint64_t streamingTwoMatrices = cfg->tryU64("streamingTwoMatrixes", 0, true);
        double throughput;
        if (threads) {
            AMMBench::BlockPartitionStreamer ss;
            ss.setConfig(cfg);
            if (streamingTwoMatrices) {
                *matC = ss.streamingAmm2S(A, B, sketchSize);
            } else *matC = ss.streamingAmm(A, B, sketchSize);
            throughput = ss.getThroughput();
            metrics->edit("throughput", throughput);
            metrics->edit("throughputByElements", (double) (throughput * A.size(1)));
            metrics->edit("95%latency", (double) ss.getLatencyPercentage(0.95));
        } else {
            AMMBench::SingleThreadStreamer ss;
            ss.setConfig(cfg);
            if (streamingTwoMatrices) {
                *matC = ss.streamingAmm2S(A, B, sketchSize);
            } else {
                *matC = ss.streamingAmm(A, B, sketchSize);
            }
            throughput = ss.getThroughput();
            metrics->edit("throughput", throughput);
            metrics->edit("throughputByElements", (double) (throughput * A.size(1)));
            metrics->edit("95%latency", (double) ss.getLatencyPercentage(0.95));
        }
        metrics->edit("elapsedTime", (uint64_t)((A.size(0) * 1e6) / throughput));
    }
    else{
        // non-streaming
        std::string meterTag = cfg->tryString("meterTag", "intelMsr", true);
        cfg->edit("useCPP", (uint64_t)1);
        uint64_t forceMP = cfg->tryU64("forceMP", 1, true);

        ThreadPerf pef(-1);
        pef.setPerfList();
        AMMBench::BlockPartitionRunner br;
        if (threads > 1 || forceMP) {
            br.setConfig(cfg);
            br.createABC(A, B);
            pef.start();
            *matC = br.parallelForward();
            pef.end();
        } else {
            AMMBench::CPPAlgoTable cppAlgoTable;
            std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
            AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
            cppAlgoPtr->setConfig(cfg);
            pef.start();
            *matC = cppAlgoPtr->amm(A, B, sketchSize);
            pef.end();
        }
        uint64_t elapsedTime = br.getElapsedTime();
        double throughput = (A.size(0) * 1e6) / elapsedTime;
        metrics->edit("throughput", throughput);
        metrics->edit("throughputByElements", (double) (throughput * A.size(1)));
        metrics->edit("elapsedTime", elapsedTime);
    }

    // calculate error
    double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, *matC);
    double froBNormal = B.norm().item<double>();
    double errorBoundRatio = froError / froBNormal;
    metrics->edit("froError", (double) froError);
    metrics->edit("errorBoundRatio", (double) errorBoundRatio);
    return *matC;
}

