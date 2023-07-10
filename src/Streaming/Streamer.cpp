//
// Created by haolan on 8/7/23.
//

#include <Streaming/Streamer.h>
#include <Utils/UtilityFunctions.h>
#include "Utils/ConfigMap.hpp"
#include <Streaming/SingleThreadStreamer.h>
#include <Streaming/BlockPartitionStreamer.h>

torch::Tensor AMMBench::Streamer::run(INTELLI::ConfigMapPtr cfg, torch::Tensor A, torch::Tensor B, uint64_t sketchSize = 1) {
    metrics = newConfigMap();
    matC = newTensor(torch::zeros({A.size(0), B.size(1)}));
    uint64_t threads = cfg->tryU64("threads", 1, true);
    uint64_t streamingTwoMatrices = cfg->tryU64("streamingTwoMatrixes", 0, true);
    double throughput;
    if (threads){
        AMMBench::BlockPartitionStreamer ss;
        ss.setConfig(cfg);
        if (streamingTwoMatrices){
            *matC = ss.streamingAmm2S(A, B, sketchSize);
        }
        else *matC = ss.streamingAmm(A, B, sketchSize);
        throughput = ss.getThroughput();
        metrics->edit("throughput", throughput);
        metrics->edit("throughputByElements", (double) (throughput * A.size(1)));
        metrics->edit("95%latency", (double) ss.getLatencyPercentage(0.95));
    }
    else{
        AMMBench::SingleThreadStreamer ss;
        ss.setConfig(cfg);
        if(streamingTwoMatrices){
            *matC = ss.streamingAmm2S(A, B, sketchSize);
        }
        else{
            *matC = ss.streamingAmm(A, B, sketchSize);
        }
        throughput = ss.getThroughput();
        metrics->edit("throughput", throughput);
        metrics->edit("throughputByElements", (double) (throughput * A.size(1)));
        metrics->edit("95%latency", (double) ss.getLatencyPercentage(0.95));
    }
    torch::Tensor realC = torch::matmul(A, B);
    double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, *matC);
    double froBNormal = B.norm().item<double>();
    double errorBoundRatio = froError / froBNormal;
    metrics->edit("froError", (double) froError);
    metrics->edit("errorBoundRatio", (double) errorBoundRatio);
    metrics->edit("elapsedTime", (A.size(0) * 1e6) / throughput);
    return *matC;
}

