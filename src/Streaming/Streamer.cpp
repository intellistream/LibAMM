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

torch::Tensor AMMBench::Streamer::run(INTELLI::ConfigMapPtr cfg,
                                      torch::Tensor A,
                                      torch::Tensor B,
                                      uint64_t sketchSize,
                                      string metricsPrefix) {
  metrics = newConfigMap();
  matC = newTensor(torch::zeros({A.size(0), B.size(1)}));
  uint64_t isStreaming = cfg->tryU64("isStreaming", 0, true);
  uint64_t threads = cfg->tryU64("threads", 1, true);
  sketchSize = cfg->tryU64("sketchDimension", sketchSize, true);

  if (isStreaming) {
    uint64_t streamingTwoMatrices = cfg->tryU64("streamingTwoMatrixes", 0, true);
    double throughput;
    if (threads > 1) {
      INTELLI_INFO("streaming, multithread " + to_string(threads));
      AMMBench::BlockPartitionStreamer ss;
      ss.setConfig(cfg);
      if (streamingTwoMatrices) {
        *matC = ss.streamingAmm2S(A, B, sketchSize);
      } else *matC = ss.streamingAmm(A, B, sketchSize);
      throughput = ss.getThroughput();
      metrics->edit(metricsPrefix + "Throughput", throughput);
      metrics->edit(metricsPrefix + "ThroughputByElements", (double) (throughput * A.size(1)));
      metrics->edit(metricsPrefix + "95%latency", (double) ss.getLatencyPercentage(0.95));
    } else {
      INTELLI_INFO("streaming, singlethread");
      AMMBench::SingleThreadStreamer ss;
      ss.setConfig(cfg);
      if (streamingTwoMatrices) {
        *matC = ss.streamingAmm2S(A, B, sketchSize);
      } else {
        *matC = ss.streamingAmm(A, B, sketchSize);
      }
      throughput = ss.getThroughput();
      metrics->edit(metricsPrefix + "Throughput", throughput);
      metrics->edit(metricsPrefix + "ThroughputByElements", (double) (throughput * A.size(1)));
      metrics->edit(metricsPrefix + "95%latency", (double) ss.getLatencyPercentage(0.95));
    }
    metrics->edit(metricsPrefix + "ElapsedTime", (uint64_t) ((A.size(0) * 1e6) / throughput));
  } else {
    // non-streaming
    cfg->edit(metricsPrefix + "useCPP", (uint64_t) 1);
    uint64_t forceMP = cfg->tryU64("forceMP", 1, true);
    uint64_t elapsedTime = 0;

    if (threads > 1 || forceMP) {
      INTELLI_INFO("non-streaming, multithread " + to_string(threads));
      AMMBench::BlockPartitionRunner br;
      br.setConfig(cfg);
      br.createABC(A, B);
      *matC = br.parallelForward();
      elapsedTime = br.getElapsedTime();
    } else {
      INTELLI_INFO("non-streaming, singlethread");
      uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
      UtilityFunctions::bind2Core((int) coreBind);
      AMMBench::CPPAlgoTable cppAlgoTable;
      std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
      AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
      cppAlgoPtr->setConfig(cfg);
      ThreadPerf pef(-1);
      pef.setPerfList();
      // pef result example
      // cacheMiss       0       U64
      // cacheRefs       0       U64
      // cpuClock        0       U64
      // cpuCycle        0       U64
      // instructions    0       U64
      // perfElapsedTime 83432   U64
      // taskClock       0       U64
      pef.start();
      *matC = cppAlgoPtr->amm(A, B, sketchSize);
      pef.end();
      auto resultCsv = pef.resultToConfigMap();
      elapsedTime = resultCsv->getU64("perfElapsedTime");
    }
    double throughput = (A.size(0) * 1e6) / elapsedTime;
    metrics->edit(metricsPrefix + "Throughput", throughput);
    metrics->edit(metricsPrefix + "ThroughputByElements", (double) (throughput * A.size(1)));
    metrics->edit(metricsPrefix + "ElapsedTime", elapsedTime);
  }

  // calculate error
  torch::Tensor realC = torch::matmul(A, B);
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, *matC);
  metrics->edit(metricsPrefix + "FroError", (double) froError);
  return *matC;
}

