//
// Created by tony on 23/06/23.
//

#include <Streaming/SingleThreadStreamer.h>
#include <Utils/UtilityFunctions.h>
#include "Utils/ThreadPerf.hpp"
#include <time.h>
#include <chrono>

using namespace INTELLI;

bool AMMBench::SingleThreadStreamer::setConfig(INTELLI::ConfigMapPtr cfg) {
    cfgGlobal = cfg;
    /**
    * @brief 1.set the algo
    */
    std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
    cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
    cppAlgoPtr->setConfig(cfg);
    /**
     * @brief 2. set the batch size
     */
    batchSize = cfg->tryU64("batchSize", 1, true);
    coreBind = cfg->tryU64("coreBind", 0, true);
    INTELLI::UtilityFunctions::bind2Core(coreBind);
    return true;
}

torch::Tensor AMMBench::SingleThreadStreamer::streamingAmm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
    assert(sketchSize);
    uint64_t aRows = A.size(0);
    cfgGlobal->edit("streamingTupleCnt", (uint64_t) aRows);
    if (batchSize > aRows) {
        batchSize = aRows;
    }
    AMMBench::TimeStamper tsGen;
    tsGen.setConfig(cfgGlobal);
    myTs = tsGen.getTimeStamps(); // a vector of pointers to timestamp objects
    INTELLI_INFO("Generate time stamp done");
    matC = newTensor(torch::zeros({A.size(0), B.size(1)}));
    
    //INTELLI_INFO("I am mm");
    INTELLI_INFO("Start Streaming A rows");
    uint64_t startRow = 0;
    uint64_t endRow = startRow + batchSize;
    uint64_t tNow = 0; // used to store the elapsed time from tstart
    uint64_t tEXpectedArrival = myTs[endRow - 1]->arrivalTime;
    uint64_t tp = 0; // the processing time of the current batch
    uint64_t tDone = 0; // the total time taken to complete the computation

    //pre-partition
    std::vector<torch::Tensor> partitions; // store partitions of matrix A to process in batches
    for (uint64_t i = 0; i < aRows; i += batchSize) {
        uint64_t end = std::min(i + batchSize , aRows);
        auto subA = A.slice(0, i, end);
        partitions.push_back(subA);
    }
    uint64_t index = -1;

    ThreadPerf pef(-1);
    pef.setPerfList();
    pef.start();
    auto tstart = std::chrono::high_resolution_clock::now();
    // struct timeval tstart;
    // gettimeofday(&tstart, NULL);

    while (startRow < aRows) {
        tNow = chronoElapsedTime(tstart);
        index++;
        while (tNow < tEXpectedArrival) {
            tNow = chronoElapsedTime(tstart);
        }
        /**
         * @brief now, the whole batch has arrived, compute
         */
        //auto subA = A.slice(0, startRow, endRow);
        auto subA = partitions[index];
        matC->slice(0, startRow, endRow) = cppAlgoPtr->amm(subA, B, sketchSize);
        tp = chronoElapsedTime(tstart);
        /**
         * @brief the new arrived A will be no longer probed, so we can assign the processed time now
         */
        for (size_t i = startRow; i < endRow; i++) {
            myTs[i]->processedTime = tp;
        }
        /**
         * @brief update the indexes
         */
        startRow += batchSize;
        endRow += batchSize;
        if (endRow >= aRows) {
            endRow = aRows;
        }
        tEXpectedArrival = myTs[endRow - 1]->arrivalTime;
    }
    tDone = chronoElapsedTime(tstart);
    pef.end();

    metrics = pef.resultToConfigMap();

    INTELLI_INFO("Done in " + to_string(tDone) + "us");
    throughput = aRows * 1e6 / tDone;
    double throughputByElements = throughput * A.size(1);
    double latency95 = getLatencyPercentage(0.95);
    metrics->edit("throughput", throughput);
    metrics->edit("throughputByElements", throughputByElements);
    metrics->edit("95%latency", latency95);

    return *matC;
}

torch::Tensor AMMBench::SingleThreadStreamer::streamingAmm2S(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
  assert(sketchSize);
  uint64_t aRows = A.size(0);
  cfgGlobal->edit("streamingTupleCnt", (uint64_t) aRows);
  if (batchSize > aRows) {
    batchSize = aRows;
  }
  AMMBench::TimeStamper tsGen,tsGenB;
  tsGen.setConfig(cfgGlobal);
  myTs = tsGen.getTimeStamps();

  tsGenB.setSeed(7758258);
  tsGenB.setConfig(cfgGlobal);
  myTsB = tsGenB.getTimeStamps();
  INTELLI_INFO("Generate time stamps for two streams done");
  matC = newTensor(torch::zeros({A.size(0), B.size(1)}));
  
  //INTELLI_INFO("I am mm");
  INTELLI_INFO("Start Streaming A rows and B cols");
  uint64_t startRow = 0;
  uint64_t endRow = startRow + batchSize;
  uint64_t tNow = 0;
  uint64_t tEXpectedArrival = myTs[endRow - 1]->arrivalTime;
  if(myTsB[endRow-1]->arrivalTime>tEXpectedArrival)
  {
    tEXpectedArrival=myTsB[endRow-1]->arrivalTime;
  }
  uint64_t tDone = 0;

  ThreadPerf pef(-1);
  pef.setPerfList();
  pef.start();

  auto tstart = std::chrono::high_resolution_clock::now();
  // struct timeval tstart;
  // gettimeofday(&tstart, NULL);
  uint64_t iterationCnt=0;
  torch::Tensor incomingA,incomingB,newArrivedB,oldArrivedA;
  uint64_t aBCols=0,lastABCols=0;
  while (startRow < aRows) {
    tNow = chronoElapsedTime(tstart);;
    //auto subA = A.slice(0, startRow, endRow);
    incomingA =A.slice(0, startRow, endRow);
    incomingB=B.slice(1,startRow,endRow);
    newArrivedB=B.slice(1,0,endRow);
    while (tNow < tEXpectedArrival) {
      tNow = chronoElapsedTime(tstart);;
      //usleep(1);
    }
    INTELLI_INFO("batch of "+ to_string(startRow)+" to "+ to_string(endRow)+" are ready");
    /**
     * @brief now, the whole batch has arrived, compute
     */
     /**
      * @brief do the incomingA*newArrivedB part
      */
      auto aB=cppAlgoPtr->amm(incomingA, newArrivedB, sketchSize);
      lastABCols=aBCols;
      aBCols=aB.size(1);
      matC->slice(0,startRow,endRow).slice(1,0,aBCols).copy_(aB);
    /**
    * @brief do the oldArrivedA*incomingB part
    */
    if(iterationCnt!=0)
    {
      auto aB2=cppAlgoPtr->amm(oldArrivedA, incomingB, sketchSize);
      uint64_t aB2Rows=aB2.size(0);
      uint64_t aB2Cols=aB2.size(1);
      matC->slice(0,0,aB2Rows).slice(1,lastABCols,lastABCols+aB2Cols).copy_(aB2);
    }
    oldArrivedA=A.slice(0, 0, endRow);
    /**
     * @brief update the indexes
     */
    startRow += batchSize;
    endRow += batchSize;
    if (endRow >= aRows) {
      endRow = aRows;
    }
    tEXpectedArrival = myTs[endRow - 1]->arrivalTime;
    if(myTsB[endRow-1]->arrivalTime>tEXpectedArrival)
    {
      tEXpectedArrival=myTsB[endRow-1]->arrivalTime;
    }
    iterationCnt++;
  }
  tDone = chronoElapsedTime(tstart);
  pef.end();
  /**
   * @brief The latency calculation is different from one stream case here,
   * as older A will still be probed by newer B
   */
    for (size_t i = 0; i < aRows; i++) {
      myTs[i]->processedTime = tDone;
    }

  metrics = pef.resultToConfigMap();

  INTELLI_INFO("Done in " + to_string(tDone) + "us");
  throughput = aRows * 1e6 / tDone;
  double throughputByElements = throughput * A.size(1);
  double latency95 = getLatencyPercentage(0.95);
  metrics->edit("throughput", throughput);
  metrics->edit("throughputByElements", throughputByElements);
  metrics->edit("95%latency", latency95);
  return *matC;
}

double AMMBench::SingleThreadStreamer::getLatencyPercentage(double fraction) {
    size_t rLen = myTs.size();
    size_t nonZeroCnt = 0;
    std::vector<uint64_t> validLatency;
    for (size_t i = 0; i < rLen; i++) {
        if (myTs[i]->processedTime >= myTs[i]->arrivalTime && myTs[i]->processedTime != 0) {
            validLatency.push_back(myTs[i]->processedTime - myTs[i]->arrivalTime);
            nonZeroCnt++;
        }
    }
    if (nonZeroCnt == 0) {
        INTELLI_ERROR("No valid latency, maybe there is no AMM result?");
        return 0;
    }
    std::sort(validLatency.begin(), validLatency.end());
    double t = nonZeroCnt;
    t = t * fraction;
    size_t idx = (size_t) t + 1;
    if (idx >= validLatency.size()) {
        idx = validLatency.size() - 1;
    }
    return validLatency[idx];
}