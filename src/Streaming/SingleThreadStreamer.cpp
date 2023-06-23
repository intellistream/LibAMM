//
// Created by tony on 23/06/23.
//

#include <Streaming/SingleThreadStreamer.h>
#include <Utils/UtilityFunctions.h>
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
     batchSize=cfg->tryU64("batchSize",1, true);
    return true;
}
torch::Tensor AMMBench::SingleThreadStreamer::streamingAmm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize){
    assert(sketchSize);
    uint64_t aRows=A.size(0);
    cfgGlobal->edit("streamingTupleCnt",(uint64_t)aRows);
    if(batchSize>aRows)
    {
        batchSize=aRows;
    }
    AMMBench::TimeStamper tsGen;
    tsGen.setConfig(cfgGlobal);
    myTs=tsGen.getTimeStamps();
    INTELLI_INFO("Generate time stamp done");
    matC = newTensor(torch::zeros({A.size(0), B.size(1)}));
    struct timeval tstart;
    //INTELLI_INFO("I am mm");
    INTELLI_INFO("Start Streaming");
    uint64_t startRow=0;
    uint64_t endRow=startRow+batchSize;
    uint64_t tNow=0;
    uint64_t tEXpectedArrival=myTs[endRow-1]->arrivalTime;
    uint64_t tp=0;
    uint64_t tDone=0;
    gettimeofday(&tstart,NULL);
    while(startRow<aRows)
    {
        tNow=INTELLI::UtilityFunctions::timeLastUs(tstart);
        auto  subA=A.slice(0,startRow,endRow);
        while(tNow<tEXpectedArrival)
        {
            tNow=INTELLI::UtilityFunctions::timeLastUs(tstart);
            usleep(1);
        }
        /**
         * @brief now, the whole batch has arrived, compute
         */
        matC->slice(0, startRow, endRow) = cppAlgoPtr->amm(subA, B, sketchSize);
        tp=INTELLI::UtilityFunctions::timeLastUs(tstart);
        for(size_t i=startRow;i<endRow;i++)
        {
            myTs[i]->processedTime=tp;
        }
        /**
         * @brief update the indexes
         */
        startRow+=batchSize;
        endRow+=batchSize;
        if(endRow>=aRows)
        {
            endRow=aRows;
        }
        tEXpectedArrival=myTs[endRow-1]->arrivalTime;
    }
    tDone=INTELLI::UtilityFunctions::timeLastUs(tstart);
    INTELLI_INFO("Done in "+ to_string(tDone)+"us");
    throughput=aRows;
    throughput=throughput*1e6/tDone;
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