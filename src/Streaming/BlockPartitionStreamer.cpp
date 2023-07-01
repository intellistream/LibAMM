//
// Created by haolan on 27/6/23.
//

#include <Streaming/BlockPartitionStreamer.h>
#include <Utils/UtilityFunctions.h>
#include <Parallelization/BlockPartitionRunner.h>
#include <Utils/BS_thread_pool.hpp>

bool AMMBench::BlockPartitionStreamer::setConfig(INTELLI::ConfigMapPtr cfg) {
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
    threads = cfg->tryU64("threads", 1, true);
    return true;
}

torch::Tensor AMMBench::BlockPartitionStreamer::streamingAmm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
    assert(sketchSize);
    uint64_t aRows = A.size(0);
    cfgGlobal->edit("streamingTupleCnt", (uint64_t) aRows);
    if (batchSize > aRows) {
        batchSize = aRows;
    }
    std::string cppAlgoTag = cfgGlobal->tryString("cppAlgoTag", "mm", true);
    AMMBench::TimeStamper tsGen;
    tsGen.setConfig(cfgGlobal);
    myTs = tsGen.getTimeStamps();
    INTELLI_INFO("Generate time stamp done");
    matC = newTensor(torch::zeros({A.size(0), B.size(1)}));
    struct timeval tstart;
    //INTELLI_INFO("I am mm");
    INTELLI_INFO("Start Streaming A rows");
    uint64_t startRow = 0;
    uint64_t endRow = startRow + batchSize;
    uint64_t tNow = 0;
    uint64_t tEXpectedArrival = myTs[endRow - 1]->arrivalTime;
    uint64_t tp = 0;
    uint64_t tDone = 0;
    gettimeofday(&tstart, NULL);
    size_t slice_size = batchSize / threads;

    //std::vector<std::thread> threadVector(threads);
    auto pool = std::make_shared<BS::thread_pool>(threads);
    BS::multi_future<void> tasks(threads);
    while (startRow < aRows) {
        tNow = INTELLI::UtilityFunctions::timeLastUs(tstart);
        while (tNow < tEXpectedArrival) {
            tNow = INTELLI::UtilityFunctions::timeLastUs(tstart);
            //usleep(1);
        }
        // stream by rows
        for (size_t i = 0; i < threads; ++i) {
            tasks[i] = pool->submit([&, i]() { // Capture by reference, but capture 'i' by value
                cout << "created" << endl;
                    size_t startRowThread = startRow + i * slice_size;
                    size_t endRowThread = (i == threads - 1) ? endRow : startRowThread + slice_size;
                auto subA = A.slice(0, startRowThread, endRowThread);
                matC->slice(0, startRowThread, endRowThread) = cppAlgoPtr->amm(subA, B, sketchSize);
            });
        }
        tasks.wait();
        // stream by columns
        /*
        for (size_t i = 0; i < threads; ++i) {
            size_t startColThread = i * slice_size;
            size_t endColThread = (i == threads - 1) ? A.size(1) : startColThread + slice_size;

            threadVector[i] = std::thread([&](size_t start, size_t end) {
                auto subA = A.t().slice(start, 0, end).t();
                auto subB = B.slice(start, 0, end);
                matC->slice(start, 0, end) = cppAlgoPtr->amm(subA, subB, sketchSize).t();
            }, startColThread, endColThread);
        }*/
        /*for(auto& thread : threadVector) {
            thread.join();
        }*/
        tp = INTELLI::UtilityFunctions::timeLastUs(tstart);
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
    tDone = INTELLI::UtilityFunctions::timeLastUs(tstart);
    INTELLI_INFO("Done in " + to_string(tDone) + "us");
    throughput = aRows;
    throughput = throughput * 1e6 / tDone;
    return *matC;
}


torch::Tensor AMMBench::BlockPartitionStreamer::streamingAmm2S(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
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
    struct timeval tstart;
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
    gettimeofday(&tstart, NULL);
    uint64_t iterationCnt=0;
    torch::Tensor incomingA,incomingB,newArrivedB,oldArrivedA;
    uint64_t aBCols=0,lastABCols=0;
    while (startRow < aRows) {
        tNow = INTELLI::UtilityFunctions::timeLastUs(tstart);
        //auto subA = A.slice(0, startRow, endRow);
        incomingA =A.slice(0, startRow, endRow);
        incomingB=B.slice(1,startRow,endRow);
        newArrivedB=B.slice(1,0,endRow);
        while (tNow < tEXpectedArrival) {
            tNow = INTELLI::UtilityFunctions::timeLastUs(tstart);
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
    tDone = INTELLI::UtilityFunctions::timeLastUs(tstart);
    /**
     * @brief The latency calculation is different from one stream case here,
     * as older A will still be probed by newer B
     */
    for (size_t i = 0; i < aRows; i++) {
        myTs[i]->processedTime = tDone;
    }
    INTELLI_INFO("Done in " + to_string(tDone) + "us");
    throughput = aRows;
    throughput = throughput * 1e6 / tDone;
    return *matC;
}
double AMMBench::BlockPartitionStreamer::getLatencyPercentage(double fraction) {
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