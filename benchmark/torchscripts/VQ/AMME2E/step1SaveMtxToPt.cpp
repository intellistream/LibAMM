
//
// Created by haolan on 25/6/23.
//
#include <vector>
#include <AMMBench.h>
#include <iostream>
#include <CPPAlgos/VectorQuantization.h>

using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace DIVERSE_METER;
using namespace AMMBench;

int main(){

    vector<string> dataSetNames={"AST","BUS","DWAVE","ECO","QCD","RDB","UTM","ZENIOS"};
    vector<string> srcAVec={"datasets/AST/mcfe.mtx","datasets/BUS/gemat1.mtx","datasets/DWAVE/dwa512.mtx","datasets/ECO/wm2.mtx","datasets/QCD/qcda_small.mtx","datasets/RDB/rdb2048.mtx","datasets/UTM/utm1700a.mtx","datasets/ZENIOS/zenios.mtx"};
    vector<string> srcBVec={"datasets/AST/mcfe.mtx","datasets/BUS/gemat1.mtx","datasets/DWAVE/dwb512.mtx","datasets/ECO/wm3.mtx","datasets/QCD/qcdb_small.mtx","datasets/RDB/rdb2048l.mtx","datasets/UTM/utm1700b.mtx","datasets/ZENIOS/zenios.mtx"};

    for (int i=0; i<8; i++){
        INTELLI_INFO(dataSetNames[i]);
        INTELLI_INFO(srcAVec[i]);
        INTELLI_INFO(srcBVec[i]);

        ConfigMapPtr cfg = newConfigMap();
        cfg->edit("srcA", "../../../"+srcAVec[i]);
        cfg->edit("srcB", "../../../"+srcBVec[i]);
        cfg->edit("transposeB", uint64_t(1)); 
        cfg->edit("normalizeA", uint64_t(1));
        cfg->edit("normalizeB", uint64_t(1));

        AMMBench::MatrixLoaderTable mLoaderTable;
        auto matLoaderPtr = mLoaderTable.findMatrixLoader("mtx");
        assert(matLoaderPtr);
        matLoaderPtr->setConfig(cfg);
        auto A = matLoaderPtr->getA();
        auto B = matLoaderPtr->getB();
        
        auto pickledA = torch::pickle_save(A);
        std::ofstream foutA("MtxPt/"+dataSetNames[i]+"_A.pt", std::ios::out | std::ios::binary);
        foutA.write(pickledA.data(), pickledA.size());
        foutA.close();

        auto pickledB = torch::pickle_save(B);
        std::ofstream foutB("MtxPt/"+dataSetNames[i]+"_B.pt", std::ios::out | std::ios::binary);
        foutB.write(pickledB.data(), pickledB.size());
        foutB.close();
    }

    return 0;
}


