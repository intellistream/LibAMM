//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/CPPAlgoTable.h>
#include <CPPAlgos/CRSCPPAlgo.h>
#include <CPPAlgos/CRSV2CPPAlgo.h>

#include <CPPAlgos/CountSketchCPPAlgo.h>

#include <CPPAlgos/BCRSCPPAlgo.h>
#include <CPPAlgos/EWSCPPAlgo.h>
#include <CPPAlgos/CoOccurringFDCPPAlgo.h>
#include <CPPAlgos/BetaCoOFDCPPAlgo.h>
#include <CPPAlgos/WeightedCRCPPAlgo.h>
#include <CPPAlgos/SMPPCACPPAlgo.h>
#include <CPPAlgos/INT8CPPAlgo.h>
#include <CPPAlgos/TugOfWarCPPAlgo.h>
namespace AMMBench {
AMMBench::CPPAlgoTable::CPPAlgoTable() {
  algoMap["mm"] = newAbstractCPPAlgo();
  algoMap["crs"] = newCRSCPPAlgo();
  algoMap["crsV2"] = newCRSV2CPPAlgo();
  algoMap["countSketch"] = newCountSketchCPPAlgo();
  algoMap["bcrs"] = newBCRSCPPAlgo();
  algoMap["ews"] = newEWSCPPAlgo();
  algoMap["cooFD"] = newCoOccurringFDCPPAlgo();
  algoMap["bcooFD"] = newBetaCoOFDCPPAlgo();
  algoMap["int8"] = newINT8CPPAlgo();
  algoMap["tugOfWar"] = newTugOfWarCPPAlgo();
  algoMap["weighted-cr"] = newWeightedCRCPPAlgo();
  algoMap["smp-pca"] = newSMPPCACPPAlgo();
}

} // AMMBench
