add_sources(
        AbstractCPPAlgo.cpp
        CRSCPPAlgo.cpp
        WeightedCRCPPAlgo.cpp
        CPPAlgoTable.cpp
        CRSV2CPPAlgo.cpp
        BCRSCPPAlgo.cpp
        # TODO: All other algorithms require advanced PyTorch features
        # that need extensive implementation work with Eigen:
        # - ProductQuantization*.cpp: requires tensor indexing, select, unsqueeze, stack, argmin
        # - VectorQuantization.cpp: same as above
        # - EWSCPPAlgo.cpp: masked operations
        # - CoOccurringFDCPPAlgo.cpp: SVD, QR decomposition
        # - BetaCoOFDCPPAlgo.cpp: similar to CoOccurringFDCPPAlgo
        # - CountSketchCPPAlgo.cpp: randperm, multinomial
        # - INT8CPPAlgo.cpp: quantization
        # - TugOfWarCPPAlgo.cpp: bernoulli
        # - SMPPCACPPAlgo.cpp: advanced operations
        # - FastJLTCPPAlgo.cpp: randn
        # - RIPCPPAlgo.cpp: uniform_, sign_, view
        # - BlockLRACPPAlgo.cpp: index, SVD, narrow
)
if (DEFINED ENV{LibAMM_CL})
    add_sources(
            CLMMCPPAlgo.cpp
    )
endif ()

