/*! \file UtilityFunctions.hpp*/
// Copyright (C) 2021 by the INTELLI team (https://github.com/intellistream)

#ifndef IntelliStream_SRC_UTILS_UTILITYFUNCTIONS_HPP_
#define IntelliStream_SRC_UTILS_UTILITYFUNCTIONS_HPP_

#include <string>
#include <experimental/filesystem>
#include <barrier>
#include <functional>
#include <torch/torch.h>
#include <ATen/ATen.h>
//#include <Common/Types.h>

#include <vector>
/* Period parameters */

#define TRUE 1
#define FALSE 0

#include <sys/time.h>

namespace INTELLI {
    typedef std::shared_ptr<std::barrier<>> BarrierPtr;
#define TIME_LAST_UNIT_MS 1000
#define TIME_LAST_UNIT_US 1000000
#define chronoElapsedTime(start) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()
/**
 * @defgroup
 */
    class UtilityFunctions {

    public:
        UtilityFunctions();

        //static std::shared_ptr<std::barrier<>> createBarrier(int count);

        // static void timerStart(Result &result);

        //static void timerEnd(Result &result);

        static size_t timeLast(struct timeval past, struct timeval now);

        static size_t timeLastUs(struct timeval past);

        //bind to CPU
        /*!
         bind to CPU
         \li bind the thread to core according to id
         \param id the core you plan to bind, -1 means let os decide
         \return cpuId, the real core that bind to
         \todo unsure about hyper-thread
          */
        static int bind2Core(int id);
        //partition

        static std::vector<size_t> avgPartitionSizeFinal(size_t inS, std::vector<size_t> partitionWeight);

        static std::vector<size_t> weightedPartitionSizeFinal(size_t inS, std::vector<size_t> partitionWeight);

        static size_t to_periodical(size_t val, size_t period) {
            if (val < period) {
                return val;
            }
            size_t ru = val % period;
            /* if(ru==0)
             {
               return  period;
             }*/
            return ru;
        }

        static double relativeFrobeniusNorm(torch::Tensor A, torch::Tensor B) {
            torch::Tensor error = A - B;
            double frobeniusNormA = A.norm().item<double>();
            double frobeniusNormError = error.norm().item<double>();

            return frobeniusNormError / frobeniusNormA;
        }

        static double relativeSpectralNormError(torch::Tensor A, torch::Tensor B) {

            torch::Tensor UA;
            torch::Tensor SA;
            torch::Tensor VhA;
            std::tie(UA, SA, VhA) = torch::linalg::svd(A, false, c10::nullopt);
            torch::Tensor UError;
            torch::Tensor SError;
            torch::Tensor VhError;
            std::tie(UError, SError, VhError) = torch::linalg::svd(A-B, false, c10::nullopt);

            double SpectralNormA = SA[0].item<double>();
            // std::cout << "SA: " << SA << " ";
            double SpectralNormError = SError[0].item<double>();
            // std::cout << "SError: " << SError << " ";

            // c10::IntArrayRef shape = eigenvaluesA.sizes();
            // std::vector<int64_t> shapeVec(shape.vec());
            // std::cout << "eigenvaluesA shape: ";
            // for (int64_t dim : shapeVec) {
            //     std::cout << dim << " ";
            // }
            // std::cout << std::endl;

            double relativeSpectralNormError = abs(SpectralNormError/SpectralNormA);

            // std::cout << "SpectralNormA: " << SpectralNormA << " ";
            // std::cout << "SpectralNormError: " << SpectralNormError << " ";
            // std::cout << "relativeSpectralNormError: " << relativeSpectralNormError << " ";

            return relativeSpectralNormError;
        }

        static double errorBoundRatio(torch::Tensor A, torch::Tensor B) {
            torch::Tensor error = A - B;
            double frobeniusNormA = A.norm().item<double>();
            double frobeniusNormB = B.norm().item<double>();
            double frobeniusNormError = error.norm().item<double>();

            return frobeniusNormError / frobeniusNormA / frobeniusNormB;
        }

    };
}
#endif //IntelliStream_SRC_UTILS_UTILITYFUNCTIONS_HPP_
