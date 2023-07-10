//
// Created by yuhao on 6/30/23.
//

#ifndef INTELLISTREAM_MNISTMATRIXLOADER_H
#define INTELLISTREAM_MNISTMATRIXLOADER_H

#include <MatrixLoader/AbstractMatrixLoader.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_MatrixLOADER
 * @{
 */
/**
 * @ingroup AMMBENCH_MatrixLOADER_MNIST The MNIST training image dataset generator
 * @{
 */
/**
 * @class MNISTMatrixLoader MatrixLoader/MNISTMatrixLoader.h
 * @brief The MNIST class of matrix loader https://www.kaggle.com/datasets/hojjatk/mnist-dataset
 * @ingroup AMMBENCH_MatrixLOADER_MNIST
 * @note:
 * - Must have a global config by @ref setConfig
 * @note  Default behavior
* - create
* - call @ref setConfig, this function will also generate the tensor A and B correspondingly
* - call @ref getA and @ref getB (assuming we are benchmarking torch.mm(A,B))
 * @note: does not need config
 * @note: default name tags
 * "MNIST": @ref MNISTMatrixLoader
 */
    class MNISTMatrixLoader : public AbstractMatrixLoader {
    protected:
        torch::Tensor A, B;

        /**
         * @brief Inline logic of reading a config file
         * @param cfg the config
         */
        void paraseConfig(INTELLI::ConfigMapPtr cfg);

        /**
         * @brief inline logic of generating A and B
         */
        void generateAB();

    public:
        MNISTMatrixLoader() = default;

        ~MNISTMatrixLoader() = default;

        /**
           * @brief Set the GLOBAL config map related to this loader
           * @param cfg The config map
            * @return bool whether the config is successfully set
            * @note
           */
        virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

        /**
         * @brief get the A matrix
         * @return the generated A matrix
         */
        virtual torch::Tensor getA();

        /**
        * @brief get the B matrix
        * @return the generated B matrix
        */
        virtual torch::Tensor getB();
    };

/**
 * @ingroup AMMBENCH_MatrixLOADER_MNIST
 * @typedef MNISTMatrixLoaderPtr
 * @brief The class to describe a shared pointer to @ref MNISTMatrixLoader

 */
    typedef std::shared_ptr<class AMMBench::MNISTMatrixLoader> MNISTMatrixLoaderPtr;
/**
 * @ingroup AMMBENCH_MatrixLOADER_MNIST
 * @def newMNISTMatrixLoader
 * @brief (Macro) To creat a new @ref MNISTMatrixLoader under shared pointer.
 */
#define newMNISTMatrixLoader std::make_shared<AMMBench::MNISTMatrixLoader>
/**
 * @}
 */
/**
 * @}
 */
}
#endif //INTELLISTREAM_MNISTMATRIXLOADER_H
