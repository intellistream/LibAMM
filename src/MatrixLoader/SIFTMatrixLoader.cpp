//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/SIFTMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <AMMBench.h>

void AMMBench::SIFTMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {

    aRow = cfg->tryU64("aRow", 0, true);
    aCol = cfg->tryU64("aCol", 0, true);
    bCol = cfg->tryU64("bCol", 0, true);

    if (aRow!=0 && aCol!=0 && bCol!=0){
        if (aRow!=bCol){
            std::runtime_error("in PCA task, B=A.t, so bCol==aRow should be hold all the time. Otherwise we can not get a symmetric real matrix for the rest of PCA task");
        }
        INTELLI_INFO("aRow, aCol, bCol are specified in config.csv, so will make the default SIFT dataset [128x10000]*[10000x128] into the specified size [" + to_string(aRow) + "x" + to_string(aCol) + "]*[" + to_string(aCol) + "x" + to_string(bCol) + "]");
    }
    else{ // use default SIFT dataset 128*10000, 10000*128
        INTELLI_INFO("At least one of [aRow, aCol, bCol] is not specified in config.csv, so will use default SIFT dataset [128x10000]*[10000x128]");
    }
}

torch::Tensor repeatAndCropMatrix(torch::Tensor matrix, int height, int width) {
    // Get the dimensions of the original matrix
    int originalHeight = matrix.size(0);
    int originalWidth = matrix.size(1);

    // Compute the number of repetitions required
    int repeatHeight = (height + originalHeight - 1) / originalHeight;
    int repeatWidth = (width + originalWidth - 1) / originalWidth;

    // Repeat the matrix horizontally and vertically
    torch::Tensor repeatedMatrix = matrix.repeat({repeatHeight, repeatWidth});

    // Crop the repeated matrix to the desired height and width
    torch::Tensor croppedMatrix = repeatedMatrix.slice(0, 0, height).slice(1, 0, width);

    // // Check distribution
    // std::unordered_map<float, int> distribution;
    // float interval=0.1;

    // // Flatten the matrix into a 1D tensor
    // auto flattenedMatrix = croppedMatrix.reshape({-1});

    // // Iterate over the flattened tensor and count the occurrences of each value
    // for (int i = 0; i < flattenedMatrix.size(0); ++i) {
    //     float value = flattenedMatrix[i].item<float>();
    //     float roundedValue = std::floor(value / interval) * interval;  // Round down to the nearest interval
    //     distribution[roundedValue]++;
    // }

    // std::cout << std::endl;
    // for (const auto& pair : distribution) {
    //     // std::cout << "Value: " << pair.first << ", Count: " << pair.second << std::endl;
    //     std::cout << pair.first << ", ";
    // }
    // std::cout << std::endl;
    // for (const auto& pair : distribution) {
    //     // std::cout << "Value: " << pair.first << ", Count: " << pair.second << std::endl;
    //     std::cout << pair.second << ", ";
    // }
    // std::cout << std::endl;

    return croppedMatrix;
}

void AMMBench::SIFTMatrixLoader::generateAB() {

    // Step1. locate file
    char filename[] = "/home/heyuhao/AMMBench/src/MatrixLoader/siftsmall_base.fvecs";
    float* data = NULL;
    unsigned num, dim;

    // Step2. read in binary
    std::ifstream in(filename, std::ios::binary);	//以二进制的方式打开文件
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    in.read((char*)&dim, 4);	//读取向量维度
    in.seekg(0, std::ios::end);	//光标定位到文件末尾
    std::ios::pos_type ss = in.tellg();	//获取文件大小（多少字节）
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);	//数据的个数
    data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);	//光标定位到起始处
    for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);	//光标向右移动4个字节
    in.read((char*)(data + i * dim), dim * 4);	//读取数据到一维数据data中
    }
    in.close();

    // Step3. convert to torch tensor and standardize the matrix
    torch::TensorOptions options(torch::kFloat32);
    B = torch::from_blob(data, {(int)num, (int)dim}, options).clone();

    // 3.1 Compute the mean and standard deviation along each feature (column)
    torch::Tensor mean = B.mean(/*dim=*/0);
    torch::Tensor std = B.std(/*dim=*/0);

    // 3.2 Standardize the matrix
    torch::Tensor standardizedB = (B - mean) / std;

    // 3.3 Check if need to resize
    if (aRow!=0 && aCol!=0 && bCol!=0){
        B = repeatAndCropMatrix(standardizedB, aCol, bCol);
    }
    else{
        B = standardizedB;
    }
    
    A = B.t();

    int ACol = A.size(0);
    int ARow = A.size(1);
    int BCol = B.size(0);
    int BRow = B.size(1);
    INTELLI_INFO(
            "Generating [" + to_string(ACol) + "x" + to_string(ARow) + "]*[" + to_string(BCol) + "x" + to_string(BRow) + "]");

    delete[] data; // deallocate
}

//do nothing in abstract class
bool AMMBench::SIFTMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
    paraseConfig(cfg);
    generateAB();
    return true;
}

torch::Tensor AMMBench::SIFTMatrixLoader::getA() {
    return A;
}

torch::Tensor AMMBench::SIFTMatrixLoader::getB() {
    return B;
}

// int main() {
//     AMMBench::MatrixLoaderTable mLoaderTable;
//     auto matLoaderPtr = mLoaderTable.findMatrixLoader("SIFT");
//     assert(matLoaderPtr);

//     INTELLI::ConfigMapPtr cfg = newConfigMap();
//     matLoaderPtr->setConfig(cfg);
    
//     auto A = matLoaderPtr->getA();
//     auto B = matLoaderPtr->getB();
//     std::cout << A.sizes() << endl;
//     std::cout << B.sizes() << endl;
// }