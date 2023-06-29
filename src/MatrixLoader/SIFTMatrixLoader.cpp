//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/SIFTMatrixLoader.h>
#include <Utils/IntelliLog.h>

void AMMBench::SIFTMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
    INTELLI_INFO("For SIFT dataset, parsing config step is skipped");
}

void AMMBench::SIFTMatrixLoader::generateAB() {

    char filename[] = "siftsmall_base.fvecs";
    float* data = NULL;
    unsigned num, dim;

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

    torch::Tensor tensor A = torch.tensor(float_ptr, size=((size_t)num * (size_t)dim,), dtype=torch.float32)

    torch::Tensor tensor B = A.t()

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