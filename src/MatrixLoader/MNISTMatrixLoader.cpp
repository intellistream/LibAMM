//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/MNISTMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <AMMBench.h>

void AMMBench::MNISTMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
    assert(cfg); // do nothing, as cfg is not needed
    INTELLI_INFO("For MNIST dataset, parsing config step is skipped");
}

int reverseInt(int i) { 
   
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void AMMBench::MNISTMatrixLoader::generateAB() {

    string fileName = "/home/yuhao/Documents/work/SUTD/AMM/codespace/AMMBench/src/MatrixLoader/train-images.idx3-ubyte";
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

    // Dynamically allocate memory for left half and right half
    float** leftHalf = new float*[60000];
    for (int i = 0; i < 60000; ++i) {
        leftHalf[i] = new float[28*14];
    }
    float** rightHalf = new float*[60000];
    for (int i = 0; i < 60000; ++i) {
        rightHalf[i] = new float[28*14];
    }

    // Read in file
	ifstream file(fileName, ios::binary);
	if (file.is_open())
	{ 
   
		// cout << "Reading metadata ..." << endl;

		file.read((char*)&magic_number, sizeof(magic_number));//幻数（文件格式）
		file.read((char*)&number_of_images, sizeof(number_of_images));//图像总数
		file.read((char*)&n_rows, sizeof(n_rows));//每个图像的行数
		file.read((char*)&n_cols, sizeof(n_cols));//每个图像的列数

		magic_number = reverseInt(magic_number);
		number_of_images = reverseInt(number_of_images);
		n_rows = reverseInt(n_rows);
		n_cols = reverseInt(n_cols);
        INTELLI_INFO(
            "File format:" + to_string(magic_number) + " Number of Images:" + to_string(number_of_images) + " Number of Rows:" + to_string(n_rows) + " Number of Cols:" + to_string(n_cols));

		// cout << "Reading images......" << endl;

		for (int i = 0; i < number_of_images; i++) {
			for (int j = 0; j < n_rows * n_cols; j++) {
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				//可以在下面这一步将每个像素值归一化
				float pixel_value = float(temp);
				if ((j%28)<14){
					leftHalf[i][int(floor(static_cast<double>(j)/28)*14+(j%28))] = pixel_value;
				}
				else{
					rightHalf[i][int(floor(static_cast<double>(j)/28)*14+(j%28)-14)] = pixel_value;
				}
			}
		}

		// cout << "Finished reading images......" << endl;

        INTELLI_INFO("Generating [60000 x 392]*[60000 x 392]");

	}
	file.close();

    torch::TensorOptions options(torch::kFloat32);
    A = torch::from_blob(leftHalf[0], {60000, 28*14}, options).clone();
    B = torch::from_blob(rightHalf[0], {60000, 28*14}, options).clone();

    // Deallocate memory for data when no longer needed
    for (int i = 0; i < 60000; ++i) {
    delete[] leftHalf[i];
    }
    delete[] leftHalf;
	for (int i = 0; i < 60000; ++i) {
    delete[] rightHalf[i];
    }
    delete[] rightHalf;
}

//do nothing in abstract class
bool AMMBench::MNISTMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
    paraseConfig(cfg);
    generateAB();
    return true;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getA() {
    return A;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getB() {
    return B;
}

// int main() {
//     AMMBench::MatrixLoaderTable mLoaderTable;
//     auto matLoaderPtr = mLoaderTable.findMatrixLoader("MNIST");
//     assert(matLoaderPtr);

//     INTELLI::ConfigMapPtr cfg = newConfigMap();
//     matLoaderPtr->setConfig(cfg);
    
//     auto A = matLoaderPtr->getA();
//     auto B = matLoaderPtr->getB();
//     std::cout << A.sizes() << endl;
//     std::cout << B.sizes() << endl;
//     // std::cout << A[0] << endl;
//     // std::cout << B[0] << endl;
// }