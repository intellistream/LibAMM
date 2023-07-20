//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/MNISTMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <AMMBench.h>
#include <fstream>
#include <iostream>

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

	string fileName = "../../../../../../datasets/train-images.idx3-ubyte";
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

    // Dynamically allocate memory for left half and right half
	A = torch::empty({60000, 28*14});
	B = torch::empty({60000, 28*14});

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
				if (pixel_value>255 || pixel_value<0){
					std::cout << pixel_value << " " << endl;
				}
				if ((j%28)<14){
					A[i][int(floor(static_cast<double>(j)/28)*14+(j%28))] = pixel_value; // 60000*392
				}
				else{
					B[i][int(floor(static_cast<double>(j)/28)*14+(j%28)-14)] = pixel_value; // 60000*392
				}
			}
		}
	
		// cout << "Finished reading images......" << endl;
	}
	file.close();

	// auto pickledA = torch::pickle_save(A);
	// std::ofstream foutA("A.pt", std::ios::out | std::ios::binary);
	// foutA.write(pickledA.data(), pickledA.size());
	// foutA.close();

	// auto pickledB = torch::pickle_save(B);
	// std::ofstream foutB("B.pt", std::ios::out | std::ios::binary);
	// foutB.write(pickledB.data(), pickledB.size());
	// foutB.close();

	// torch::save(A, "A.pt");
    // torch::save(B, "B.pt");

	// std::cout << A.mean(/*dim=*/0) << endl;
	// std::cout << B.mean(/*dim=*/0) << endl;

	// normalization and transpose
	// A = A - A.mean(/*dim=*/0); // mean along feature (column)
	// B = B - B.mean(/*dim=*/0); // mean along feature (column)
	// std::cout << A.mean(/*dim=*/0) << endl;
	// std::cout << B.mean(/*dim=*/0) << endl;
	// std::cout << A.std(/*dim=*/0) << endl;
	// std::cout << B.std(/*dim=*/0) << endl;

	A = A.t(); // 392*60000
	B = B.t(); // 392*60000
	// std::cout << A.sizes() << endl;
	// std::cout << B.sizes() << endl;

	INTELLI_INFO("Generating [" + to_string(A.size(0)) +  " x " + to_string(A.size(1)) + "]*[" + to_string(B.size(0)) +" x " + to_string(B.size(1)) + "]");
}

//do nothing in abstract class
bool AMMBench::MNISTMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
    paraseConfig(cfg);
    generateAB();
    return true;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getA() {
	std::cout << A.sizes() << endl;
    return A;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getB() {
	std::cout << B.sizes() << endl;
    return B;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getSxx() {
    return Sxx;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getSyy() {
    return Syy;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getSxy() {
    return Sxy;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getSxxNegativeHalf() {
    return SxxNegativeHalf;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getSyyNegativeHalf() {
    return SyyNegativeHalf;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getM() {
    return M;
}

torch::Tensor AMMBench::MNISTMatrixLoader::getCorrelation() {
    return correlation;
}

void AMMBench::MNISTMatrixLoader::calculate_correlation() {

	// Sxx, Syy, Sxy: covariance matrix
	Sxx = torch::matmul(A, A.t())/A.size(1); // 392x60000 * 60000x392 max 12752.4 min -3912.51
	Syy = torch::matmul(B, B.t())/A.size(1); // 392x60000 * 60000x392 max 12953.4 min -5121.09
	Sxy = torch::matmul(A, B.t())/A.size(1); // 392x60000 * 60000x392 max 10653.1 min -5307.15

	// Sxx^(-1/2), Syy^(-1/2), M
	// Sxx^(-1/2)
	torch::Tensor eigenvaluesSxx, eigenvectorsSxx;
	std::tie(eigenvaluesSxx, eigenvectorsSxx) = torch::linalg::eig(Sxx); // diagonization
	torch::Tensor diagonalMatrixSxx = torch::diag(1.0 / torch::sqrt(eigenvaluesSxx+torch::full({}, 1e-12))); // 1/sqrt(eigenvalue+epsilon) +epsilon to avoid nan
	SxxNegativeHalf = torch::matmul(torch::matmul(eigenvectorsSxx, diagonalMatrixSxx), eigenvectorsSxx.t());
	SxxNegativeHalf = at::real(SxxNegativeHalf); // ignore complex part, it comes from numerical computations
	// Syy^(-1/2)
	torch::Tensor eigenvaluesSyy, eigenvectorsSyy;
	std::tie(eigenvaluesSyy, eigenvectorsSyy) = torch::linalg::eig(Syy);
	torch::Tensor diagonalMatrixSyy = torch::diag(1.0 / torch::sqrt(eigenvaluesSyy+torch::full({}, 1e-12)));
	SyyNegativeHalf = torch::matmul(torch::matmul(eigenvectorsSyy, diagonalMatrixSyy), eigenvectorsSyy.t());
	SyyNegativeHalf = at::real(SyyNegativeHalf);
	// M
	M = torch::matmul(torch::matmul(SxxNegativeHalf, Sxy), SyyNegativeHalf);
	
	// correlation
	torch::Tensor U, S, Vh;
    std::tie(U, S, Vh) = torch::linalg::svd(M, false, c10::nullopt);
	correlation = torch::clamp(S, -1.0, 1.0);
}


// int main() {
//     AMMBench::MatrixLoaderTable mLoaderTable;
//     std::shared_ptr<AMMBench::AbstractMatrixLoader> basePtr = mLoaderTable.findMatrixLoader("MNIST");
// 	std::shared_ptr<AMMBench::MNISTMatrixLoader> matLoaderPtr = std::dynamic_pointer_cast<AMMBench::MNISTMatrixLoader>(basePtr);
//     assert(matLoaderPtr);

//     INTELLI::ConfigMapPtr cfg = newConfigMap();
//     matLoaderPtr->setConfig(cfg);
    
//     auto A = matLoaderPtr->getA();
//     auto B = matLoaderPtr->getB();
//     std::cout << A.sizes() << endl;
//     std::cout << B.sizes() << endl;

// 	// if you want to visualize the image, pls comment out normalization and transpose step in generateAB()
// 	// auto A0 = A[0].view({28, 14});
// 	// auto B0 = B[0].view({28, 14});
// 	// std::cout << A0 << endl;
//     // std::cout << B0 << endl;

// 	matLoaderPtr->calculate_correlation();
// }