//
// Created by haolan on 17/7/23.
//
#include <CPPAlgos/VectorQuantization.h>

using namespace std;

int findClosestDivisor(int A, int B) {
  for (int i = B; i >= 1; --i) {
    if (A % i == 0) {
      return i;
    }
  }
  return 1;
}

void AMMBench::VectorQuantization::setConfig(INTELLI::ConfigMapPtr cfg) {
    columnCodeIndexXPath = cfg->tryString("columnCodeIndexXPath", "torchscripts/columnCodeIndexX.txt", true);
    rowCodeIndexYPath = cfg->tryString("rowCodeIndexYPath", "torchscripts/rowCodeIndexY.txt", true);
    columnCodeBookXvecPath = cfg->tryString("columnCodeBookXvecPath", "torchscripts/columnCodeBookXvec.txt", true);
    rowCodeBookYvecPath = cfg->tryString("rowCodeBookYvecPath", "torchscripts/rowCodeBookYvec.txt", true);
    INTELLI_INFO("columnCodeIndexXPath: " + columnCodeIndexXPath);
    INTELLI_INFO("rowCodeIndexYPath: " + rowCodeIndexYPath);
    INTELLI_INFO("columnCodeBookXvecPath: " + columnCodeBookXvecPath);
    INTELLI_INFO("rowCodeBookYvecPath: " + rowCodeBookYvecPath);
    }

torch::Tensor AMMBench::VectorQuantization::amm(const torch::Tensor A, const torch::Tensor B, uint64_t l) {

  torch::Tensor A1 = A.to(torch::kDouble);
  torch::Tensor B1 = B.to(torch::kDouble);

  // int m = A1.size(1) / findClosestDivisor(A1.size(1), l);
  // int m=1;
  // l=765;
  int m=1;
  l=1700;
  PQMM pqmm(A1, B1, l, m);
  pqmm.setFilePath(columnCodeIndexXPath, rowCodeIndexYPath, columnCodeBookXvecPath, rowCodeBookYvecPath);
  return pqmm.runAMM(true);
}

void AMMBench::PQMM::setFilePath(string columnCodeIndexXPathPassedIn, string rowCodeIndexYPathPassedIn, string columnCodeBookXvecPathPassedIn, string rowCodeBookYvecPathPassedIn){
  columnCodeIndexXPath = columnCodeIndexXPathPassedIn;
  rowCodeIndexYPath = rowCodeIndexYPathPassedIn;
  columnCodeBookXvecPath = columnCodeBookXvecPathPassedIn;
  rowCodeBookYvecPath = rowCodeBookYvecPathPassedIn;
}

torch::Tensor AMMBench::PQMM::runAMM(bool training) {
  // 1. initialize result matrix
  res = torch::zeros({X.size(0), Y.size(1)});

  // 2. construct codebooks
  if (training) constructCodeBooks();
  else {
    vector<vector<vector<double>>> columnCodeBookXvec, rowCodeBookYvec;  // codebooks mxlxM, mxlxK
    load2DVectorIntFromFile(columnCodeIndexXPath, columnCodeIndexX);
    load2DVectorIntFromFile(rowCodeIndexYPath, rowCodeIndexY);
    this->m = columnCodeIndexX.size();
    this->l = columnCodeIndexX[0].size();

    // For columnCodeBookX, rowCodeBookY
    load3DVectorDoubleFromFile(columnCodeBookXvecPath, columnCodeBookXvec);
    load3DVectorDoubleFromFile(rowCodeBookYvecPath, rowCodeBookYvec);

    for (vector<vector<double>> vec : columnCodeBookXvec) {
      std::vector<double> flat_vec;
      for (const auto &v : vec) {
        flat_vec.insert(flat_vec.end(), v.begin(), v.end());
      }

      // Convert the flattened 1D vector to a 1D tensor
      torch::Tensor tensor = torch::tensor(flat_vec);
      // Reshape the 1D tensor to 2D tensor
      columnCodeBookX.push_back(tensor.view({static_cast<long>(vec.size()), static_cast<long>(vec[0].size())}));
    }

    for (vector<vector<double>> vec : rowCodeBookYvec) {
      std::vector<double> flat_vec;
      for (const auto &v : vec) {
        flat_vec.insert(flat_vec.end(), v.begin(), v.end());
      }

      // Convert the flattened 1D vector to a 1D tensor
      torch::Tensor tensor = torch::tensor(flat_vec);
      // Reshape the 1D tensor to 2D tensor
      rowCodeBookY.push_back(tensor.view({static_cast<long>(vec.size()), static_cast<long>(vec[0].size())}));
    }
  }

  // 3. run
  for (int i = 0; i < m; ++i) {
    // for every sub-codebook
    for (int j = 0; j < columnCodeBookX[i].size(0); ++j) {
      torch::Tensor codeX = columnCodeBookX[i][columnCodeIndexX[i][j]];
      torch::Tensor codeY = rowCodeBookY[i][rowCodeIndexY[i][j]];
      torch::Tensor outerProduct = matrixOuterProduct(codeX, codeY); // MxK
      res += outerProduct;
    }
  }

  // 4. free memory
  vector<torch::Tensor>().swap(columnCodeBookX);
  vector<torch::Tensor>().swap(rowCodeBookY);
  vector<vector<int>>().swap(columnCodeIndexX);
  vector<vector<int>>().swap(rowCodeIndexY);

  return res;
}

torch::Tensor AMMBench::PQMM::matrixOuterProduct(torch::Tensor A, torch::Tensor B) {
  return torch::ger(A, B); // compute the outer product
}

void AMMBench::PQMM::save2DVectorIntToFile(string filename, vector<vector<int>> &vec) {
  ofstream file(filename);
  if (file.is_open()) {
    for (const auto &subVec : vec) {
      for (const auto &num : subVec) {
        file << num << ',';
      }
      file << '\n';
    }
    file.close();
  } else {
    cout << "Unable to open file";
  }
}

void AMMBench::PQMM::save3DVectorDoubleToFile(string filename, vector<vector<vector<double>>> &vec) {
  ofstream file(filename);
  if (file.is_open()) {
    for (const auto &matrix : vec) {
      for (const auto &subVec : matrix) {
        for (const auto &num : subVec) {
          file << num << ',';
        }
        file << '|';
      }
      file << '\n';
    }
    file.close();
  } else {
    cout << "Unable to open file";
  }
}

void AMMBench::PQMM::load2DVectorIntFromFile(string filename, vector<vector<int>> &vec) {
  ifstream file(filename);
  if (file.is_open()) {
    string line;
    while (getline(file, line)) {
      vector<int> subVec;
      stringstream ss(line);
      string numStr;
      while (getline(ss, numStr, ',')) {
        subVec.push_back(stoi(numStr));
      }
      vec.push_back(subVec);
    }
    file.close();
  } else {
    cout << "Unable to open file";
  }
}

void AMMBench::PQMM::load3DVectorDoubleFromFile(string filename, vector<vector<vector<double>>> &vec) {
  ifstream file(filename);
  if (file.is_open()) {
    string line;
    while (getline(file, line)) {
      vector<vector<double>> matrix;
      stringstream ss(line);
      string vecStr;
      while (getline(ss, vecStr, '|')) {
        stringstream ssVec(vecStr);
        string numStr;
        vector<double> subVec;
        while (getline(ssVec, numStr, ',')) {
          subVec.push_back(stod(numStr));
        }
        matrix.push_back(subVec);
      }
      vec.push_back(matrix);
    }
    file.close();
  } else {
    cout << "Unable to open file";
  }
}

class Point {
 public:
  int clusterId;
  int dimensions;
  vector<double> values;

  Point(vector<double> feature) {
    values = feature;
    dimensions = values.size();
    clusterId = 0; // Initially not assigned to any cluster
  }

  int getDimensions() { return dimensions; }

  int getCluster() { return clusterId; }

  void setCluster(int val) { clusterId = val; }

  double getVal(int pos) { return values[pos]; }
};

class Cluster {
 public:
  int clusterId;
  vector<double> centroid;
  vector<Point> points;

  Cluster(int clusterId, Point centroid) {
    this->clusterId = clusterId;
    for (int i = 0; i < centroid.getDimensions(); i++) {
      this->centroid.push_back(centroid.getVal(i));
    }
    this->addPoint(centroid);
  }

  void addPoint(Point p) {
    p.setCluster(this->clusterId);
    points.push_back(p);
  }

  void removeAllPoints() { points.clear(); }

  int getId() { return clusterId; }

  Point getPoint(int pos) { return points[pos]; }

  int getSize() { return points.size(); }

  double getCentroidByPos(int pos) { return centroid[pos]; }

  void setCentroidByPos(int pos, double val) { this->centroid[pos] = val; }
};

class KMeans {
 private:
  int K, iters, dimensions, total_points;
  vector<Cluster> clusters;

  vector<Point> all_points;

  void clearClusters() {
    for (int i = 0; i < K; i++) {
      clusters[i].removeAllPoints();
    }
  }

  int getNearestClusterId(Point point) {
    double sum = 0.0, min_dist;
    int NearestClusterId;
    if (dimensions == 1) {
      min_dist = abs(clusters[0].getCentroidByPos(0) - point.getVal(0));
    } else {
      for (int i = 0; i < dimensions; i++) {
        sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
        // sum += abs(clusters[0].getCentroidByPos(i) - point.getVal(i));
      }
      min_dist = sqrt(sum);
    }
    NearestClusterId = clusters[0].getId();

    for (int i = 1; i < K; i++) {
      double dist;
      sum = 0.0;

      if (dimensions == 1) {
        dist = abs(clusters[i].getCentroidByPos(0) - point.getVal(0));
      } else {
        for (int j = 0; j < dimensions; j++) {
          sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
          // sum += abs(clusters[i].getCentroidByPos(j) - point.getVal(j));
        }

        dist = sqrt(sum);
        // dist = sum;
      }
      if (dist < min_dist) {
        min_dist = dist;
        NearestClusterId = clusters[i].getId();
      }
    }

    return NearestClusterId;
  }

 public:
  KMeans(int K, int iterations, vector<vector<double>> raw_points) {
    this->K = K;
    this->iters = iterations;
    for (const auto &raw_point : raw_points) {
      Point p(raw_point);
      all_points.push_back(p);
    }
  }

  vector<vector<double>> getClusterCenters() {
    vector<vector<double>> clusterCenters;
    for (int i = 0; i < K; i++) {
      clusterCenters.push_back(clusters[i].centroid);
    }
    return clusterCenters;
  }

  vector<int> getClusterIndexOfPoints() {
    vector<int> clusterIndexList;
    for (int i = 0; i < total_points; i++) {
      clusterIndexList.push_back(all_points[i].getCluster() - 1);
    }
    return clusterIndexList;
  }

  void run() {
    total_points = all_points.size();
    dimensions = all_points[0].getDimensions();

    // Initializing Clusters
    vector<int> used_pointIds;

    for (int i = 1; i <= K; i++) {
      while (true) {
        int index = rand() % total_points;

        if (find(used_pointIds.begin(), used_pointIds.end(), index) ==
            used_pointIds.end()) {
          used_pointIds.push_back(index);
          all_points[index].setCluster(i);
          Cluster cluster(i, all_points[index]);
          clusters.push_back(cluster);
          break;
        }
      }
    }
    // cout << "Clusters initialized = " << clusters.size() << endl
    //      << endl;

    // cout << "Running K-Means Clustering.." << endl;

    int iter = 1;
    while (true) {
      // cout << "Iter - " << iter << "/" << iters << endl;
      bool done = true;

      // Add all points to their nearest cluster
      // #pragma omp parallel for reduction(&&: done) num_threads(16)
      for (int i = 0; i < total_points; i++) {
        int currentClusterId = all_points[i].getCluster();
        int nearestClusterId = getNearestClusterId(all_points[i]);

        if (currentClusterId != nearestClusterId) {
          all_points[i].setCluster(nearestClusterId);
          done = false;
        }
      }

      // clear all existing clusters
      clearClusters();

      // reassign points to their new clusters
      for (int i = 0; i < total_points; i++) {
        // cluster index is ID-1
        clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
      }

      // Recalculating the center of each cluster
      for (int i = 0; i < K; i++) {
        int ClusterSize = clusters[i].getSize();

        for (int j = 0; j < dimensions; j++) {
          double sum = 0.0;
          if (ClusterSize > 0) {
            // #pragma omp parallel for reduction(+: sum) num_threads(16)
            for (int p = 0; p < ClusterSize; p++) {
              sum += clusters[i].getPoint(p).getVal(j);
            }
            clusters[i].setCentroidByPos(j, sum / ClusterSize);
          }
        }
      }

      if (done || iter >= iters) {
        // cout << "Clustering completed in iteration : " << iter << endl
        //  << endl;
        break;
      }
      iter++;
    }
  }
};

void AMMBench::PQMM::constructCodeBooks() {
  INTELLI_WARNING("constructCodeBooks..");
  // define max interation times for KMeans using the max value of int
  int maxValue = std::numeric_limits<int>::max();

  X = X.contiguous();
  // Convert tensor to a 1D vector
  std::vector<double> flat_vec(X.data_ptr<double>(), X.data_ptr<double>() + X.numel());

  // Reshape the 1D vector to 2D vector
  std::vector<std::vector<double>> vecX;
  for (int i = 0; i < X.size(0); i++) {
    std::vector<double> row(flat_vec.begin() + i * X.size(1), flat_vec.begin() + (i + 1) * X.size(1));
    vecX.push_back(row);
  }

  Y = Y.contiguous();
  // Convert tensor to a 1D vector
  std::vector<double> flat_vecY(Y.data_ptr<double>(), Y.data_ptr<double>() + Y.numel());

  // Reshape the 1D vector to 2D vector
  std::vector<std::vector<double>> vecY;
  for (int i = 0; i < Y.size(0); i++) {
    std::vector<double> row(flat_vec.begin() + i * Y.size(1), flat_vec.begin() + (i + 1) * Y.size(1));
    vecY.push_back(row);
  }

  int d = vecX[0].size();
  int gap = ceil(d / m);
  assert(gap >= l);
  vector<vector<vector<double>>> columnCodeBookXvec, rowCodeBookYvec;  // codebooks mxlxM, mxlxK
  for (int i = 0; i < m; i++) {
    // 1. construct subVector of X, please note that for X we need to extract its column as the member of codebook
    vector<vector<double>> subColumnsX;
    // need to consider the size of the last block may not be larger than l
    for (int j = i * gap; j < d && j < (i + 1) * gap; j++) {
      vector<double> column;
      for (auto &c : vecX) {
        column.push_back(c[j]);
      }
      subColumnsX.push_back(column);
    }

    // 2. run kmeans on subVector of X
    KMeans kmeansX(l, maxValue, subColumnsX);
    kmeansX.run();
    vector<vector<double>> columnCentroidsX = kmeansX.getClusterCenters();

    columnCodeBookXvec.push_back(columnCentroidsX);
    columnCodeIndexX.push_back(kmeansX.getClusterIndexOfPoints());

    // 3. construct subVector of Y
    vector<vector<double>> subRowsY;
    for (int j = i * gap; j < d && j < (i + 1) * gap; j++) {
      subRowsY.push_back(vecY[j]);
    }

    // 4. run kmeans on subVector of Y, please note that for Y we need to extract its row as the member of codebook
    KMeans kmeansY(l, maxValue, subRowsY);
    kmeansY.run();
    vector<vector<double>> rowcentroidsY = kmeansY.getClusterCenters();
    rowCodeBookYvec.push_back(rowcentroidsY);
    rowCodeIndexY.push_back(kmeansY.getClusterIndexOfPoints());
  }

  // For columnCodeIndexX, rowCodeIndexY
  save2DVectorIntToFile(columnCodeIndexXPath, columnCodeIndexX);
  save2DVectorIntToFile(rowCodeIndexYPath, rowCodeIndexY);

  // For columnCodeBookX, rowCodeBookY
  save3DVectorDoubleToFile(columnCodeBookXvecPath, columnCodeBookXvec);
  save3DVectorDoubleToFile(rowCodeBookYvecPath, rowCodeBookYvec);

  // reconstruct columnCodeBookX and rowCodeBookY
  for (vector<vector<double>> vec : columnCodeBookXvec) {
    std::vector<double> flat_vec;
    for (const auto &v : vec) {
      flat_vec.insert(flat_vec.end(), v.begin(), v.end());
    }

    // Convert the flattened 1D vector to a 1D tensor
    torch::Tensor tensor = torch::tensor(flat_vec);
    // Reshape the 1D tensor to 2D tensor
    columnCodeBookX.push_back(tensor.view({static_cast<long>(vec.size()), static_cast<long>(vec[0].size())}));
  }

  for (vector<vector<double>> vec : rowCodeBookYvec) {
    std::vector<double> flat_vec;
    for (const auto &v : vec) {
      flat_vec.insert(flat_vec.end(), v.begin(), v.end());
    }

    // Convert the flattened 1D vector to a 1D tensor
    torch::Tensor tensor = torch::tensor(flat_vec);
    // Reshape the 1D tensor to 2D tensor
    rowCodeBookY.push_back(tensor.view({static_cast<long>(vec.size()), static_cast<long>(vec[0].size())}));
  }

  cout << "construct codebooks done!" << endl;
}