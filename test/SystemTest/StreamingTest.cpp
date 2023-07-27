#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <AMMBench.h>
#include <iostream>

using namespace std;
using namespace INTELLI;
using namespace torch;

/*
TEST_CASE("Test the basic streaming batch 1", "[short]")
{
    ConfigMapPtr cfg = newConfigMap();
    torch::manual_seed(114514);
    auto A = torch::rand({(long) 4, (long) 4});
    auto B = torch::rand({(long) 4, (long) 4});
    auto rawC = torch::matmul(A, B);
    AMMBench::SingleThreadStreamer ss;
    //cfg->edit("",(uint64_t)100);
    ss.setConfig(cfg);
    auto ssC = ss.streamingAmm(A, B);
    std::cout << "raw C:" << std::endl;
    std::cout << rawC << std::endl;
    std::cout << "streaming C:" << std::endl;
    std::cout << ssC << std::endl;
    std::cout << "95% latency=" + to_string(ss.getLatencyPercentage(0.95)) << std::endl;
    std::cout << "throughput=" + to_string(ss.getThroughput()) << std::endl;
    // REQUIRE(froError < 0.5);
}
*/

TEST_CASE("Test the basic streaming batch 2", "[short]")
{
  ConfigMapPtr cfg = newConfigMap();
  torch::manual_seed(114514);
  auto A = torch::rand({(long) 4, (long) 4});
  auto B = torch::rand({(long) 4, (long) 4});
  auto rawC = torch::matmul(A, B);
  AMMBench::SingleThreadStreamer ss;
  cfg->edit("batchSize", (uint64_t) 2);
  ss.setConfig(cfg);
  ss.prepareRun(A, B);
  auto ssC = ss.streamingAmm(A, B);
  std::cout << "raw C:" << std::endl;
  std::cout << rawC << std::endl;
  std::cout << "streaming C:" << std::endl;
  std::cout << ssC << std::endl;
  std::cout << "95% latency=" + to_string(ss.getLatencyPercentage(0.95)) << std::endl;
  std::cout << "throughput=" + to_string(ss.getThroughput()) << std::endl;
  // REQUIRE(froError < 0.5);
}
TEST_CASE("Test the basic streaming batch 1, 2 matrix in streaming", "[short]")
{
  ConfigMapPtr cfg = newConfigMap();
  torch::manual_seed(114514);
  auto A = torch::rand({(long) 4, (long) 4});
  auto B = torch::rand({(long) 4, (long) 4});
  auto rawC = torch::matmul(A, B);
  AMMBench::SingleThreadStreamer ss;
  cfg->edit("batchSize", (uint64_t) 2);
  //cfg->edit("",(uint64_t)100);
  ss.setConfig(cfg);
  ss.prepareRun(A, B);
  auto ssC = ss.streamingAmm2S(A, B);
  std::cout << "raw C:" << std::endl;
  std::cout << rawC << std::endl;
  std::cout << "streaming C:" << std::endl;
  std::cout << ssC << std::endl;
  std::cout << "95% latency=" + to_string(ss.getLatencyPercentage(0.95)) << std::endl;
  std::cout << "throughput=" + to_string(ss.getThroughput()) << std::endl;
  // REQUIRE(froError < 0.5);
}

TEST_CASE("Test the basic streaming batch 1, 2 matrix in streaming, full lazy", "[short]")
{
  ConfigMapPtr cfg = newConfigMap();
  torch::manual_seed(114514);
  auto A = torch::rand({(long) 4, (long) 4});
  auto B = torch::rand({(long) 4, (long) 4});
  auto rawC = torch::matmul(A, B);
  AMMBench::SingleThreadStreamer ss;
  cfg->edit("batchSize", (uint64_t) 2);
  cfg->edit("fullLazt", (uint64_t) 2);
  //cfg->edit("",(uint64_t)100);
  ss.setConfig(cfg);
  ss.prepareRun(A, B);
  auto ssC = ss.streamingAmm2S(A, B);
  std::cout << "raw C:" << std::endl;
  std::cout << rawC << std::endl;
  std::cout << "streaming C:" << std::endl;
  std::cout << ssC << std::endl;
  std::cout << "95% latency=" + to_string(ss.getLatencyPercentage(0.95)) << std::endl;
  std::cout << "throughput=" + to_string(ss.getThroughput()) << std::endl;
  // REQUIRE(froError < 0.5);
}