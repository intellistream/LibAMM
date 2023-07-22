#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <AMMBench.h>

using namespace std;
using namespace INTELLI;
using namespace torch;

TEST_CASE("Test RIP in cpp", "[short]")
{
  torch::manual_seed(114514);
  AMMBench::RIPCPPAlgo rip;
  auto A = torch::rand({600, 400});
  auto B = torch::rand({400, 1000});
  auto realC = torch::matmul(A, B);
  auto ammC = rip.amm(A, B, 20);
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
  std::cout << froError << std::endl;
  // REQUIRE(froError < 0.5);
}