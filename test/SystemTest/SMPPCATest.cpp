#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <LibAMM.h>

using namespace std;
using namespace INTELLI;
using namespace torch;

TEST_CASE("Test SMPPCA in cpp", "[short]")
{
  torch::manual_seed(114514);
  LibAMM::SMPPCACPPAlgo smppca;
  auto A = torch::rand({600, 400});
  auto B = torch::rand({400, 1000});
  auto realC = torch::matmul(A, B);
  auto ammC = smppca.amm(A, B, 20);
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
  std::cout << froError << std::endl;
  // REQUIRE(froError < 0.5);
}