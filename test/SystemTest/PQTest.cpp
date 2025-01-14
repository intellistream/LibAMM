
//
// Created by haolan on 25/6/23.
//
#include <vector>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <LibAMM.h>
#include <iostream>
TEST_CASE("Test PQ", "[short]")
{
  torch::manual_seed(114514);
  LibAMM::ProductQuantizationRaw pqRaw;
  auto A = torch::rand({1000, 1000});
  auto B = torch::rand({1000, 1000});
  auto realC = torch::matmul(A, B);
  auto ammC = pqRaw.amm(A, B, 20);
  std::cout << "PQ:" << std::endl;
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
  std::cout << froError << std::endl;
  // REQUIRE(froError < 0.5);
}
TEST_CASE("Test PQ Hash", "[short]")
{
  torch::manual_seed(114514);
  LibAMM::ProductQuantizationHash pqHash;
  auto A = torch::rand({1000, 1000});
  auto B = torch::rand({1000, 1000});
  auto realC = torch::matmul(A, B);
  auto ammC = pqHash.amm(A, B, 20);
  std::cout << "PQ Hash:" << std::endl;
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
  std::cout << froError << std::endl;
  // REQUIRE(froError < 0.5);
}

TEST_CASE("Test VQ", "[short]")
{
  torch::manual_seed(114514);
  LibAMM::VectorQuantization vq;
  auto A = torch::rand({1000, 1000});
  auto B = torch::rand({1000, 1000});
  auto realC = torch::matmul(A, B);
  auto ammC = vq.amm(A, B, 1000);
  std::cout << "VQ:" << std::endl;
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
  std::cout << froError << std::endl;
  // REQUIRE(froError < 0.5);
}