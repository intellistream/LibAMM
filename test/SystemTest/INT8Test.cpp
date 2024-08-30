//
// Created by tony on 05/06/23.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <LibAMM.h>
#include <iostream>

TEST_CASE("Test int8", "[short]")
{
  torch::manual_seed(114514);
  LibAMM::INT8CPPAlgo int8mm;
  auto A = torch::rand({4, 4});
  auto B = torch::rand({4, 4});
  auto realC = torch::matmul(A, B);
  auto ammC = int8mm.amm(A, B, 20);
  std::cout << "int8:" << std::endl;
  std::cout << ammC << std::endl;
  std::cout << "fp32:" << std::endl;
  std::cout << realC << std::endl;
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
  REQUIRE(froError < 0.5);
}