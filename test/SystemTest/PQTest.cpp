//
// Created by haolan on 25/6/23.
//
#include <vector>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <AMMBench.h>
#include <iostream>
TEST_CASE("Test PQ", "[short]")
{
    torch::manual_seed(114514);
    AMMBench::ProductQuantizationRaw pqRaw;
    auto A = torch::rand({1000, 1000});
    auto B = torch::rand({1000, 1000});
    auto realC = torch::matmul(A, B);
    auto ammC = pqRaw.amm(A, B, 20);
    std::cout << "PQ:" << std::endl;
    std::cout << ammC << std::endl;
    std::cout << "exact:" << std::endl;
    std::cout << realC << std::endl;
    double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
    REQUIRE(froError < 0.5);
}
TEST_CASE("Test PQ Hash", "[short]")
{
    torch::manual_seed(114514);
    AMMBench::ProductQuantizationHash pqHash;
    auto A = torch::rand({1000, 1000});
    auto B = torch::rand({1000, 1000});
    auto realC = torch::matmul(A, B);
    auto ammC = pqHash.amm(A, B, 20);
    std::cout << "PQ:" << std::endl;
    std::cout << ammC << std::endl;
    std::cout << "exact:" << std::endl;
    std::cout << realC << std::endl;
    double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
    REQUIRE(froError < 0.5);
}