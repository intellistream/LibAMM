#include <vector>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace std;

TEST_CASE("Test basic", "[short]")
{
  int a = 0;
  // place your test here
  REQUIRE(a == 0);
}