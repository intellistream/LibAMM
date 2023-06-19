#include <vector>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <AMMBench.h>
using namespace std;
using namespace INTELLI;
using namespace torch;

TEST_CASE("Test Load PQ", "[short]")
{
  torch::serialize::InputArchive archive;
  archive.load_from("torchscripts/PQ/prototypes.pt");
  torch::Tensor prototypes;
  archive.read("prototypes", prototypes);
  auto  pt_size = prototypes.sizes();
  std::cout<<"prototype size:"+ to_string(pt_size[0])+"x"+to_string(pt_size[1])+"x"+to_string(pt_size[2])<<std::endl;
  std::cout<<prototypes<<endl;
  std::cout<<"print first one"<<endl;
  std::cout<<prototypes[0]<<endl;
  std::cout<<prototypes[0][0][0]<<endl;
}