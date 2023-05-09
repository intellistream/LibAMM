// Copyright (C) 2021 by the IntelliStream team (https://github.com/intellistream)

/**
 * @brief This is the main entry point of the entire program.
 * We use this as the entry point for benchmarking.
 */

#include <torch/torch.h>
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <memory>
using namespace std;

int main() {
  torch::jit::script::Module module;
//do some example function, here is the fdamm
  module = torch::jit::load("torchscripts/FDAMM.pt");
  auto A = torch::rand({10000, 1000});
  auto B = torch::rand({5000, 1000});
  auto C =module.forward({A, B, 25}).toTensor();
  return 0;
}

