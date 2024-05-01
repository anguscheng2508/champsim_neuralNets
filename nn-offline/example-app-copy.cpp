//
//  example-app.cpp
//  
//
//  Created by Angus Cheng on 08/02/2024.
//
//  This is a test file to check that all libtorch libraries/dependencies are correctly installed

#include <stdio.h>
#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
