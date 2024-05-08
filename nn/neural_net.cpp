// Copyright 2020-present pytorch-cpp Authors
#include "neural_net.h"
#include <torch/torch.h>

NeuralNetImpl::NeuralNetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes)
    : fc1(input_size, hidden_size), fc3(hidden_size, hidden_size), fc4(hidden_size, hidden_size), fc5(hidden_size, hidden_size), fc6(hidden_size, hidden_size), fc2(hidden_size, num_classes) {
    register_module("fc1", fc1);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    register_module("fc5", fc5);
    register_module("fc6", fc6);
//    register_module("fc7", fc7);
//    register_module("fc8", fc8);
//    register_module("fc9", fc9);
//    register_module("fc10", fc10);
    register_module("fc2", fc2);
}

torch::Tensor NeuralNetImpl::forward(torch::Tensor x) {
    x = torch::nn::functional::relu(fc1->forward(x));
    x = torch::nn::functional::relu(fc3->forward(x));
    x = torch::nn::functional::relu(fc4->forward(x));
    x = torch::nn::functional::relu(fc5->forward(x));
    x = torch::nn::functional::relu(fc6->forward(x));
//    x = torch::nn::functional::relu(fc7->forward(x));
//    x = torch::nn::functional::relu(fc8->forward(x));
//    x = torch::nn::functional::relu(fc9->forward(x));
//    x = torch::nn::functional::relu(fc10->forward(x));
    return fc2->forward(x);
}
