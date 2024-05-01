	//
//  example-app.cpp
//  
//
//  Created by Angus Cheng on 08/02/2024.
//

#include <stdio.h>
#include <torch/torch.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <algorithm>
#include <torch/script.h>


// Copyright 2020-present pytorch-cpp Authors
#include <iomanip>
#include "neural_net.h"



int main() {
//********** This section will read in deltas from a text file and store in variable ********
    char word[2000];
    char string_match[50];
    char str_deltaValue[60], str_numInstr[60];
    long int_delta, int_numInstr;
    char *stopstring;
    
    int num_delta = 0;
    int numDeltas_found = 0;
        
    std::vector<float> delta_vector;
    // Define a dataset in C++
    std::vector<std::pair<std::vector<float>, float>> data3;
    int num_inputs = 3;

    
    // List of file paths
    std::vector<std::string> file_paths = {
        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,410.bwaves-1963B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,450.soplex-92B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,483.xalancbmk-127B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,602.gcc_s-2226B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,603.bwaves_s-2609B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,619.lbm_s-2677B.txt",
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,654.roms_s-523B.txt"
        // Add more file paths as needed
    };

    // Array of FILE pointers
    std::vector<FILE*> file_pointers;

    // Open each file
    for (const auto& file_path : file_paths) {
        FILE* file_ptr = fopen(file_path.c_str(), "r");
        if (file_ptr == nullptr) {
            std::cout << "Error opening file: " << file_path << std::endl;
            // Close already opened files
            for (auto fp : file_pointers) {
                fclose(fp);
            }
            return 1;
        }
        // Add the file pointer to the array
        file_pointers.push_back(file_ptr);

        // Read and process the file
        char string_match[2000];
        char delta[] = "Delta_in_range:";
        char sim_instr[] = "Simulation";

        while (fscanf(file_ptr, "%s", string_match) == 1) {
            // If statement to get the number of simulation instructions on startup.
            // Do at the start of the file so we can declare an array for the size of the number of instructions, aka the number of deltas.
            if (numDeltas_found == 0) {
                if (strstr(string_match, sim_instr) != 0) {
                    fgets(str_numInstr, 60, file_ptr);
                    numDeltas_found = 1;
                    for (int i = 0; i < strlen(str_numInstr); i++) {
                        if (isdigit(str_numInstr[i])) {
                            char d[1];
                            d[0] = str_numInstr[i];
                            int digit = atoi(d);
                            num_delta = (num_delta * 10) + digit;
                        }
                    }
                }
            }
            
            // Add a for loop till strstr(string, delta) does not return null.
            if (strstr(string_match, delta) != 0) { // If match found
                fgets(str_deltaValue, 60, file_ptr); // Get the delta value from the line
                int_delta = strtol(str_deltaValue, &stopstring, 10); // Cast str to int for use
                delta_vector.push_back(int_delta);
                
            }
        }
        for (int i = 0; i < delta_vector.size(); i++) {
            // Extract input sequence (three consecutive numbers)
            std::vector<float> input_sequence(delta_vector.begin() + i, delta_vector.begin() + i + num_inputs);

            // Extract label (next number after the input sequence)
            float label = delta_vector[i + num_inputs];

            // Add the sample to the dataset
            data3.push_back({input_sequence, label});
        }
        // Close the file
        fclose(file_ptr);
    }

    // Close all the files when done
    for (auto fp : file_pointers) {
        fclose(fp);
    }
    
//*************************** Data Vector *************************
//   I need to pair the sequence with a label
//   e.g. tensor[([0, 1, 2], [1]), ([10, 11, 12], [11])]
//    Look here, this is an example of what i have in my data3 std::vector<std::pair<std::vector<float>, std::vector<float>>> data3 = {
//        { {1, 2, 3}, {1} },
//        { {5, 6, 7}, {8} }
//    };

    

    
//    // Print and access the data
//    for (const auto& sample : data3) {
//        const auto& input_sequence = sample.first;
//        const auto& label = sample.second;
//
//        // Print the input sequence
//        std::cout << "Input Sequence: ";
//        for (float value : input_sequence) {
//            std::cout << value << " ";
//        }
//
//        // Print the label
//        std::cout << "\nLabel: " << label << "\n\n";
//    }
//************************* End Section *******************************
    
    
//************************* Neural Network *******************************
    std::cout << "Neural Network\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
    
    // Hyper parameters
    const int64_t input_size = num_inputs;
    const int64_t hidden_size = 48;
    const int64_t num_classes = 1;
    const int64_t batch_size = 40;
    const size_t num_epochs = 10;
    const double learning_rate = 0.00001;
    auto num_train_samples = data3.size();
    
    // Neural Network model
    NeuralNet model(input_size, hidden_size, num_classes);
    model->to(device);
    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));
    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);
    
    
    std::cout << "Training...\n";
    std::cout << "Hidden Size: " << hidden_size << std::endl;
    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        double epoch_loss = 0.0; // Accumulator for epoch loss
        size_t num_correct = 0;
        
        // Define inputTensor outside the loop
        torch::Tensor inputTensor;

        // Train the model using batches
        for (size_t batch_start = 0; batch_start < data3.size(); batch_start += batch_size) {
            // Collect inputs and targets for the current batch
            std::vector<torch::Tensor> batch_inputs, batch_targets;
            for (size_t i = batch_start; i < std::min<int>(batch_start + batch_size, data3.size()); ++i) {
                batch_inputs.push_back(torch::tensor(data3[i].first));
                batch_targets.push_back(torch::tensor(data3[i].second));
            }
            
            // Concatenate tensors along a new dimension to create batches
            inputTensor = torch::stack(batch_inputs).unsqueeze(1);
            auto targetTensor = torch::stack(batch_targets).unsqueeze(1);
            
            // Forward pass
            auto output = model->forward(inputTensor);
            
            auto loss = torch::nn::functional::mse_loss(output.squeeze(2), targetTensor);
            
            // Extract predictions from the last output
            auto prediction = output[-1].item<float>();

            // Calculate accuracy
            for (size_t i = 0; i < targetTensor.size(0); ++i) {
                float target = targetTensor[i].item<float>();
                num_correct += static_cast<int64_t>(round(prediction)) == static_cast<int64_t>(target) ? 1 : 0;
            }
            
            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            // Accumulate batch loss
            epoch_loss += loss.item<double>();
        }

        // Calculate average loss for the epoch
        epoch_loss /= static_cast<double>(data3.size() / batch_size);
        auto accuracy = static_cast<float>(num_correct) / data3.size();

        // Print epoch loss and accuracy
        std::cout << "\nEpoch [" << (epoch + 1) << "/" << num_epochs <<
                    "], Loss: " << epoch_loss << ", Accuracy: " << accuracy*100 << "\n";
        
    }
    
    std::cout << "Training finished!\n\n";
    //************************* End Section *******************************
    
//    // Convert input sequences to a tensors
//    std::vector<torch::Tensor> input_tensors;
//    for (const auto& sample : data3) {
//        const auto& input_sequence = sample.first;
//        input_tensors.push_back(torch::tensor(input_sequence));
//    }
//    
//    
//    std::vector<torch::Tensor> saving = model->parameters();
//    
////    // Save the model to a file
////    std::string model_path = "/Users/anguscheng/nn-offline/model.pt"; // Replace "/path/to/save/model.pt" with your desired file path
////    torch::save(saving, model_path);
//    
//    
//    // Save the parameters to a file
//    std::string params_path = "/Users/anguscheng/nn-offline/params.pt";
//    torch::save(model->parameters(), params_path);
//    
//    std::cout << "Parameters saved to: " << params_path << std::endl;
//    
//    // Load the parameters from the file
//    std::vector<torch::Tensor> loaded_params;
//    torch::load(loaded_params, params_path);
//    
//    // Assign loaded parameters to the model
//    auto model_params = model->parameters();
//    for (size_t i = 0; i < model_params.size(); ++i) {
//        model_params[i].set_data(loaded_params[i]);
//    }
//
//    
//    // Perform inference using the model
//    torch::Tensor input_tensor = torch::stack(input_tensors);
//    torch::Tensor output_tensor = model->forward(input_tensor);
////    std::cout << "Output tensor: " << output_tensor << std::endl;
//    // Extract predictions from the last output
//    auto prediction = round(output_tensor[-1].item<float>());
//    std::cout << prediction << std::endl;
    
    return 0;
    
}


