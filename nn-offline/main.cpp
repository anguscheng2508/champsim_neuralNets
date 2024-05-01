// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "rnn.h"
#include <typeinfo>
#include <vector>
#include <utility>


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
//    "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,410.bwaves-1963B.txt"
    //        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,450.soplex-92B.txt"
    //        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,483.xalancbmk-127B.txt"
    //        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,602.gcc_s-2226B.txt"
    //        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,603.bwaves_s-2609B.txt"
    FILE *in_file = fopen("/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_exapp,410.bwaves-1963B.txt", "r");
    if(in_file == NULL) {
        printf("Error file missing\n");
        return 1;
    }
    
    char delta[] = "Delta_in_range:";
    char sim_instr[] = "Simulation";
    
    while(fscanf(in_file,"%s", string_match) == 1) {
        // If statement to get the number of simulation instructions on start up.
        //Do at the start of file so we can declare an array for the size of number of instructions. aka the number of deltas.
        if (numDeltas_found == 0) {
            if(strstr(string_match, sim_instr)!=0) {
                fgets(str_numInstr, 60, in_file);
                printf("String value num instr:%s", str_numInstr);
                numDeltas_found = 1;
                for(int i = 0; i < strlen(str_numInstr); i++) {
                    if(isdigit(str_numInstr[i])) {
                        char d[1];
                        d[0] = str_numInstr[i];
                        int digit = atoi(d);
                        num_delta = (num_delta * 10) + digit;
                    }
                }
            }
        }
        //        float delta_array[num_delta + 10]; //+10 to allow for headroom of any extra instructions executed.
        //        printf("%i\n", num_delta);
        
        
        //Add a for loop till strstr(string, delta) does-not returns null.
        if(strstr(string_match, delta)!=0) {//if match found
            fgets(str_deltaValue, 60, in_file);//get the delta value from the line
            int_delta = strtol(str_deltaValue, &stopstring, 10);//cast str to int for use
            delta_vector.push_back(int_delta);
        }
        
        //Print delta vector
//      for (float i: delta_vector)
//          std::cout << i << ' ';
        
    }
    fclose(in_file);
    
    
// *************************** Data Vector *************************
// I need to pair the sequence with a label
// e.g. tensor[([0, 1, 2], [1]), ([10, 11, 12], [11])]
//    std::vector<std::pair<std::vector<float>, std::vector<float>>> data2 = {
//        { {1, 2, 3}, {1} },
//        { {5, 6, 7}, {8} }
//    };
    // Define a dataset in C++
    std::vector<std::pair<std::vector<float>, float>> data3;
    
    // Number of deltas being used as input for the sequence
    int num_inputs = 3;
    for (int i = 0; i < delta_vector.size(); i++) {
        // Extract input sequence (three consecutive numbers)
        std::vector<float> input_sequence(delta_vector.begin() + i, delta_vector.begin() + i + num_inputs);

        // Extract label (next number after the input sequence)
        float label = delta_vector[i + num_inputs];

        // Add the sample to the dataset
        data3.push_back({input_sequence, label});
    }
    
    // Print and access the data
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
    
    std::cout << "Recurrent Neural Network\n\n";
    
    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
    
    // Hyper parameters
    
    // Note: input_size is set to 1 for the LSTM layer since the input is a sequence of single numbers
    const int64_t input_size = 1;
    const int64_t hidden_size = 48;
    const int64_t num_layers = 3;
    const int64_t num_classes = 1;
    
    const size_t num_epochs = 5;
    const double learning_rate = 0.00001;
    
//    // Example usage
//
//    size_t batch_size = 64; // Adjust batch size according to your requirements
//    bool shuffle = true;    // Whether to shuffle the data during training
//
//    // Create an instance of CustomDataLoader with your data
//    CustomDataLoader data_loader(data3, batch_size, shuffle);
//
//    // Access the DataLoader
//    auto loader = data_loader.get_loader();
//    
//
//
//    // Iterate over the DataLoader
//    for (auto& batch : *loader) {
//        auto inputs = batch.data;   // Input tensor
//        auto targets = batch.target; // Target tensor
//
//        // Your training loop goes here...
//    }

    
    auto num_train_samples = data3.size();
    
    // Model
    RNN model(input_size, hidden_size, num_layers, num_classes);
    model->to(device);
    
    torch::nn::MSELoss loss_func;
    
    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    
    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << "Number of inputs: " << num_inputs << std::endl;
    std::cout << "Training...\n";
    
    
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_correct = 0;

        // Iterate over your dataset manually
        for (const auto& example : data3) {
            const std::vector<float>& input_sequence = example.first;
//            std::vector<float> input_sequence = example.first;
            float target = example.second;
            
            //**** Print input sequence and the target ****
//            std::cout << "Input Sequence: [";
//            for (const float& value : input_sequence) {
//                std::cout << value << ", ";
//            }
//            std::cout << "]" << std::endl;
//            std::cout << "Target " << target << std::endl;
            //**** Printing Done *****

            // Convert input_sequence to torch::Tensor
            torch::Tensor input_tensor = torch::tensor(input_sequence, torch::kFloat32).view({1, static_cast<long long>(input_sequence.size()), 1});

            // Convert target to torch::Tensor
            torch::Tensor target_tensor = torch::tensor({target}, torch::kFloat32);
            
            
            // Forward pass
            auto output = model->forward(input_tensor);
//            std::cout << "Output shape: " << output.sizes() << std::endl;
//            std::cout << "Output tensor: " << output << std::endl;
            
            // Extract the last element along the first dimension of 'output'. Extract single scalar from tensor using .item
            auto prediction = output[-1].item<float>();
            //std::cout << "Predicted: " << prediction << std::endl << std::endl;

            // Update number of correctly classified samples
            num_correct += static_cast<int64_t>(prediction) == static_cast<int64_t>(target) ? 1 : 0;


            // Compute the loss
//            auto loss_value = (output.squeeze(1), target_tensor);
            auto loss = torch::nn::functional::mse_loss(output.squeeze(1), target_tensor);
            // Update running loss
            running_loss += loss.item<double>() * input_sequence.size();

            // Backward pass
            optimizer.zero_grad();
            loss.backward();

            // Optimization step
            optimizer.step();

            // Accumulate the loss
//            running_loss += loss_value.item<double>() * input_sequence.size();
//            std::cout << "Num correct: " << num_correct << std::endl;
//            std::cout << "Num train samples: " << num_train_samples << std::endl;
            
        }
        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<float>(num_correct) / num_train_samples;
        


        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Accuracy: " << accuracy << '\n';
        
//        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
//            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }
    std::cout << "Training finished!\n\n";
    
    return 0;
}
