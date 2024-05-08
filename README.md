# ChampSim Neural Nets

This repository is my barebones progress of developing neural networks for use with ChampSim simulations. It was developed as part of my final year project for my 4th year. The aim is to develop efficient neural network architectures to be used for prefetching. I have only performed preliminary 'offline' testing, where the input data for the neural networks are readily prepared. The goal is to build 'online' architectures, where the neural network is performing predictions and training 'on-board' while the CPU is running. 


## Information
This assumes that ChampSim and traces have been installed locally.
I used Libtorch (PyTorch for C++) for the machine learning libraries, and compiled/built on a Mac with Intel core. The Libtorch libraries/dependencies can be installed on their website. 
After successful installation/build of Libtorch dependencies, make sure to keep note of where they were installed to - as it will be necessary to add their paths to the Makefiles.

The 'Champsim' folder contains the necessary files to compile and build the ChamSim binary. The outputs of the simulation will have to be stored into a text file via a Python script.

The 'nn' folder contains a simple neural network architecture for offline testing.

The 'rnn' folder contains a preliminary recurrent neural network architecture for testing.
