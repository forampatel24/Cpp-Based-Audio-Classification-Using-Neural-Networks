cmake_minimum_required(VERSION 4.1)
project(CP)

set(CMAKE_CXX_STANDARD 20)

add_executable(CP
        main.cpp
        AudioLoader.cpp
        FeatureExtractor.cpp
        Preprocessor.cpp
        NeuralNetwork.cpp
        Trainer.cpp
        Evaluator.cpp
        Utils.cpp)
