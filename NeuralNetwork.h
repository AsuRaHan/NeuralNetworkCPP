#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#include "Layer.h"

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes, double l_rate);
    std::vector<double> feedforward(const std::vector<double>& inputs);
    void backpropagate(std::vector<double> input, std::vector<double> target);
    void save_model(const std::string& filepath);
    //void load_model(const std::string& filepath);
private:
    std::vector<Layer> layers;

    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;

    std::vector<std::vector<double>> ih_weights; // веса связей между входным и скрытым слоями
    std::vector<std::vector<double>> ho_weights; // веса связей между скрытым и выходным слоями
    std::vector<double> ih_biases; // смещения для скрытого слоя
    std::vector<double> ho_biases; // смещения для выходного слоя

    std::vector<Neuron*> input_layer_;
    std::vector<Neuron*> hidden_layer_;
    std::vector<Neuron*> output_layer_;

    // функция активации sigmoid
    double sigmoid(double x);

    // производная функции активации sigmoid
    double sigmoid_derivative(double x);
};



