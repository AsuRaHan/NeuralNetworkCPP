#pragma once

#include <vector>

class Layer {
public:
    Layer(int input_size, int output_size);
    std::vector<double> forward(std::vector<double> inputs);
    std::vector<double> get_weights();
    void set_weights(std::vector<double> weights);
    std::vector<double> get_biases();
    void set_biases(std::vector<double> biases);


    Layer() = default;
private:
    int input_size;
    int output_size;

    std::vector<double> weights;
    std::vector<double> biases;
    double random(double min, double max) {
        return min + (max - min) * static_cast<double>(rand()) / RAND_MAX;
    }
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
};