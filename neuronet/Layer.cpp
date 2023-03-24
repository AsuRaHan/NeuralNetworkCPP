#include "Layer.h"

#include <cmath>
#include <random>

using namespace std;

Layer::Layer(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;
    weights.resize(input_size * output_size);
    biases.resize(output_size);
    for (int i = 0; i < weights.size(); i++) {
        weights[i] = random(-1.0, 1.0);
    }
    for (int i = 0; i < biases.size(); i++) {
        biases[i] = random(-1.0, 1.0);
    }
}

std::vector<double> Layer::forward(std::vector<double> inputs) {
    std::vector<double> outputs(output_size, 0.0);
    for (int i = 0; i < output_size; i++) {
        double activation = 0.0;
        for (int j = 0; j < input_size; j++) {
            activation += inputs[j] * weights[i * input_size + j];
        }
        activation += biases[i];
        outputs[i] = sigmoid(activation);
    }
    return outputs;
}
std::vector<double> Layer::get_weights() {
    return weights;
}
void Layer::set_weights(std::vector<double> weights) {
    this->weights = weights;
}
std::vector<double> Layer::get_biases() {
    return biases;
}
void Layer::set_biases(std::vector<double> biases) {
    this->biases = biases;
}
//void Layer::backward(const std::vector<double>& inputs, const std::vector<double>& outputs, const std::vector<double>& next_layer_weights, const std::vector<double>& next_layer_delta) {
//    for (int i = 0; i < num_neurons(); i++) {
//        double error = 0.0;
//        if (next_layer_weights.empty()) {
//            error = outputs[i] - inputs[i];
//        }
//        else {
//            for (int j = 0; j < next_layer_delta.size(); j++) {
//                error += next_layer_delta[j] * next_layer_weights[j][i];
//            }
//        }
//        delta_[i] = error * outputs[i] * (1.0 - outputs[i]);
//    }
//}

//void Layer::randomize_weights() {
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> dist(-1.0, 1.0);
//    for (int i = 0; i < num_neurons(); i++) {
//        for (int j = 0; j < input_size(); j++) {
//            weights_[i][j] = dist(gen);
//        }
//        bias_[i] = dist(gen);
//    }
//}