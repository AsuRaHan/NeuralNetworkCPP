#include "Neuron.h"

Neuron::Neuron(int num_inputs) : 
    weights(num_inputs), 
    bias(random_weight()), 
    output_(0.0), 
    error_gradient_(0.0), 
    inputs_(num_inputs) {
    num_inputs_ = num_inputs;
    for (auto& weight : weights) {
        weight = random_weight();
    }
}

double Neuron::feedforward(const std::vector<double>& inputs) {
    double total = 0.0;
    for (int i = 0; i < inputs.size(); ++i) {
        total += inputs[i] * weights[i];
    }

    total += bias;
    output_ = sigmoid(total);
    inputs_ = inputs;
    return output_;
}

double Neuron::get_output() const {
    return output_;
}

void Neuron::set_error_gradient(double error_gradient) {
    error_gradient_ = error_gradient;
}

double Neuron::get_weight(int index) const {
    return weights[index];
}

double Neuron::get_error_gradient() const {
    return error_gradient_;
}

void Neuron::update_weights(double learning_rate) {
    const double factor = learning_rate * error_gradient_ * output_ * (1 - output_);
    for (int i = 0; i < weights.size(); i++) {
        weights[i] += factor * inputs_[i];
    }
    bias += factor;
}

double Neuron::random_weight() {
    // Initialize weights with random values between -1 and 1
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(-1.0, 1.0);
    return dis(gen);
}

inline double Neuron::sigmoid(double x) {
    // Sigmoid activation function
    return 1 / (1 + exp(-x));
}