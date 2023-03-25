#pragma once
#include <vector>
#include <random>

class Neuron {
public:
    explicit Neuron(int num_inputs);

    double feedforward(const std::vector<double>& inputs);

    double get_output() const;

    void set_error_gradient(double error_gradient);

    double get_weight(int index) const;

    double get_error_gradient() const;

    void update_weights(double learning_rate);

private:
    std::vector<double> weights;
    std::vector<double> inputs_;

    double bias;
    double output_;
    double error_gradient_;
    
    int num_inputs_;

    static double random_weight();

    static inline double sigmoid(double x);
};


