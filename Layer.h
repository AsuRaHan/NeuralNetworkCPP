#pragma once
#include <vector>
#include "Neuron.h"

class Layer {
public:
    Layer(int num_inputs, int num_outputs);

    std::vector<double> feedforward(const std::vector<double>& inputs);
    // Методы доступа к отдельным нейронам и их весам
    int size() const;
private:
    std::vector<Neuron> neurons;
};


