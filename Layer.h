#pragma once
#include <vector>
#include "Neuron.h"

class Layer {
public:
    Layer(int num_inputs, int num_outputs);

    std::vector<double> feedforward(const std::vector<double>& inputs);
    // ������ ������� � ��������� �������� � �� �����
    int size() const;
private:
    std::vector<Neuron> neurons;
};


