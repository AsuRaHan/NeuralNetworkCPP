#include "Layer.h"

Layer::Layer(int num_inputs, int num_outputs) {
    for (int i = 0; i < num_outputs; i++) {
        neurons.push_back(Neuron(num_inputs));
    }
}

std::vector<double> Layer::feedforward(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    for (int i = 0; i < neurons.size(); i++) {
        outputs.push_back(neurons[i].feedforward(inputs));
    }
    return outputs;
}
// Методы доступа к отдельным нейронам и их весам
int Layer::size() const {
    return neurons.size();
}