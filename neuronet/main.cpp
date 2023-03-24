#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <random>

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
using namespace std;

// Класс слоя нейросети
// Определение класса Layer
class Layer {
public:
    Layer(int input_size, int output_size) {
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
    Layer() = default;
    std::vector<double> forward(std::vector<double> inputs) {
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
    std::vector<double> get_weights() {
        return weights;
    }
    void set_weights(std::vector<double> weights) {
        this->weights = weights;
    }
    std::vector<double> get_biases() {
        return biases;
    }
    void set_biases(std::vector<double> biases) {
        this->biases = biases;
    }
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

// Определение класса NeuralNetwork
class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size) {
        this->input_size = input_size;
        this->hidden_layer = Layer(input_size, hidden_size);
        this->output_layer = Layer(hidden_size, output_size);
    }
    void train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets, double learning_rate, int num_epochs) {
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            for (int j = 0; j < inputs.size(); j++) {
                // Прямое распространение
                std::vector<double> hidden_outputs = hidden_layer.forward(inputs[j]);
                std::vector<double> outputs = output_layer.forward(hidden_outputs);
                // Вычисление ошибки
                std::vector<double> output_errors(output_size);
                for (int k = 0; k < output_size; k++) {
                    output_errors[k] = targets[j][k] - outputs[k];
                }
                std::vector<double> hidden_errors(hidden_layer.get_biases().size());
                for (int k = 0; k < hidden_layer.get_biases().size(); k++) {
                    double error = 0.0;
                    for (int l = 0; l < output_size; l++) {
                        error += output_errors[l] * output_layer.get_weights()[k * output_size + l];
                    }
                    hidden_errors[k] = error * hidden_outputs[k] * (1.0 - hidden_outputs[k]);
                }
                // Обновление весов и смещений
                std::vector<double> output_weights = output_layer.get_weights();
                std::vector<double> output_biases = output_layer.get_biases();
                for (int k = 0; k < output_size; k++) {
                    for (int l = 0; l < hidden_outputs.size(); l++) {
                        output_weights[k * hidden_outputs.size() + l] += learning_rate * output_errors[k] * hidden_outputs[l];
                    }
                    output_biases[k] += learning_rate * output_errors[k];
                }
                output_layer.set_weights(output_weights);
                output_layer.set_biases(output_biases);
                std::vector<double> hidden_weights = hidden_layer.get_weights();
                std::vector<double> hidden_biases = hidden_layer.get_biases();
                for (int k = 0; k < hidden_layer.get_biases().size(); k++) {
                    for (int l = 0; l < input_size; l++) {
                        hidden_weights[k * input_size + l] += learning_rate * hidden_errors[k] * inputs[j][l];
                    }
                    hidden_biases[k] += learning_rate * hidden_errors[k];
                }
                hidden_layer.set_weights(hidden_weights);
                hidden_layer.set_biases(hidden_biases);
            }
        }
    }
    std::vector<double> predict(std::vector<double> inputs) {
        std::vector<double> hidden_outputs = hidden_layer.forward(inputs);
        std::vector<double> outputs = output_layer.forward(hidden_outputs);
        return outputs;
    }
private:
    int input_size;
    Layer hidden_layer;
    Layer output_layer;
    int output_size = output_layer.get_biases().size();
};
// Основная функция
int main() {
    // Загрузка данных из файла
    std::ifstream file("iris.csv");
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        int i = 0;
        while (ss >> value) {
            if (i < 4) {
                row.push_back(value);
            }
            else {
                std::vector<double> target(3, 0.0);
                target[static_cast<int>(value)] = 1.0;
                targets.push_back(target);
            }
            if (ss.peek() == ',') {
                ss.ignore();
            }
            i++;
        }
        inputs.push_back(row);
    }
    file.close();

    // Разделение данных на обучающую и тестовую выборки
    std::vector<std::vector<double>> train_inputs;
    std::vector<std::vector<double>> train_targets;
    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_targets;
    for (int i = 0; i < inputs.size(); i++) {
        if (i % 5 == 0) {
            test_inputs.push_back(inputs[i]);
            test_targets.push_back(targets[i]);
        }
        else {
            train_inputs.push_back(inputs[i]);
            train_targets.push_back(targets[i]);
        }
    }

    // Создание нейронной сети
    NeuralNetwork nn(4, 16, 3);

    // Обучение нейронной сети
    nn.train(train_inputs, train_targets, 0.1, 1000);

    // Тестирование нейронной сети
    int num_correct = 0;
    for (int i = 0; i < test_inputs.size(); i++) {
        std::vector<double> outputs = nn.predict(test_inputs[i]);
        int predicted_class = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
        int true_class = std::distance(test_targets[i].begin(), std::max_element(test_targets[i].begin(), test_targets[i].end()));
        if (predicted_class == true_class) {
            num_correct++;
        }
    }
    double accuracy = static_cast<double>(num_correct) / test_inputs.size();
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}