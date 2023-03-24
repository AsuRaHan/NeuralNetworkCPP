#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <random>

#include "NeuralNetwork.h"

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