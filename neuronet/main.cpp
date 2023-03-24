#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <random>

#include "NeuralNetwork.h"

/* 
Функция load_data отвечает за загрузку данных из файла в два вектора: вектор входных данных и вектор целевых значений. Файл представляет собой таблицу, где каждая строка содержит значения четырех признаков и один целевой класс (0, 1 или 2).
Аргументами функции являются ссылки на векторы входных данных и целевых значений, куда будут записываться данные из файла.
Внутри функции файл открывается с помощью std::ifstream. Затем читается каждая строка файла с помощью std::getline.
Строка разбивается на отдельные значения признаков и целевого класса с помощью std::stringstream. Значения признаков добавляются в вектор row, а значение целевого класса преобразуется в вектор длиной 3 с помощью цикла и добавляется в вектор целевых значений targets.
В конце строки вектор row добавляется в вектор входных данных inputs.
Функция возвращает true, если загрузка данных прошла успешно, и false, если произошла ошибка.
*/
void load_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets) {
    std::ifstream file("iris.csv");
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
}

// Функция перемешивания данных
void shuffle_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(inputs.begin(), inputs.end(), gen);
    std::shuffle(targets.begin(), targets.end(), gen);
}

/*
Функция split_data отвечает за разбиение данных на обучающую и тестовую выборки. Аргументами функции являются вектор входных данных, вектор целевых значений, и процентное соотношение размера тестовой выборки к общему размеру данных.
Внутри функции происходит вычисление количества примеров, которые должны быть в тестовой выборке, и количество примеров, которые должны быть в обучающей выборке.
Затем, индексы примеров перемешиваются случайным образом с помощью функции std::random_shuffle.
Далее, из перемешанного списка индексов берется заданное количество индексов для тестовой выборки и для обучающей выборки. Векторы входных данных и целевых значений разбиваются на обучающую и тестовую выборки соответственно, используя выбранные индексы.
Функция возвращает кортеж из четырех векторов: вектор обучающих входных данных, вектор обучающих целевых значений, вектор тестовых входных данных и вектор тестовых целевых значений.
Разбиение данных на обучающую и тестовую выборки позволяет оценить качество работы нейронной сети на неизвестных данных и избежать переобучения (overfitting) модели на обучающей выборке.
*/

void split_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets,
    std::vector<std::vector<double>>& train_inputs, std::vector<std::vector<double>>& train_targets,
    std::vector<std::vector<double>>& test_inputs, std::vector<std::vector<double>>& test_targets) {
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
}

/*
Функция test_network отвечает за тестирование нейронной сети на тестовых данных. Аргументами функции являются ссылки на тестовые данные, целевые значения тестовых данных и обученную нейронную сеть.
Внутри функции происходит проход по всем тестовым данным. Для каждого тестового примера, входные данные передаются в нейронную сеть методом feedforward, который возвращает выходные значения сети.
Затем, выходные значения сети сравниваются с целевыми значениями из тестовых данных. Сравнение происходит путем выбора индекса максимального значения вектора выходных значений и сравнения этого индекса с целевым значением.
Для каждого тестового примера, функция увеличивает счетчик num_correct, если сеть выдала правильный предсказанный класс.
В конце функция выводит процент правильно предсказанных классов на тестовых данных и возвращает значение точности в процентах.
*/

double test_network(NeuralNetwork& nn, std::vector<std::vector<double>>& test_inputs, std::vector<std::vector<double>>& test_targets) {
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
    return accuracy;
}

int main() {
    // Загрузка данных из файла
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
    load_data(inputs, targets);

    // Перемешивание данных
    shuffle_data(inputs, targets);

    // Разделение данных на обучающую и тестовую выборки
    std::vector<std::vector<double>> train_inputs;
    std::vector<std::vector<double>> train_targets;
    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_targets;
    split_data(inputs, targets, train_inputs, train_targets, test_inputs, test_targets);

    // Создание нейронной сети
    NeuralNetwork nn(4, 16, 3);

    // Обучение нейронной сети
    nn.train(train_inputs, train_targets, 0.1, 1000);

    // Тестирование нейронной сети
    double accuracy = test_network(nn, test_inputs, test_targets);

    return 0;
}