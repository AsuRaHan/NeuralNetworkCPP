#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <memory>
#include <execution>

#include "NeuralNetwork.h"




class IrisClassifier {
public:
    IrisClassifier(const std::string& filename, int num_hidden_neurons)
        : num_output_neurons(3), learning_rate(0.1)
    {
        // Load the data from a file
        load_data(filename, inputs, outputs);
        // Initialize the neural network
        std::vector<int> layer_sizes = { 4, num_hidden_neurons, num_output_neurons };
        net = NeuralNetwork(layer_sizes);
    }

    void load_data(std::string filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            std::stringstream ss(line);
            std::string token;
            std::vector<double> input_row;
            std::vector<double> output_row;

            // Read the input values
            for (int i = 0; i < 4; i++) {
                if (std::getline(ss, token, ',')) {
                    input_row.push_back(std::stod(token));
                }
                else {
                    throw std::runtime_error("Invalid data format in file: " + filename);
                }
            }

            // Read the output values
            if (std::getline(ss, token)) {
                if (token == "Iris-setosa") {
                    output_row = { 1.0, 0.0, 0.0 };
                }
                else if (token == "Iris-versicolor") {
                    output_row = { 0.0, 1.0, 0.0 };
                }
                else if (token == "Iris-virginica") {
                    output_row = { 0.0, 0.0, 1.0 };
                }
                else {
                    throw std::runtime_error("Invalid data format in file: " + filename);
                }
            }
            else {
                throw std::runtime_error("Invalid data format in file: " + filename);
            }

            inputs.push_back(input_row);
            outputs.push_back(output_row);
        }

        file.close();
    }

    void train(int num_epochs) {
        // Shuffle the data
        std::shuffle(training_indices.begin(), training_indices.end(),
            std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

        // Train the neural network
        for (int epoch = 1; epoch <= num_epochs; ++epoch) {
            double total_error = 0.0;
            for (int i : training_indices) {
                std::vector<double> predicted = net.feedforward(inputs[i]);
                std::vector<double> error(outputs[i].size());
                std::transform(outputs[i].begin(), outputs[i].end(), predicted.begin(), error.begin(), std::minus<double>());
                net.backpropagate(error, learning_rate);
                total_error += std::inner_product(error.begin(), error.end(), error.begin(), 0.0);
            }
            std::cout << "Epoch " << epoch << " error: " << total_error << std::endl;
        }
    }

    std::string classify(double sepal_length, double sepal_width, double petal_length, double petal_width) {
        std::vector<double> input = { sepal_length, sepal_width, petal_length, petal_width };
        std::vector<double> output = net.feedforward(input);
        int class_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        switch (class_index) {
        case 0: return "Iris Setosa";
        case 1: return "Iris Versicolour";
        case 2: return "Iris Virginica";
        default: throw std::runtime_error("Unknown class index");
        }
    }

private:
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    std::vector<int> training_indices = { 0, 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149 };
    int num_output_neurons;
    double learning_rate;
    NeuralNetwork net;
};


int main() {
    try {
        // Create an instance of the IrisClassifier
        IrisClassifier classifier("iris.data", 5);

        // Train the classifier
        classifier.train(100);

        // Test the classifier
        std::cout << "Classifying new iris:" << std::endl;
        std::cout << "Sepal length (cm): ";
        double sepal_length;
        std::cin >> sepal_length;
        std::cout << "Sepal width (cm): ";
        double sepal_width;
        std::cin >> sepal_width;
        std::cout << "Petal length (cm): ";
        double petal_length;
        std::cin >> petal_length;
        std::cout << "Petal width (cm): ";
        double petal_width;
        std::cin >> petal_width;
        std::string result = classifier.classify(sepal_length, sepal_width, petal_length, petal_width);
        std::cout << "The iris is a " << result << "." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
