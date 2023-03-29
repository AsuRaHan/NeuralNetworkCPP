#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <fstream>
#include <iostream>
#include <sstream>
#include <random>

void load_dataset(std::string filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    int wIter = 0;
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
        wIter++;
    }

    file.close();
}

bool fileExists(const std::string& fileName) {
    std::ifstream infile(fileName.c_str());
    return infile.good();
}

// Функция перемешивания данных
void shuffle_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(inputs.begin(), inputs.end(), gen);
    std::shuffle(targets.begin(), targets.end(), gen);
}

class NeuralNetwork {
private:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    double learningRate;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;

public:
    NeuralNetwork(int input, int hidden, int output, double rate = 0.1) {
        inputNodes = input;
        hiddenNodes = hidden;
        outputNodes = output;
        learningRate = rate;

        weightsInputHidden.resize(inputNodes, std::vector<double>(hiddenNodes));
        weightsHiddenOutput.resize(hiddenNodes, std::vector<double>(outputNodes));
        randomizeWeights();
    }

    void train(std::vector<double> inputs, std::vector<double> targets) {
        std::vector<double> hiddenOutputs(hiddenNodes);
        std::vector<double> finalOutputs(outputNodes);

        // Forward propagation
        for (int i = 0; i < hiddenNodes; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputNodes; j++) {
                sum += inputs[j] * weightsInputHidden[j][i];
            }
            hiddenOutputs[i] = sigmoid(sum);
        }

        for (int i = 0; i < outputNodes; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenNodes; j++) {
                sum += hiddenOutputs[j] * weightsHiddenOutput[j][i];
            }
            finalOutputs[i] = sigmoid(sum);
        }

        // Backpropagation
        std::vector<double> outputErrors(outputNodes);
        for (int i = 0; i < outputNodes; i++) {
            outputErrors[i] = (targets[i] - finalOutputs[i]) * sigmoidDerivative(finalOutputs[i]);
        }

        std::vector<double> hiddenErrors(hiddenNodes);
        for (int i = 0; i < hiddenNodes; i++) {
            double error = 0.0;
            for (int j = 0; j < outputNodes; j++) {
                error += outputErrors[j] * weightsHiddenOutput[i][j];
            }
            hiddenErrors[i] = error * sigmoidDerivative(hiddenOutputs[i]);
        }

        // Update weights
        for (int i = 0; i < hiddenNodes; i++) {
            for (int j = 0; j < inputNodes; j++) {
                weightsInputHidden[j][i] += learningRate * hiddenErrors[i] * inputs[j];
            }
        }

        for (int i = 0; i < outputNodes; i++) {
            for (int j = 0; j < hiddenNodes; j++) {
                weightsHiddenOutput[j][i] += learningRate * outputErrors[i] * hiddenOutputs[j];
            }
        }
    }

    std::vector<double> predict(std::vector<double> inputs) {
        std::vector<double> hiddenOutputs(hiddenNodes);
        std::vector<double> finalOutputs(outputNodes);

        for (int i = 0; i < hiddenNodes; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputNodes; j++) {
                sum += inputs[j] * weightsInputHidden[j][i];
            }
            hiddenOutputs[i] = sigmoid(sum);
        }

        for (int i = 0; i < outputNodes; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenNodes; j++) {
                sum += hiddenOutputs[j] * weightsHiddenOutput[j][i];
            }
            finalOutputs[i] = sigmoid(sum);
        }

        return finalOutputs;
    }

    void saveModelRaw(std::string fileName) {
        FILE* file;
        if (fopen_s(&file, fileName.c_str(), "wb") != 0) {
            std::cerr << "Ошибка открытия файла для записи: " << fileName << std::endl;
            return;
        }

        fwrite(&inputNodes, sizeof(int), 1, file);
        fwrite(&hiddenNodes, sizeof(int), 1, file);
        fwrite(&outputNodes, sizeof(int), 1, file);

        for (int i = 0; i < inputNodes; i++) {
            fwrite(&weightsInputHidden[i][0], sizeof(double), hiddenNodes, file);
        }

        for (int i = 0; i < hiddenNodes; i++) {
            fwrite(&weightsHiddenOutput[i][0], sizeof(double), outputNodes, file);
        }

        fclose(file);
    }
    // Метод для загрузки модели из файла
    void loadModelRaw(std::string fileName) {
        FILE* file;
        if (fopen_s(&file, fileName.c_str(), "rb") != 0) {
            std::cerr << "Ошибка открытия файла для чтения: " << fileName << std::endl;
            return;
        }

        fread(&inputNodes, sizeof(int), 1, file);
        fread(&hiddenNodes, sizeof(int), 1, file);
        fread(&outputNodes, sizeof(int), 1, file);

        weightsInputHidden.resize(inputNodes, std::vector<double>(hiddenNodes, 0.0));
        for (int i = 0; i < inputNodes; i++) {
            fread(&weightsInputHidden[i][0], sizeof(double), hiddenNodes, file);
        }

        weightsHiddenOutput.resize(hiddenNodes, std::vector<double>(outputNodes, 0.0));
        for (int i = 0; i < hiddenNodes; i++) {
            fread(&weightsHiddenOutput[i][0], sizeof(double), outputNodes, file);
        }

        fclose(file);
    }
private:

    void randomizeWeights() {
        for (int i = 0; i < inputNodes; i++) {
            for (int j = 0; j < hiddenNodes; j++) {
                weightsInputHidden[i][j] = randomWeight();
            }
        }

        for (int i = 0; i < hiddenNodes; i++) {
            for (int j = 0; j < outputNodes; j++) {
                weightsHiddenOutput[i][j] = randomWeight();
            }
        }
    }

    double randomWeight() {
        return (double)rand() / RAND_MAX - 0.5;
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    float relu(float x) {
        return std::max(0.0f, x);
    }

    float relu_prime(float x) {
        return (x >= 0) ? 1.0f : 0.0f;
    }
};


int main() {
    setlocale(LC_CTYPE, "Russian");

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;

    std::string fileName = "model.data";

    // Create neural network
    NeuralNetwork nn(4, 8, 3, 0.1);

    if (!fileExists(fileName)) {
        std::cout << "Файл " << fileName << " не найден." << std::endl;
        load_dataset("iris.data", inputs, outputs);
        //shuffle_data(inputs, outputs);
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < inputs.size(); j++) {
                nn.train(inputs[j], outputs[j]);
            }
        }


        // Test neural network
        int correct = 0;
        for (int i = 0; i < inputs.size(); i++) {
            std::vector<double> output = nn.predict(inputs[i]);
            int prediction = max_element(output.begin(), output.end()) - output.begin();
            std::cout << "The iris is a \033[1;31m" << prediction << "\033[0m = {" << outputs[i][0] << "," << outputs[i][1] << "," << outputs[i][2] << "} i=" << i << std::endl;
            std::cout << "The output = {" << output[0] << "," << output[1] << "," << output[2] << "}" << std::endl << std::endl;
            int target = max_element(inputs[i].begin(), inputs[i].end()) - inputs[i].begin();
            if (prediction == target) {
                correct++;
            }
        }

        std::cout << "Accuracy: " << (double)correct / inputs.size() << std::endl;


        nn.saveModelRaw(fileName);


    }
    else {
        std::cout << "Файл " << fileName << " существует!" << std::endl;
        nn.loadModelRaw(fileName);
    }


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

    std::vector<double> input = { sepal_length, sepal_width, petal_length, petal_width };
    std::vector<double> output = nn.predict(input);
    int class_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

    std::cout << "The iris is a " << class_index << "." << std::endl;

    return 0;
}