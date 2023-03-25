#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes, double l_rate) {

    this->input_size = layer_sizes[0];
    this->hidden_size = layer_sizes[1];
    this->output_size = layer_sizes[2];
    this->learning_rate = l_rate;

    for (int i = 1; i < layer_sizes.size(); i++) {
        layers.push_back(Layer(layer_sizes[i - 1], layer_sizes[i]));
    }

    // Инициализируем веса между входным и скрытым слоями
    ih_weights.resize(input_size);
    for (int i = 0; i < input_size; i++)
    {
        ih_weights[i].resize(hidden_size);
        for (int j = 0; j < hidden_size; j++)
        {
            // Задаем начальные значения весов случайным образом
            ih_weights[i][j] = 0.1;
        }
    }

    // Инициализируем веса между скрытым и выходным слоями
    ho_weights.resize(hidden_size);
    for (int i = 0; i < hidden_size; i++)
    {
        ho_weights[i].resize(output_size);
        for (int j = 0; j < output_size; j++)
        {
            // Задаем начальные значения весов случайным образом
            ho_weights[i][j] = 0.1;
        }
    }

    // Инициализируем смещения для скрытого слоя
    ih_biases.resize(hidden_size);
    for (int i = 0; i < hidden_size; i++)
    {
        // Задаем начальные значения смещений случайным образом
        ih_biases[i] = 0.1;
    }

    // Инициализируем смещения для выходного слоя
    ho_biases.resize(output_size);
    for (int i = 0; i < output_size; i++)
    {
        // Задаем начальные значения смещений случайным образом
        ho_biases[i] = 0.1;
    }
}

std::vector<double> NeuralNetwork::feedforward(const std::vector<double>& inputs) {
    std::vector<double> outputs = inputs;
    for (int i = 0; i < layers.size(); i++) {
        outputs = layers[i].feedforward(outputs);
    }
    return outputs;
}
//void NeuralNetwork::backpropagate(std::vector<double> input, std::vector<double> target) {
//    // 1. Вычисление выходного значения каждого нейрона в сети для заданного входного вектора.
//    std::vector<double> hidden_outputs(hidden_size);
//    for (int i = 0; i < hidden_size; i++) {
//        double activation = 0.0;
//        for (int j = 0; j < input_size; j++) {
//            activation += input[j] * ih_weights[j][i];
//        }
//        activation += ih_biases[i];
//        hidden_outputs[i] = sigmoid(activation);
//    }
//
//    std::vector<double> outputs(output_size);
//    for (int i = 0; i < output_size; i++) {
//        double activation = 0.0;
//        for (int j = 0; j < hidden_size; j++) {
//            activation += hidden_outputs[j] * ho_weights[j][i];
//        }
//        activation += ho_biases[i];
//        outputs[i] = sigmoid(activation);
//    }
//
//    // 2. Вычисление ошибки выходного слоя сети на основе ожидаемых выходных значений и фактических выходных значений.
//    std::vector<double> output_errors(output_size);
//    for (int i = 0; i < output_size; i++) {
//        output_errors[i] = (target[i] - outputs[i]) * sigmoid_derivative(outputs[i]);
//    }
//
//    // 3. Распространение ошибки на предыдущие слои сети, вычисление ошибки для каждого нейрона и корректировка весов связей.
//    std::vector<double> hidden_errors(hidden_size);
//    for (int i = 0; i < hidden_size; i++) {
//        double error = 0.0;
//        for (int j = 0; j < output_size; j++) {
//            error += output_errors[j] * ho_weights[i][j];
//        }
//        hidden_errors[i] = error * sigmoid_derivative(hidden_outputs[i]);
//    }
//
//    // корректировка весов связей между скрытым и выходным слоями
//    for (int i = 0; i < hidden_size; i++) {
//        for (int j = 0; j < output_size; j++) {
//            ho_weights[i][j] += learning_rate * output_errors[j] * hidden_outputs[i];
//        }
//        ho_biases[i] += learning_rate * output_errors[i];
//    }
//
//    // корректировка весов связей между входным и скрытым слоями
//    for (int i = 0; i < input_size; i++) {
//        for (int j = 0; j < hidden_size; j++) {
//            ih_weights[i][j] += learning_rate * hidden_errors[j] * input[i];
//        }
//        ih_biases[i] += learning_rate * hidden_errors[i];
//    }
//}
void NeuralNetwork::backpropagate(std::vector<double> input, std::vector<double> target)
{
    // Прямой проход
    std::vector<double> hidden_output(hidden_size);
    for (int i = 0; i < hidden_size; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < input_size; j++)
        {
            sum += input[j] * ih_weights[j][i];
        }
        // Добавляем смещение скрытого слоя
        sum += ih_biases[i];
        // Применяем функцию активации
        hidden_output[i] = sigmoid(sum);
    }

    std::vector<double> output(output_size);
    for (int i = 0; i < output_size; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < hidden_size; j++)
        {
            sum += hidden_output[j] * ho_weights[j][i];
        }
        // Добавляем смещение выходного слоя
        sum += ho_biases[i];
        // Применяем функцию активации
        output[i] = sigmoid(sum);
    }

    // Обратный проход
    std::vector<double> output_error(output_size);
    for (int i = 0; i < output_size; i++)
    {
        // Вычисляем ошибку для каждого выходного нейрона
        output_error[i] = output[i] * (1 - output[i]) * (target[i] - output[i]);
    }

    std::vector<double> hidden_error(hidden_size);
    for (int i = 0; i < hidden_size; i++)
    {
        double sum = 0.0;
        // Вычисляем суммарную ошибку для каждого скрытого нейрона
        for (int j = 0; j < output_size; j++)
        {
            sum += output_error[j] * ho_weights[i][j];
        }
        hidden_error[i] = hidden_output[i] * (1 - hidden_output[i]) * sum;
    }

    // Обновляем веса между скрытым и выходным слоями
    for (int i = 0; i < hidden_size; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            ho_weights[i][j] += learning_rate * output_error[j] * hidden_output[i];
        }
    }

    // Обновляем веса между входным и скрытым слоями
    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < hidden_size; j++)
        {
            ih_weights[i][j] += learning_rate * hidden_error[j] * input[i];
        }
    }

    // Обновляем смещения для выходного слоя
    for (int i = 0; i < output_size; i++)
    {
        ho_biases[i] += learning_rate * output_error[i];
    }

    // Обновляем смещения для скрытого слоя
    for (int i = 0; i < hidden_size; i++)
    {
        ih_biases[i] += learning_rate * hidden_error[i];
    }
}
// функция активации sigmoid
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// производная функции активации sigmoid
double NeuralNetwork::sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

void NeuralNetwork::save_model(const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        // Сохраняем параметры модели
        file << "input_size: " << input_size << "\n";
        file << "hidden_size: " << hidden_size << "\n";
        file << "output_size: " << output_size << "\n";
        file << "learning_rate: " << learning_rate << "\n";
        //file << "num_inputs_: " << num_inputs_ << "\n";

        // Сохраняем веса и смещения для каждого слоя
        file << "ih_weights:\n";
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                file << ih_weights[i][j] << " ";
            }
            file << "\n";
        }
        file << "ho_weights:\n";
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                file << ho_weights[i][j] << " ";
            }
            file << "\n";
        }
        file << "ih_biases:\n";
        for (int i = 0; i < hidden_size; i++) {
            file << ih_biases[i] << " ";
        }
        file << "\n";
        file << "ho_biases:\n";
        for (int i = 0; i < output_size; i++) {
            file << ho_biases[i] << " ";
        }
        file << "\n";

        // Сохраняем функцию активации
        file << "activation_function: sigmoid\n";

        file.close();
    }
    else {
        std::cerr << "Unable to save model to file " << filepath << "\n";
    }
}

//void NeuralNetwork::load_model(const std::string& filepath) {
//    std::ifstream file(filepath);
//    if (file.is_open()) {
//        std::string line;
//        while (std::getline(file, line)) {
//            std::istringstream iss(line);
//            std::string key;
//            if (std::getline(iss, key, ':')) {
//                if (key == "input_size") {
//                    iss >> input_size;
//                }
//                else if (key == "hidden_size") {
//                    iss >> hidden_size;
//                }
//                else if (key == "output_size") {
//                    iss >> output_size;
//                }
//                else if (key == "learning_rate") {
//                    iss >> learning_rate;
//                }
//                else if (key == "ih_weights") {
//                    ih_weights.resize(input_size, std::vector<double>(hidden_size));
//                    for (int i = 0; i < input_size; i++) {
//                        for (int j = 0; j < hidden_size; j++) {
//                            file >> ih_weights[i][j];
//                        }
//                    }
//                }
//                else if (key == "ho_weights") {
//                    ho_weights.resize(hidden_size, std::vector<double>(output_size));
//                    for (int i = 0; i < hidden_size; i++) {
//                        for (int j = 0; j < output_size; j++) {
//                            file >> ho_weights[i][j];
//                        }
//                    }
//                }
//                else if (key == "ih_biases") {
//                    ih_biases.resize(hidden_size);
//                    for (int i = 0; i < hidden_size; i++) {
//                        file >> ih_biases[i];
//                    }
//                }
//                else if (key == "ho_biases") {
//                    ho_biases.resize(output_size);
//                    for (int i = 0; i < output_size; i++) {
//                        file >> ho_biases[i];
//                    }
//                }
//                else if (key == "activation_function") {
//                    std::string activation_function;
//                    iss >> activation_function;
//                    if (activation_function != "sigmoid") {
//                        std::cerr << "Error: Unsupported activation function " << activation_function << "\n";
//                        exit(1);
//                    }
//                }
//            }
//        }
//        file.close();
//    }
//    else {
//        std::cerr << "Unableto load model from file " << filepath << "\n";
//    }
//
//    // Инициализируем слои и нейроны
//    input_layer_.resize(input_size);
//    hidden_layer_.resize(hidden_size);
//    output_layer_.resize(output_size);
//
//    for (int i = 0; i < input_size; i++) {
//        input_layer_[i] = new Neuron();
//    }
//
//    for (int i = 0; i < hidden_size; i++) {
//        hidden_layer_[i] = new Neuron(ih_biases[i], sigmoid);
//        for (int j = 0; j < input_size; j++) {
//            hidden_layer_[i]->add_input(input_layer_[j], ih_weights[j][i]);
//        }
//    }
//
//    for (int i = 0; i < output_size; i++) {
//        output_layer_[i] = new Neuron(ho_biases[i], sigmoid);
//        for (int j = 0; j < hidden_size; j++) {
//            output_layer_[i]->add_input(hidden_layer_[j], ho_weights[j][i]);
//        }
//    }
//
//    layers.push_back(Layer(input_layer_));
//    layers.push_back(Layer(hidden_layer_));
//    layers.push_back(Layer(output_layer_));
//}