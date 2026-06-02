#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class NeuralNetwork {
private:
    using Vector = std::vector<double>;
    using Matrix = std::vector<std::vector<double>>;

    std::size_t inputSize;
    std::size_t hidden1Size;
    std::size_t hidden2Size;
    std::size_t outputSize;
    double learningRate;
    Vector classWeights;

    Matrix weightsInputToHidden1;
    Vector biasHidden1;
    Matrix weightsHidden1ToHidden2;
    Vector biasHidden2;
    Matrix weightsHidden2ToOutput;
    Vector biasOutput;

    struct ForwardCache {
        Vector input;
        Vector z1;
        Vector a1;
        Vector z2;
        Vector a2;
        Vector logits;
        Vector probabilities;
    };

    static Matrix initializeWeightMatrix(std::size_t rows, std::size_t cols, std::mt19937& rng) {
        const double stddev = std::sqrt(2.0 / static_cast<double>(cols)); // He initialization.
        std::normal_distribution<double> dist(0.0, stddev);
        Matrix matrix(rows, Vector(cols, 0.0));
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                matrix[i][j] = dist(rng);
            }
        }
        return matrix;
    }

    static Vector affine(const Matrix& weights, const Vector& bias, const Vector& input) {
        if (weights.size() != bias.size()) {
            throw std::runtime_error("Weight/bias dimension mismatch in affine transform.");
        }

        Vector output(weights.size(), 0.0);
        for (std::size_t neuron = 0; neuron < weights.size(); ++neuron) {
            if (weights[neuron].size() != input.size()) {
                throw std::runtime_error("Input dimension mismatch in affine transform.");
            }
            double sum = bias[neuron];
            for (std::size_t j = 0; j < input.size(); ++j) {
                sum += weights[neuron][j] * input[j];
            }
            output[neuron] = sum;
        }
        return output;
    }

    static Vector relu(const Vector& input) {
        Vector output(input.size(), 0.0);
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::max(0.0, input[i]);
        }
        return output;
    }

    static Vector softmax(const Vector& logits) {
        if (logits.empty()) {
            throw std::runtime_error("Softmax input cannot be empty.");
        }

        constexpr double temperature = 1.5;
        if (temperature <= 0.0) {
            throw std::runtime_error("Softmax temperature must be positive.");
        }

        const double maxLogit = *std::max_element(logits.begin(), logits.end());
        Vector exps(logits.size(), 0.0);
        double sumExp = 0.0;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            exps[i] = std::exp((logits[i] - maxLogit) / temperature);
            sumExp += exps[i];
        }
        if (sumExp <= 0.0) {
            throw std::runtime_error("Invalid softmax denominator.");
        }

        Vector probabilities(logits.size(), 0.0);
        for (std::size_t i = 0; i < logits.size(); ++i) {
            probabilities[i] = exps[i] / sumExp;
        }
        return probabilities;
    }

    static void validateOneHotLabel(const Vector& oneHotLabel, std::size_t expectedSize) {
        if (oneHotLabel.size() != expectedSize) {
            throw std::runtime_error("Label dimension mismatch for cross-entropy.");
        }

        int oneCount = 0;
        for (double value : oneHotLabel) {
            if (value == 1.0) {
                ++oneCount;
            } else if (value != 0.0) {
                throw std::runtime_error("One-hot label must contain only 0 or 1.");
            }
        }
        if (oneCount != 1) {
            throw std::runtime_error("One-hot label must contain exactly one 1.");
        }
    }

    [[nodiscard]] Vector buildSmoothedLabel(const Vector& oneHotLabel) const {
        validateOneHotLabel(oneHotLabel, outputSize);
        const double smoothing = 0.1;
        const double base = smoothing / static_cast<double>(outputSize);
        Vector smoothed(outputSize, base);
        for (std::size_t i = 0; i < outputSize; ++i) {
            smoothed[i] += oneHotLabel[i] * (1.0 - smoothing);
        }
        return smoothed;
    }

    [[nodiscard]] ForwardCache forwardWithCache(const Vector& input) const {
        if (input.size() != inputSize) {
            throw std::runtime_error("NeuralNetwork input dimension mismatch.");
        }

        ForwardCache cache;
        cache.input = input;
        cache.z1 = affine(weightsInputToHidden1, biasHidden1, input);
        cache.a1 = relu(cache.z1);
        cache.z2 = affine(weightsHidden1ToHidden2, biasHidden2, cache.a1);
        cache.a2 = relu(cache.z2);
        cache.logits = affine(weightsHidden2ToOutput, biasOutput, cache.a2);
        cache.probabilities = softmax(cache.logits);
        return cache;
    }

    static Vector reluDerivative(const Vector& preActivation) {
        Vector gradient(preActivation.size(), 0.0);
        for (std::size_t i = 0; i < preActivation.size(); ++i) {
            gradient[i] = preActivation[i] > 0.0 ? 1.0 : 0.0;
        }
        return gradient;
    }

public:
    NeuralNetwork(
        std::size_t input = 6,
        std::size_t hidden1 = 12,
        std::size_t hidden2 = 8,
        std::size_t output = 4,
        double lr = 0.01,
        unsigned int randomSeed = 42
    )
        : inputSize(input),
          hidden1Size(hidden1),
          hidden2Size(hidden2),
          outputSize(output),
          learningRate(lr),
          classWeights(output, 1.0),
          weightsInputToHidden1(hidden1, Vector(input, 0.0)),
          biasHidden1(hidden1, 0.0),
          weightsHidden1ToHidden2(hidden2, Vector(hidden1, 0.0)),
          biasHidden2(hidden2, 0.0),
          weightsHidden2ToOutput(output, Vector(hidden2, 0.0)),
          biasOutput(output, 0.0) {
        if (inputSize == 0 || hidden1Size == 0 || hidden2Size == 0 || outputSize == 0) {
            throw std::runtime_error("Network layer sizes must be positive.");
        }
        if (learningRate <= 0.0) {
            throw std::runtime_error("Learning rate must be positive.");
        }

        std::mt19937 rng(randomSeed);
        weightsInputToHidden1 = initializeWeightMatrix(hidden1Size, inputSize, rng);
        weightsHidden1ToHidden2 = initializeWeightMatrix(hidden2Size, hidden1Size, rng);
        weightsHidden2ToOutput = initializeWeightMatrix(outputSize, hidden2Size, rng);

        if (outputSize == 4) {
            classWeights = {1.2, 1.0, 1.0, 1.0}; // clap, knock, noise, silence
        }
    }

    [[nodiscard]] Vector forward(const Vector& input) const {
        return forwardWithCache(input).probabilities;
    }

    [[nodiscard]] int predictClass(const Vector& input) const {
        const Vector probabilities = forward(input);
        const auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
        return static_cast<int>(std::distance(probabilities.begin(), maxIt));
    }

    [[nodiscard]] double getLearningRate() const {
        return learningRate;
    }

    [[nodiscard]] double categoricalCrossEntropy(const Vector& probabilities, const Vector& oneHotLabel) const {
        const Vector smoothedLabel = buildSmoothedLabel(oneHotLabel);
        if (probabilities.size() != outputSize) {
            throw std::runtime_error("Probability dimension mismatch for cross-entropy.");
        }

        constexpr double minProbability = 1e-9;
        double loss = 0.0;
        for (std::size_t i = 0; i < outputSize; ++i) {
            const double p = std::max(probabilities[i], minProbability);
            loss += -classWeights[i] * smoothedLabel[i] * std::log(p);
        }
        return loss;
    }

    [[nodiscard]] double computeSampleLoss(const Vector& input, const Vector& oneHotLabel) const {
        const Vector probabilities = forward(input);
        return categoricalCrossEntropy(probabilities, oneHotLabel);
    }

    [[nodiscard]] double computeBatchLoss(const Matrix& batchInputs, const Matrix& batchLabels) const {
        if (batchInputs.empty()) {
            throw std::runtime_error("Batch input cannot be empty.");
        }
        if (batchInputs.size() != batchLabels.size()) {
            throw std::runtime_error("Batch input/label count mismatch.");
        }

        double totalLoss = 0.0;
        for (std::size_t i = 0; i < batchInputs.size(); ++i) {
            totalLoss += computeSampleLoss(batchInputs[i], batchLabels[i]);
        }
        return totalLoss / static_cast<double>(batchInputs.size());
    }

    [[nodiscard]] double trainSingleSample(const Vector& input, const Vector& oneHotLabel) {
        const Vector smoothedLabel = buildSmoothedLabel(oneHotLabel);
        const ForwardCache cache = forwardWithCache(input);
        const double sampleLoss = categoricalCrossEntropy(cache.probabilities, oneHotLabel);

        Vector deltaOutput(outputSize, 0.0);
        double weightedLabelSum = 0.0;
        for (std::size_t i = 0; i < outputSize; ++i) {
            weightedLabelSum += classWeights[i] * smoothedLabel[i];
        }
        for (std::size_t i = 0; i < outputSize; ++i) {
            deltaOutput[i] = cache.probabilities[i] * weightedLabelSum - (classWeights[i] * smoothedLabel[i]);
        }

        Matrix gradW3(outputSize, Vector(hidden2Size, 0.0));
        Vector gradB3(outputSize, 0.0);
        constexpr double gradientClip = 5.0;
        for (std::size_t i = 0; i < outputSize; ++i) {
            gradB3[i] = std::clamp(deltaOutput[i], -gradientClip, gradientClip);
            for (std::size_t j = 0; j < hidden2Size; ++j) {
                gradW3[i][j] = std::clamp(deltaOutput[i] * cache.a2[j], -gradientClip, gradientClip);
            }
        }

        Vector deltaHidden2(hidden2Size, 0.0);
        const Vector reluGrad2 = reluDerivative(cache.z2);
        for (std::size_t j = 0; j < hidden2Size; ++j) {
            double propagated = 0.0;
            for (std::size_t i = 0; i < outputSize; ++i) {
                propagated += weightsHidden2ToOutput[i][j] * deltaOutput[i];
            }
            deltaHidden2[j] = propagated * reluGrad2[j];
        }

        Matrix gradW2(hidden2Size, Vector(hidden1Size, 0.0));
        Vector gradB2(hidden2Size, 0.0);
        for (std::size_t i = 0; i < hidden2Size; ++i) {
            gradB2[i] = std::clamp(deltaHidden2[i], -gradientClip, gradientClip);
            for (std::size_t j = 0; j < hidden1Size; ++j) {
                gradW2[i][j] = std::clamp(deltaHidden2[i] * cache.a1[j], -gradientClip, gradientClip);
            }
        }

        Vector deltaHidden1(hidden1Size, 0.0);
        const Vector reluGrad1 = reluDerivative(cache.z1);
        for (std::size_t j = 0; j < hidden1Size; ++j) {
            double propagated = 0.0;
            for (std::size_t i = 0; i < hidden2Size; ++i) {
                propagated += weightsHidden1ToHidden2[i][j] * deltaHidden2[i];
            }
            deltaHidden1[j] = propagated * reluGrad1[j];
        }

        Matrix gradW1(hidden1Size, Vector(inputSize, 0.0));
        Vector gradB1(hidden1Size, 0.0);
        for (std::size_t i = 0; i < hidden1Size; ++i) {
            gradB1[i] = std::clamp(deltaHidden1[i], -gradientClip, gradientClip);
            for (std::size_t j = 0; j < inputSize; ++j) {
                gradW1[i][j] = std::clamp(deltaHidden1[i] * cache.input[j], -gradientClip, gradientClip);
            }
        }

        for (std::size_t i = 0; i < outputSize; ++i) {
            biasOutput[i] -= learningRate * gradB3[i];
            for (std::size_t j = 0; j < hidden2Size; ++j) {
                weightsHidden2ToOutput[i][j] -= learningRate * gradW3[i][j];
            }
        }

        for (std::size_t i = 0; i < hidden2Size; ++i) {
            biasHidden2[i] -= learningRate * gradB2[i];
            for (std::size_t j = 0; j < hidden1Size; ++j) {
                weightsHidden1ToHidden2[i][j] -= learningRate * gradW2[i][j];
            }
        }

        for (std::size_t i = 0; i < hidden1Size; ++i) {
            biasHidden1[i] -= learningRate * gradB1[i];
            for (std::size_t j = 0; j < inputSize; ++j) {
                weightsInputToHidden1[i][j] -= learningRate * gradW1[i][j];
            }
        }

        return sampleLoss;
    }

    [[nodiscard]] double computeBatchAccuracy(const Matrix& batchInputs, const Matrix& batchLabels) const {
        if (batchInputs.empty()) {
            throw std::runtime_error("Batch input cannot be empty.");
        }
        if (batchInputs.size() != batchLabels.size()) {
            throw std::runtime_error("Batch input/label count mismatch.");
        }

        std::size_t correct = 0;
        for (std::size_t i = 0; i < batchInputs.size(); ++i) {
            validateOneHotLabel(batchLabels[i], outputSize);
            const int predictedClass = predictClass(batchInputs[i]);
            const auto targetIt = std::max_element(batchLabels[i].begin(), batchLabels[i].end());
            const int targetClass = static_cast<int>(std::distance(batchLabels[i].begin(), targetIt));
            if (predictedClass == targetClass) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / static_cast<double>(batchInputs.size());
    }

    void saveModel(const std::string& modelPath) const {
        std::filesystem::path path(modelPath);
        const auto parent = path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream out(modelPath, std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open model file for writing: " + modelPath);
        }

        out << inputSize << ' ' << hidden1Size << ' ' << hidden2Size << ' ' << outputSize << '\n';
        out << std::setprecision(17) << learningRate << '\n';

        for (const auto& row : weightsInputToHidden1) {
            for (std::size_t j = 0; j < row.size(); ++j) {
                out << row[j] << (j + 1 < row.size() ? ' ' : '\n');
            }
        }
        for (std::size_t i = 0; i < biasHidden1.size(); ++i) {
            out << biasHidden1[i] << (i + 1 < biasHidden1.size() ? ' ' : '\n');
        }

        for (const auto& row : weightsHidden1ToHidden2) {
            for (std::size_t j = 0; j < row.size(); ++j) {
                out << row[j] << (j + 1 < row.size() ? ' ' : '\n');
            }
        }
        for (std::size_t i = 0; i < biasHidden2.size(); ++i) {
            out << biasHidden2[i] << (i + 1 < biasHidden2.size() ? ' ' : '\n');
        }

        for (const auto& row : weightsHidden2ToOutput) {
            for (std::size_t j = 0; j < row.size(); ++j) {
                out << row[j] << (j + 1 < row.size() ? ' ' : '\n');
            }
        }
        for (std::size_t i = 0; i < biasOutput.size(); ++i) {
            out << biasOutput[i] << (i + 1 < biasOutput.size() ? ' ' : '\n');
        }

        if (!out.good()) {
            throw std::runtime_error("Failed while writing model file: " + modelPath);
        }
    }

    void loadModel(const std::string& modelPath) {
        std::ifstream in(modelPath);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open model file for reading: " + modelPath);
        }

        std::size_t fileInput = 0;
        std::size_t fileHidden1 = 0;
        std::size_t fileHidden2 = 0;
        std::size_t fileOutput = 0;
        double fileLearningRate = 0.0;
        in >> fileInput >> fileHidden1 >> fileHidden2 >> fileOutput;
        in >> fileLearningRate;
        if (!in.good()) {
            throw std::runtime_error("Model file header is corrupted: " + modelPath);
        }

        if (fileInput != inputSize || fileHidden1 != hidden1Size || fileHidden2 != hidden2Size || fileOutput != outputSize) {
            throw std::runtime_error("Model architecture mismatch for file: " + modelPath);
        }
        if (fileLearningRate <= 0.0) {
            throw std::runtime_error("Invalid learning rate in model file: " + modelPath);
        }

        for (std::size_t i = 0; i < hidden1Size; ++i) {
            for (std::size_t j = 0; j < inputSize; ++j) {
                in >> weightsInputToHidden1[i][j];
            }
        }
        for (std::size_t i = 0; i < hidden1Size; ++i) {
            in >> biasHidden1[i];
        }

        for (std::size_t i = 0; i < hidden2Size; ++i) {
            for (std::size_t j = 0; j < hidden1Size; ++j) {
                in >> weightsHidden1ToHidden2[i][j];
            }
        }
        for (std::size_t i = 0; i < hidden2Size; ++i) {
            in >> biasHidden2[i];
        }

        for (std::size_t i = 0; i < outputSize; ++i) {
            for (std::size_t j = 0; j < hidden2Size; ++j) {
                in >> weightsHidden2ToOutput[i][j];
            }
        }
        for (std::size_t i = 0; i < outputSize; ++i) {
            in >> biasOutput[i];
        }

        if (!in.good()) {
            throw std::runtime_error("Model file payload is corrupted: " + modelPath);
        }

        learningRate = fileLearningRate;
    }
};

int forwardNetworkSample(
    const std::vector<double>& input,
    std::vector<double>& outProbabilities,
    std::string& outErrorMessage
) {
    try {
        NeuralNetwork network(6, 12, 8, 4, 0.01, 42);
        outProbabilities = network.forward(input);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outProbabilities.clear();
        outErrorMessage = ex.what();
        return 1;
    }
}

int runNeuralNetworkDemo() {
    try {
        NeuralNetwork network(6, 12, 8, 4, 0.01, 42);
        const std::vector<double> demoInput = {0.1234000000, -0.0821000000, 0.3056000000, 0.1142000000, 0.2017000000, -0.0429000000};
        const std::vector<std::string> classNames = {"clap", "knock", "noise", "silence"};

        const std::vector<double> probabilities = network.forward(demoInput);
        const int predictedClass = network.predictClass(demoInput);

        std::cout << "NeuralNetwork architecture: 6 -> 12 (ReLU) -> 8 (ReLU) -> 4 (Softmax)\n";
        std::cout << "Learning rate: " << network.getLearningRate() << '\n';
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Demo input features (6): ";
        for (std::size_t i = 0; i < demoInput.size(); ++i) {
            std::cout << demoInput[i] << (i + 1 < demoInput.size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Softmax probabilities (4): ";
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::cout << probabilities[i] << (i + 1 < probabilities.size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Predicted class index: " << predictedClass << '\n';
        std::cout << "Predicted class name: " << classNames[static_cast<std::size_t>(predictedClass)] << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "NeuralNetwork error: " << ex.what() << '\n';
        return 1;
    }
}

int runForwardLossDemo() {
    try {
        NeuralNetwork network(6, 12, 8, 4, 0.01, 42);

        const std::vector<double> sampleInput = {0.1234000000, -0.0821000000, 0.3056000000, 0.1142000000, 0.2017000000, -0.0429000000};
        const std::vector<double> sampleLabel = {0.0, 0.0, 1.0, 0.0}; // noise
        const std::vector<double> sampleProbabilities = network.forward(sampleInput);
        const double sampleLoss = network.categoricalCrossEntropy(sampleProbabilities, sampleLabel);

        const std::vector<std::vector<double>> batchInputs = {
            {0.1234000000, -0.0821000000, 0.3056000000, 0.1142000000, 0.2017000000, -0.0429000000},
            {0.0221000000, 0.0367000000, 0.1489000000, 0.4501000000, 0.0854000000, -0.0125000000},
            {-0.2052000000, 0.0158000000, 0.0904000000, 0.0312000000, -0.1123000000, 0.2217000000}
        };
        const std::vector<std::vector<double>> batchLabels = {
            {0.0, 0.0, 1.0, 0.0}, // noise
            {1.0, 0.0, 0.0, 0.0}, // clap
            {0.0, 0.0, 0.0, 1.0}  // silence
        };

        const double batchLoss = network.computeBatchLoss(batchInputs, batchLabels);

        std::cout << "Step 7: Forward + Loss Math\n";
        std::cout << "Architecture: 6 -> 12 (ReLU) -> 8 (ReLU) -> 4 (Softmax)\n";
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Sample probabilities (4): ";
        for (std::size_t i = 0; i < sampleProbabilities.size(); ++i) {
            std::cout << sampleProbabilities[i] << (i + 1 < sampleProbabilities.size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Sample cross-entropy loss: " << sampleLoss << '\n';
        std::cout << "Mini-batch size: " << batchInputs.size() << '\n';
        std::cout << "Mini-batch average cross-entropy loss: " << batchLoss << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ForwardLoss error: " << ex.what() << '\n';
        return 1;
    }
}

int createNetworkHandle(
    std::size_t inputSize,
    std::size_t hidden1Size,
    std::size_t hidden2Size,
    std::size_t outputSize,
    double learningRate,
    unsigned int randomSeed,
    std::uintptr_t& outHandle,
    std::string& outErrorMessage
) {
    try {
        auto* network = new NeuralNetwork(inputSize, hidden1Size, hidden2Size, outputSize, learningRate, randomSeed);
        outHandle = reinterpret_cast<std::uintptr_t>(network);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outHandle = 0;
        outErrorMessage = ex.what();
        return 1;
    }
}

int destroyNetworkHandle(std::uintptr_t handle, std::string& outErrorMessage) {
    try {
        auto* network = reinterpret_cast<NeuralNetwork*>(handle);
        delete network;
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outErrorMessage = ex.what();
        return 1;
    }
}

int trainNetworkSample(
    std::uintptr_t handle,
    const std::vector<double>& input,
    const std::vector<double>& oneHotLabel,
    double& outLoss,
    std::string& outErrorMessage
) {
    try {
        auto* network = reinterpret_cast<NeuralNetwork*>(handle);
        if (network == nullptr) {
            throw std::runtime_error("Network handle is null.");
        }
        outLoss = network->trainSingleSample(input, oneHotLabel);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outLoss = 0.0;
        outErrorMessage = ex.what();
        return 1;
    }
}

int evaluateNetworkBatch(
    std::uintptr_t handle,
    const std::vector<std::vector<double>>& batchInputs,
    const std::vector<std::vector<double>>& batchLabels,
    double& outLoss,
    double& outAccuracy,
    std::string& outErrorMessage
) {
    try {
        auto* network = reinterpret_cast<NeuralNetwork*>(handle);
        if (network == nullptr) {
            throw std::runtime_error("Network handle is null.");
        }
        outLoss = network->computeBatchLoss(batchInputs, batchLabels);
        outAccuracy = network->computeBatchAccuracy(batchInputs, batchLabels);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outLoss = 0.0;
        outAccuracy = 0.0;
        outErrorMessage = ex.what();
        return 1;
    }
}

int saveNetworkModel(
    std::uintptr_t handle,
    const std::string& modelPath,
    std::string& outErrorMessage
) {
    try {
        auto* network = reinterpret_cast<NeuralNetwork*>(handle);
        if (network == nullptr) {
            throw std::runtime_error("Network handle is null.");
        }
        network->saveModel(modelPath);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outErrorMessage = ex.what();
        return 1;
    }
}

int loadNetworkModel(
    std::uintptr_t handle,
    const std::string& modelPath,
    std::string& outErrorMessage
) {
    try {
        auto* network = reinterpret_cast<NeuralNetwork*>(handle);
        if (network == nullptr) {
            throw std::runtime_error("Network handle is null.");
        }
        network->loadModel(modelPath);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outErrorMessage = ex.what();
        return 1;
    }
}

int predictNetworkClass(
    std::uintptr_t handle,
    const std::vector<double>& input,
    int& outClassIndex,
    std::string& outErrorMessage
) {
    try {
        auto* network = reinterpret_cast<NeuralNetwork*>(handle);
        if (network == nullptr) {
            throw std::runtime_error("Network handle is null.");
        }
        outClassIndex = network->predictClass(input);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outClassIndex = -1;
        outErrorMessage = ex.what();
        return 1;
    }
}

int predictNetworkProbabilities(
    std::uintptr_t handle,
    const std::vector<double>& input,
    std::vector<double>& outProbabilities,
    std::string& outErrorMessage
) {
    try {
        auto* network = reinterpret_cast<NeuralNetwork*>(handle);
        if (network == nullptr) {
            throw std::runtime_error("Network handle is null.");
        }
        outProbabilities = network->forward(input);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outProbabilities.clear();
        outErrorMessage = ex.what();
        return 1;
    }
}
