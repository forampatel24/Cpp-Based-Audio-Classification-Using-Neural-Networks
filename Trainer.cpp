#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

int loadAudioFileStrict(
    const std::string& filePath,
    std::vector<double>& outSamples,
    std::string& outErrorMessage
);

int extractFeatureVector(
    const std::vector<double>& samples,
    std::vector<double>& outFeatures,
    std::string& outErrorMessage
);

int buildNormalizedDatasetSplits(
    std::vector<std::vector<double>>& outTrainFeatures,
    std::vector<std::vector<double>>& outTrainLabels,
    std::vector<std::vector<double>>& outValFeatures,
    std::vector<std::vector<double>>& outValLabels,
    std::vector<std::vector<double>>& outTestFeatures,
    std::vector<std::vector<double>>& outTestLabels,
    std::string& outErrorMessage
);

int createNetworkHandle(
    std::size_t inputSize,
    std::size_t hidden1Size,
    std::size_t hidden2Size,
    std::size_t outputSize,
    double learningRate,
    unsigned int randomSeed,
    std::uintptr_t& outHandle,
    std::string& outErrorMessage
);

int destroyNetworkHandle(std::uintptr_t handle, std::string& outErrorMessage);

int trainNetworkSample(
    std::uintptr_t handle,
    const std::vector<double>& input,
    const std::vector<double>& oneHotLabel,
    double& outLoss,
    std::string& outErrorMessage
);

int evaluateNetworkBatch(
    std::uintptr_t handle,
    const std::vector<std::vector<double>>& batchInputs,
    const std::vector<std::vector<double>>& batchLabels,
    double& outLoss,
    double& outAccuracy,
    std::string& outErrorMessage
);

int predictNetworkClass(
    std::uintptr_t handle,
    const std::vector<double>& input,
    int& outClassIndex,
    std::string& outErrorMessage
);

int saveNetworkModel(
    std::uintptr_t handle,
    const std::string& modelPath,
    std::string& outErrorMessage
);

struct DatasetSample {
    std::vector<double> features;
    std::vector<double> oneHotLabel;
    std::string className;
    std::string filePath;
};

class Trainer {
private:
    std::vector<std::string> classOrder;

    static std::filesystem::path resolveDataRoot(const std::string& requestedDataPath) {
        const std::filesystem::path inputPath(requestedDataPath);
        if (std::filesystem::exists(inputPath) && std::filesystem::is_directory(inputPath)) {
            return std::filesystem::weakly_canonical(inputPath);
        }

        const std::filesystem::path cwd = std::filesystem::current_path();
        std::vector<std::filesystem::path> candidates = {
            cwd / inputPath,
            cwd / ".." / inputPath,
            cwd / ".." / ".." / inputPath,
            cwd / ".." / ".." / ".." / inputPath
        };

        for (const auto& candidate : candidates) {
            if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
                return std::filesystem::weakly_canonical(candidate);
            }
        }

        throw std::runtime_error("Unable to locate data directory: " + requestedDataPath);
    }

    static bool isWavFile(const std::filesystem::path& path) {
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return ext == ".wav";
    }

    [[nodiscard]] std::vector<double> buildOneHotLabel(std::size_t classIndex) const {
        std::vector<double> label(classOrder.size(), 0.0);
        label[classIndex] = 1.0;
        return label;
    }

public:
    Trainer() : classOrder({"clap", "knock", "noise", "silence"}) {}

    std::vector<DatasetSample> buildDataset(const std::string& requestedDataPath = "data") const {
        const std::filesystem::path dataRoot = resolveDataRoot(requestedDataPath);
        std::vector<DatasetSample> dataset;

        for (std::size_t classIndex = 0; classIndex < classOrder.size(); ++classIndex) {
            const std::filesystem::path classDir = dataRoot / classOrder[classIndex];
            if (!std::filesystem::exists(classDir) || !std::filesystem::is_directory(classDir)) {
                throw std::runtime_error("Missing class folder: " + classDir.string());
            }

            std::vector<std::filesystem::path> files;
            for (const auto& entry : std::filesystem::directory_iterator(classDir)) {
                if (entry.is_regular_file() && isWavFile(entry.path())) {
                    files.push_back(entry.path());
                }
            }
            std::sort(files.begin(), files.end());

            for (const auto& wavPath : files) {
                std::vector<double> samples;
                std::string loadError;
                if (loadAudioFileStrict(wavPath.string(), samples, loadError) != 0) {
                    continue;
                }

                std::vector<double> features;
                std::string featureError;
                if (extractFeatureVector(samples, features, featureError) != 0) {
                    continue;
                }

                dataset.push_back({features, buildOneHotLabel(classIndex), classOrder[classIndex], wavPath.string()});
            }
        }

        if (dataset.empty()) {
            throw std::runtime_error("No valid dataset samples could be built.");
        }

        return dataset;
    }
};

int runDatasetBuilderDemo() {
    try {
        Trainer trainer;
        const std::vector<DatasetSample> dataset = trainer.buildDataset("data");

        std::cout << "Dataset size: " << dataset.size() << '\n';
        const DatasetSample& sample = dataset.back(); // Change this index to inspect a different sample.
        std::cout << "Sample file: " << sample.filePath << '\n';
        std::cout << "Sample class: " << sample.className << '\n';
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Sample features (6): ";
        for (std::size_t i = 0; i < sample.features.size(); ++i) {
            std::cout << sample.features[i] << (i + 1 < sample.features.size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Sample one-hot label (4): ";
        for (std::size_t i = 0; i < sample.oneHotLabel.size(); ++i) {
            std::cout << sample.oneHotLabel[i] << (i + 1 < sample.oneHotLabel.size() ? " " : "");
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "DatasetBuilder error: " << ex.what() << '\n';
        return 1;
    }
}

int buildEncodedDataset(
    std::vector<std::vector<double>>& outFeatures,
    std::vector<std::vector<double>>& outLabels,
    std::vector<std::string>& outClassNames,
    std::vector<std::string>& outFilePaths,
    std::string& outErrorMessage
) {
    try {
        Trainer trainer;
        const std::vector<DatasetSample> dataset = trainer.buildDataset("data");

        outFeatures.clear();
        outLabels.clear();
        outClassNames.clear();
        outFilePaths.clear();

        outFeatures.reserve(dataset.size());
        outLabels.reserve(dataset.size());
        outClassNames.reserve(dataset.size());
        outFilePaths.reserve(dataset.size());

        for (const auto& sample : dataset) {
            outFeatures.push_back(sample.features);
            outLabels.push_back(sample.oneHotLabel);
            outClassNames.push_back(sample.className);
            outFilePaths.push_back(sample.filePath);
        }

        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outFeatures.clear();
        outLabels.clear();
        outClassNames.clear();
        outFilePaths.clear();
        outErrorMessage = ex.what();
        return 1;
    }
}

static std::filesystem::path resolveModelsRoot() {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<std::filesystem::path> candidates = {
        cwd / "models",
        cwd / ".." / "models",
        cwd / ".." / ".." / "models",
        cwd / ".." / ".." / ".." / "models"
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
            return std::filesystem::weakly_canonical(candidate);
        }
    }

    const std::filesystem::path fallback = cwd / "models";
    std::filesystem::create_directories(fallback);
    return std::filesystem::weakly_canonical(fallback);
}

int runTrainingLoopDemo() {
    std::uintptr_t networkHandle = 0;
    try {
        std::vector<std::vector<double>> trainFeatures;
        std::vector<std::vector<double>> trainLabels;
        std::vector<std::vector<double>> valFeatures;
        std::vector<std::vector<double>> valLabels;
        std::vector<std::vector<double>> testFeatures;
        std::vector<std::vector<double>> testLabels;
        std::string errorMessage;

        if (buildNormalizedDatasetSplits(
                trainFeatures, trainLabels,
                valFeatures, valLabels,
                testFeatures, testLabels,
                errorMessage) != 0) {
            std::cerr << "Trainer error: " << errorMessage << '\n';
            return 1;
        }

        constexpr std::size_t inputSize = 6;
        constexpr std::size_t hidden1Size = 12;
        constexpr std::size_t hidden2Size = 8;
        constexpr std::size_t outputSize = 4;
        constexpr double learningRate = 0.01;
        constexpr unsigned int randomSeed = 42;
        constexpr int epochs = 1000;

        if (createNetworkHandle(
                inputSize,
                hidden1Size,
                hidden2Size,
                outputSize,
                learningRate,
                randomSeed,
                networkHandle,
                errorMessage) != 0) {
            std::cerr << "Trainer error: " << errorMessage << '\n';
            return 1;
        }

        std::vector<std::size_t> trainIndices(trainFeatures.size());
        for (std::size_t i = 0; i < trainIndices.size(); ++i) {
            trainIndices[i] = i;
        }

        std::mt19937 rng(1337);
        double bestValLoss = std::numeric_limits<double>::infinity();
        int bestEpoch = -1;
        const std::filesystem::path modelPath = resolveModelsRoot() / "nn_step8_best_model.txt";

        std::cout << "Step 8: Backpropagation + Training Loop\n";
        std::cout << "Architecture: 6 -> 12 (ReLU) -> 8 (ReLU) -> 4 (Softmax)\n";
        std::cout << "Learning rate: " << learningRate << ", epochs: " << epochs << '\n';
        std::cout << "Split sizes -> train: " << trainFeatures.size()
                  << ", val: " << valFeatures.size()
                  << ", test: " << testFeatures.size() << '\n';

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::shuffle(trainIndices.begin(), trainIndices.end(), rng);
            double epochTrainLoss = 0.0;
            std::vector<int> classCorrect(outputSize, 0);
            std::vector<int> classTotal(outputSize, 0);

            for (std::size_t idx : trainIndices) {
                double sampleLoss = 0.0;
                if (trainNetworkSample(networkHandle, trainFeatures[idx], trainLabels[idx], sampleLoss, errorMessage) != 0) {
                    throw std::runtime_error(errorMessage);
                }
                epochTrainLoss += sampleLoss;

                const auto targetIt = std::max_element(trainLabels[idx].begin(), trainLabels[idx].end());
                const int targetClass = static_cast<int>(std::distance(trainLabels[idx].begin(), targetIt));
                if (targetClass < 0 || static_cast<std::size_t>(targetClass) >= outputSize) {
                    throw std::runtime_error("Invalid target class index while tracking class-wise accuracy.");
                }
                classTotal[static_cast<std::size_t>(targetClass)] += 1;

                int predictedClass = -1;
                if (predictNetworkClass(networkHandle, trainFeatures[idx], predictedClass, errorMessage) != 0) {
                    throw std::runtime_error(errorMessage);
                }
                if (predictedClass == targetClass) {
                    classCorrect[static_cast<std::size_t>(targetClass)] += 1;
                }
            }
            epochTrainLoss /= static_cast<double>(trainFeatures.size());

            double valLoss = 0.0;
            double valAccuracy = 0.0;
            if (evaluateNetworkBatch(networkHandle, valFeatures, valLabels, valLoss, valAccuracy, errorMessage) != 0) {
                throw std::runtime_error(errorMessage);
            }

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                bestEpoch = epoch;
                if (saveNetworkModel(networkHandle, modelPath.string(), errorMessage) != 0) {
                    throw std::runtime_error(errorMessage);
                }
            }

            if (epoch == 1 || epoch % 5 == 0 || epoch == epochs) {
                std::cout << std::fixed << std::setprecision(6)
                          << "Epoch " << epoch
                          << " | train_loss: " << epochTrainLoss
                          << " | val_loss: " << valLoss
                          << " | val_acc: " << valAccuracy << '\n';
                const double clapAcc = classTotal[0] > 0
                    ? static_cast<double>(classCorrect[0]) / static_cast<double>(classTotal[0])
                    : 0.0;
                const double knockAcc = classTotal[1] > 0
                    ? static_cast<double>(classCorrect[1]) / static_cast<double>(classTotal[1])
                    : 0.0;
                std::cout << "  clap_acc: " << clapAcc
                          << " | knock_acc: " << knockAcc << '\n';
            }
        }

        double testLoss = 0.0;
        double testAccuracy = 0.0;
        if (evaluateNetworkBatch(networkHandle, testFeatures, testLabels, testLoss, testAccuracy, errorMessage) != 0) {
            throw std::runtime_error(errorMessage);
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Best validation epoch: " << bestEpoch << '\n';
        std::cout << "Best model saved to: " << modelPath.string() << '\n';
        std::cout << "Test loss: " << testLoss << '\n';
        std::cout << "Test accuracy: " << testAccuracy << '\n';

        std::string destroyError;
        if (destroyNetworkHandle(networkHandle, destroyError) != 0) {
            std::cerr << "Trainer warning: " << destroyError << '\n';
        }
        return 0;
    } catch (const std::exception& ex) {
        if (networkHandle != 0) {
            std::string destroyError;
            destroyNetworkHandle(networkHandle, destroyError);
        }
        std::cerr << "Trainer error: " << ex.what() << '\n';
        return 1;
    }
}
