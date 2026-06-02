#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

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

int evaluateNetworkBatch(
    std::uintptr_t handle,
    const std::vector<std::vector<double>>& batchInputs,
    const std::vector<std::vector<double>>& batchLabels,
    double& outLoss,
    double& outAccuracy,
    std::string& outErrorMessage
);

int loadNetworkModel(
    std::uintptr_t handle,
    const std::string& modelPath,
    std::string& outErrorMessage
);

int predictNetworkClass(
    std::uintptr_t handle,
    const std::vector<double>& input,
    int& outClassIndex,
    std::string& outErrorMessage
);

int predictNetworkProbabilities(
    std::uintptr_t handle,
    const std::vector<double>& input,
    std::vector<double>& outProbabilities,
    std::string& outErrorMessage
);

int loadValidatedAudioSamples(
    const std::string& requestedPath,
    std::vector<double>& outSamples,
    std::string& outSelectedPath,
    std::string& outErrorMessage
);

int extractFeatureVector(
    const std::vector<double>& samples,
    std::vector<double>& outFeatures,
    std::string& outErrorMessage
);

int normalizeFeatureForInference(
    const std::vector<double>& inputFeature,
    std::vector<double>& outNormalizedFeature,
    std::string& outErrorMessage
);

class Evaluator {
private:
    std::vector<std::string> classNames;

    static std::filesystem::path resolveDirectory(const std::string& directoryName, bool createIfMissing) {
        const std::filesystem::path cwd = std::filesystem::current_path();
        std::vector<std::filesystem::path> candidates = {
            cwd / directoryName,
            cwd / ".." / directoryName,
            cwd / ".." / ".." / directoryName,
            cwd / ".." / ".." / ".." / directoryName
        };

        for (const auto& candidate : candidates) {
            if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
                return std::filesystem::weakly_canonical(candidate);
            }
        }

        if (createIfMissing) {
            std::filesystem::create_directories(candidates.front());
            return std::filesystem::weakly_canonical(candidates.front());
        }

        throw std::runtime_error("Unable to locate required directory: " + directoryName);
    }

    static int labelToClassIndex(const std::vector<double>& oneHotLabel) {
        if (oneHotLabel.empty()) {
            throw std::runtime_error("Empty one-hot label.");
        }
        const auto maxIt = std::max_element(oneHotLabel.begin(), oneHotLabel.end());
        return static_cast<int>(std::distance(oneHotLabel.begin(), maxIt));
    }

public:
    Evaluator() : classNames({"clap", "knock", "noise", "silence"}) {}

    int runStep9Evaluation() const {
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
                throw std::runtime_error(errorMessage);
            }

            if (createNetworkHandle(6, 12, 8, 4, 0.01, 42, networkHandle, errorMessage) != 0) {
                throw std::runtime_error(errorMessage);
            }

            const std::filesystem::path modelPath = resolveDirectory("models", false) / "nn_step8_best_model.txt";
            if (loadNetworkModel(networkHandle, modelPath.string(), errorMessage) != 0) {
                throw std::runtime_error(errorMessage);
            }

            double valLoss = 0.0;
            double valAccuracy = 0.0;
            if (evaluateNetworkBatch(networkHandle, valFeatures, valLabels, valLoss, valAccuracy, errorMessage) != 0) {
                throw std::runtime_error(errorMessage);
            }

            double testLoss = 0.0;
            double testAccuracy = 0.0;
            if (evaluateNetworkBatch(networkHandle, testFeatures, testLabels, testLoss, testAccuracy, errorMessage) != 0) {
                throw std::runtime_error(errorMessage);
            }

            std::vector<std::vector<int>> confusionMatrix(classNames.size(), std::vector<int>(classNames.size(), 0));
            for (std::size_t i = 0; i < testFeatures.size(); ++i) {
                int predictedClass = -1;
                if (predictNetworkClass(networkHandle, testFeatures[i], predictedClass, errorMessage) != 0) {
                    throw std::runtime_error(errorMessage);
                }
                const int trueClass = labelToClassIndex(testLabels[i]);
                confusionMatrix[static_cast<std::size_t>(trueClass)][static_cast<std::size_t>(predictedClass)] += 1;
            }

            struct ClassMetric {
                double precision;
                double recall;
                double f1;
                int support;
            };
            std::vector<ClassMetric> metrics;
            metrics.reserve(classNames.size());

            for (std::size_t c = 0; c < classNames.size(); ++c) {
                const int tp = confusionMatrix[c][c];
                int fp = 0;
                int fn = 0;
                for (std::size_t r = 0; r < classNames.size(); ++r) {
                    if (r != c) {
                        fp += confusionMatrix[r][c];
                    }
                }
                for (std::size_t j = 0; j < classNames.size(); ++j) {
                    if (j != c) {
                        fn += confusionMatrix[c][j];
                    }
                }
                const int support = std::accumulate(confusionMatrix[c].begin(), confusionMatrix[c].end(), 0);
                const double precision = (tp + fp) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
                const double recall = (tp + fn) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
                const double f1 = (precision + recall) > 0.0 ? (2.0 * precision * recall) / (precision + recall) : 0.0;
                metrics.push_back({precision, recall, f1, support});
            }

            double macroF1 = 0.0;
            for (const auto& metric : metrics) {
                macroF1 += metric.f1;
            }
            macroF1 /= static_cast<double>(metrics.size());

            const std::filesystem::path reportPath = resolveDirectory("results", true) / "step9_evaluation_report.txt";
            std::ofstream report(reportPath.string(), std::ios::trunc);
            if (!report.is_open()) {
                throw std::runtime_error("Failed to write evaluation report: " + reportPath.string());
            }

            report << std::fixed << std::setprecision(6);
            report << "Step 9 Evaluation Report\n";
            report << "Model: " << modelPath.string() << '\n';
            report << "Validation -> loss: " << valLoss << ", accuracy: " << valAccuracy << '\n';
            report << "Test -> loss: " << testLoss << ", accuracy: " << testAccuracy << '\n';
            report << "Macro F1: " << macroF1 << "\n\n";
            report << "Confusion Matrix (rows=true, cols=pred):\n";
            report << "          clap   knock  noise  silence\n";
            for (std::size_t i = 0; i < classNames.size(); ++i) {
                report << std::setw(8) << classNames[i] << " ";
                for (std::size_t j = 0; j < classNames.size(); ++j) {
                    report << std::setw(6) << confusionMatrix[i][j];
                }
                report << '\n';
            }
            report << "\nPer-class metrics:\n";
            report << "class     precision  recall  f1-score  support\n";
            for (std::size_t i = 0; i < classNames.size(); ++i) {
                report << std::setw(8) << classNames[i] << "  "
                       << std::setw(9) << metrics[i].precision << "  "
                       << std::setw(6) << metrics[i].recall << "  "
                       << std::setw(8) << metrics[i].f1 << "  "
                       << std::setw(7) << metrics[i].support << '\n';
            }

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Step 9: Evaluation Module\n";
            std::cout << "Loaded model: " << modelPath.string() << '\n';
            std::cout << "Validation -> loss: " << valLoss << ", accuracy: " << valAccuracy << '\n';
            std::cout << "Test -> loss: " << testLoss << ", accuracy: " << testAccuracy << '\n';
            std::cout << "Macro F1: " << macroF1 << '\n';
            std::cout << "Evaluation report saved to: " << reportPath.string() << '\n';

            std::string destroyError;
            if (destroyNetworkHandle(networkHandle, destroyError) != 0) {
                std::cerr << "Evaluator warning: " << destroyError << '\n';
            }
            return 0;
        } catch (const std::exception& ex) {
            if (networkHandle != 0) {
                std::string destroyError;
                destroyNetworkHandle(networkHandle, destroyError);
            }
            std::cerr << "Evaluator error: " << ex.what() << '\n';
            return 1;
        }
    }
};

int runEvaluatorDemo() {
    Evaluator evaluator;
    return evaluator.runStep9Evaluation();
}

static std::filesystem::path resolveDirectoryForStep10(const std::string& directoryName, bool createIfMissing) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<std::filesystem::path> candidates = {
        cwd / directoryName,
        cwd / ".." / directoryName,
        cwd / ".." / ".." / directoryName,
        cwd / ".." / ".." / ".." / directoryName
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
            return std::filesystem::weakly_canonical(candidate);
        }
    }

    if (createIfMissing) {
        std::filesystem::create_directories(candidates.front());
        return std::filesystem::weakly_canonical(candidates.front());
    }

    throw std::runtime_error("Unable to locate required directory: " + directoryName);
}

int runEndToEndInferenceDemo(const std::string& requestedPath) {
    std::uintptr_t networkHandle = 0;
    try {
        std::vector<double> samples;
        std::string selectedPath;
        std::string errorMessage;
        if (loadValidatedAudioSamples(requestedPath, samples, selectedPath, errorMessage) != 0) {
            throw std::runtime_error(errorMessage);
        }

        std::vector<double> extractedFeatures;
        if (extractFeatureVector(samples, extractedFeatures, errorMessage) != 0) {
            throw std::runtime_error(errorMessage);
        }

        std::vector<double> normalizedFeatures;
        if (normalizeFeatureForInference(extractedFeatures, normalizedFeatures, errorMessage) != 0) {
            throw std::runtime_error(errorMessage);
        }

        if (createNetworkHandle(6, 12, 8, 4, 0.01, 42, networkHandle, errorMessage) != 0) {
            throw std::runtime_error(errorMessage);
        }

        const std::filesystem::path modelPath = resolveDirectoryForStep10("models", false) / "nn_step8_best_model.txt";
        if (loadNetworkModel(networkHandle, modelPath.string(), errorMessage) != 0) {
            throw std::runtime_error(errorMessage);
        }

        std::vector<double> probabilities;
        if (predictNetworkProbabilities(networkHandle, normalizedFeatures, probabilities, errorMessage) != 0) {
            throw std::runtime_error(errorMessage);
        }

        int predictedClass = -1;
        if (predictNetworkClass(networkHandle, normalizedFeatures, predictedClass, errorMessage) != 0) {
            throw std::runtime_error(errorMessage);
        }

        const std::vector<std::string> classNames = {"clap", "knock", "noise", "silence"};
        const std::filesystem::path reportPath = resolveDirectoryForStep10("results", true) / "step10_inference_report.txt";
        std::ofstream report(reportPath.string(), std::ios::trunc);
        if (!report.is_open()) {
            throw std::runtime_error("Failed to write inference report: " + reportPath.string());
        }

        report << std::fixed << std::setprecision(6);
        report << "Step 10 Inference Report\n";
        report << "Input WAV: " << selectedPath << '\n';
        report << "Model: " << modelPath.string() << '\n';
        report << "Predicted class index: " << predictedClass << '\n';
        report << "Predicted class name: " << classNames[static_cast<std::size_t>(predictedClass)] << '\n';
        report << "Class probabilities:\n";
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            report << "  " << classNames[i] << ": " << probabilities[i] << '\n';
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Step 10: End-to-End Inference\n";
        std::cout << "Input WAV: " << selectedPath << '\n';
        std::cout << "Model: " << modelPath.string() << '\n';
        std::cout << "Predicted class index: " << predictedClass << '\n';
        std::cout << "Predicted class name: " << classNames[static_cast<std::size_t>(predictedClass)] << '\n';
        std::cout << "Class probabilities: ";
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::cout << probabilities[i] << (i + 1 < probabilities.size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Inference report saved to: " << reportPath.string() << '\n';

        std::string destroyError;
        if (destroyNetworkHandle(networkHandle, destroyError) != 0) {
            std::cerr << "Inference warning: " << destroyError << '\n';
        }
        return 0;
    } catch (const std::exception& ex) {
        if (networkHandle != 0) {
            std::string destroyError;
            destroyNetworkHandle(networkHandle, destroyError);
        }
        std::cerr << "Inference error: " << ex.what() << '\n';
        return 1;
    }
}
