#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int buildEncodedDataset(
    std::vector<std::vector<double>>& outFeatures,
    std::vector<std::vector<double>>& outLabels,
    std::vector<std::string>& outClassNames,
    std::vector<std::string>& outFilePaths,
    std::string& outErrorMessage
);

class Preprocessor {
private:
    std::vector<double> mean;
    std::vector<double> stddev;
    bool fitted;
    double epsilon;
    double minStdDev;

    [[nodiscard]] std::vector<double> applyFeatureTransform(const std::vector<double>& featureVector) const {
        std::vector<double> transformed = featureVector;
        if (transformed.size() <= 5) {
            throw std::runtime_error("Expected feature vector size >= 6 for frame variance transform.");
        }
        // Apply log transform only to frame_variance (index 5) to reduce range explosion.
        transformed[5] = std::log(1.0 + std::max(0.0, transformed[5]));
        return transformed;
    }

public:
    Preprocessor() : fitted(false), epsilon(1e-8), minStdDev(1e-6) {}

    void fit(const std::vector<std::vector<double>>& trainingFeatures) {
        if (trainingFeatures.empty()) {
            throw std::runtime_error("Cannot fit preprocessor on empty training set.");
        }

        const std::size_t featureDim = trainingFeatures.front().size();
        if (featureDim == 0) {
            throw std::runtime_error("Feature dimension cannot be zero.");
        }

        for (const auto& row : trainingFeatures) {
            if (row.size() != featureDim) {
                throw std::runtime_error("Inconsistent feature vector size in training set.");
            }
        }

        mean.assign(featureDim, 0.0);
        stddev.assign(featureDim, 0.0);

        const double n = static_cast<double>(trainingFeatures.size());
        for (const auto& row : trainingFeatures) {
            const std::vector<double> transformedRow = applyFeatureTransform(row);
            for (std::size_t j = 0; j < featureDim; ++j) {
                mean[j] += transformedRow[j];
            }
        }
        for (double& m : mean) {
            m /= n;
        }

        for (const auto& row : trainingFeatures) {
            const std::vector<double> transformedRow = applyFeatureTransform(row);
            for (std::size_t j = 0; j < featureDim; ++j) {
                const double diff = transformedRow[j] - mean[j];
                stddev[j] += diff * diff;
            }
        }
        for (double& s : stddev) {
            s = std::sqrt(s / n);
            s = std::max(s, minStdDev); // Numerical stability clamp.
        }

        fitted = true;
    }

    std::vector<double> transform(const std::vector<double>& featureVector) const {
        if (!fitted) {
            throw std::runtime_error("Preprocessor is not fitted.");
        }
        if (featureVector.size() != mean.size()) {
            throw std::runtime_error("Feature dimension mismatch in transform.");
        }

        const std::vector<double> transformed = applyFeatureTransform(featureVector);
        std::vector<double> normalized(featureVector.size(), 0.0);
        for (std::size_t j = 0; j < featureVector.size(); ++j) {
            normalized[j] = (transformed[j] - mean[j]) / (stddev[j] + epsilon);
        }
        return normalized;
    }

    std::vector<std::vector<double>> transformDataset(const std::vector<std::vector<double>>& features) const {
        std::vector<std::vector<double>> normalized;
        normalized.reserve(features.size());
        for (const auto& row : features) {
            normalized.push_back(transform(row));
        }
        return normalized;
    }
};

static std::vector<double> computeFeatureMeans(const std::vector<std::vector<double>>& features) {
    const std::size_t dim = features.front().size();
    std::vector<double> means(dim, 0.0);
    const double n = static_cast<double>(features.size());
    for (const auto& row : features) {
        for (std::size_t j = 0; j < dim; ++j) {
            means[j] += row[j];
        }
    }
    for (double& value : means) {
        value /= n;
    }
    return means;
}

static std::vector<double> computeFeatureStdDev(const std::vector<std::vector<double>>& features,
                                                const std::vector<double>& means) {
    const std::size_t dim = features.front().size();
    std::vector<double> stds(dim, 0.0);
    const double n = static_cast<double>(features.size());
    for (const auto& row : features) {
        for (std::size_t j = 0; j < dim; ++j) {
            const double diff = row[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (double& value : stds) {
        value = std::sqrt(value / n);
    }
    return stds;
}

static void buildNormalizedSplitsInternal(
    std::vector<std::vector<double>>& outTrainFeatures,
    std::vector<std::vector<double>>& outTrainLabels,
    std::vector<std::vector<double>>& outValFeatures,
    std::vector<std::vector<double>>& outValLabels,
    std::vector<std::vector<double>>& outTestFeatures,
    std::vector<std::vector<double>>& outTestLabels,
    Preprocessor* outFittedPreprocessor = nullptr
) {
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;
    std::vector<std::string> classes;
    std::vector<std::string> filePaths;
    std::string datasetError;

    if (buildEncodedDataset(features, labels, classes, filePaths, datasetError) != 0) {
        throw std::runtime_error(datasetError);
    }

    const std::size_t n = features.size();
    if (n == 0 || labels.size() != n) {
        throw std::runtime_error("Dataset is empty or malformed.");
    }

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    const std::size_t trainCount = (n * 70) / 100;
    const std::size_t valCount = (n * 15) / 100;

    std::vector<std::vector<double>> rawTrainFeatures;
    std::vector<std::vector<double>> rawValFeatures;
    std::vector<std::vector<double>> rawTestFeatures;

    outTrainLabels.clear();
    outValLabels.clear();
    outTestLabels.clear();

    rawTrainFeatures.reserve(trainCount);
    rawValFeatures.reserve(valCount);
    rawTestFeatures.reserve(n - trainCount - valCount);
    outTrainLabels.reserve(trainCount);
    outValLabels.reserve(valCount);
    outTestLabels.reserve(n - trainCount - valCount);

    for (std::size_t pos = 0; pos < n; ++pos) {
        const std::size_t idx = indices[pos];
        if (pos < trainCount) {
            rawTrainFeatures.push_back(features[idx]);
            outTrainLabels.push_back(labels[idx]);
        } else if (pos < trainCount + valCount) {
            rawValFeatures.push_back(features[idx]);
            outValLabels.push_back(labels[idx]);
        } else {
            rawTestFeatures.push_back(features[idx]);
            outTestLabels.push_back(labels[idx]);
        }
    }

    Preprocessor preprocessor;
    preprocessor.fit(rawTrainFeatures);
    outTrainFeatures = preprocessor.transformDataset(rawTrainFeatures);
    outValFeatures = preprocessor.transformDataset(rawValFeatures);
    outTestFeatures = preprocessor.transformDataset(rawTestFeatures);
    if (outFittedPreprocessor != nullptr) {
        *outFittedPreprocessor = preprocessor;
    }
}

int runPreprocessorDemo() {
    try {
        std::vector<std::vector<double>> normalizedTrain;
        std::vector<std::vector<double>> trainLabels;
        std::vector<std::vector<double>> normalizedVal;
        std::vector<std::vector<double>> valLabels;
        std::vector<std::vector<double>> normalizedTest;
        std::vector<std::vector<double>> testLabels;
        buildNormalizedSplitsInternal(
            normalizedTrain,
            trainLabels,
            normalizedVal,
            valLabels,
            normalizedTest,
            testLabels
        );

        const auto trainMeans = computeFeatureMeans(normalizedTrain);
        const auto trainStds = computeFeatureStdDev(normalizedTrain, trainMeans);
        const std::size_t n = normalizedTrain.size() + normalizedVal.size() + normalizedTest.size();

        std::cout << "Dataset size: " << n << '\n';
        std::cout << "Split sizes -> train: " << normalizedTrain.size()
                  << ", val: " << normalizedVal.size()
                  << ", test: " << normalizedTest.size() << '\n';
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Normalized train feature means: ";
        for (std::size_t j = 0; j < trainMeans.size(); ++j) {
            std::cout << trainMeans[j] << (j + 1 < trainMeans.size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Normalized train feature stddev: ";
        for (std::size_t j = 0; j < trainStds.size(); ++j) {
            std::cout << trainStds[j] << (j + 1 < trainStds.size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Sample normalized feature (train[0]): ";
        for (std::size_t j = 0; j < normalizedTrain.front().size(); ++j) {
            std::cout << normalizedTrain.front()[j] << (j + 1 < normalizedTrain.front().size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Sample label (train[0]): ";
        for (std::size_t j = 0; j < trainLabels.front().size(); ++j) {
            std::cout << trainLabels.front()[j] << (j + 1 < trainLabels.front().size() ? " " : "");
        }
        std::cout << '\n';
        std::cout << "Transformed counts -> train: " << normalizedTrain.size()
                  << ", val: " << normalizedVal.size()
                  << ", test: " << normalizedTest.size() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Preprocessor error: " << ex.what() << '\n';
        return 1;
    }
}

int buildNormalizedDatasetSplits(
    std::vector<std::vector<double>>& outTrainFeatures,
    std::vector<std::vector<double>>& outTrainLabels,
    std::vector<std::vector<double>>& outValFeatures,
    std::vector<std::vector<double>>& outValLabels,
    std::vector<std::vector<double>>& outTestFeatures,
    std::vector<std::vector<double>>& outTestLabels,
    std::string& outErrorMessage
) {
    try {
        buildNormalizedSplitsInternal(
            outTrainFeatures,
            outTrainLabels,
            outValFeatures,
            outValLabels,
            outTestFeatures,
            outTestLabels
        );
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outTrainFeatures.clear();
        outTrainLabels.clear();
        outValFeatures.clear();
        outValLabels.clear();
        outTestFeatures.clear();
        outTestLabels.clear();
        outErrorMessage = ex.what();
        return 1;
    }
}

int normalizeFeatureForInference(
    const std::vector<double>& inputFeature,
    std::vector<double>& outNormalizedFeature,
    std::string& outErrorMessage
) {
    try {
        // Cache fitted preprocessor once and reuse stored normalization parameters for inference.
        static bool isCached = false;
        static Preprocessor cachedPreprocessor;
        if (!isCached) {
            std::vector<std::vector<double>> trainFeatures;
            std::vector<std::vector<double>> trainLabels;
            std::vector<std::vector<double>> valFeatures;
            std::vector<std::vector<double>> valLabels;
            std::vector<std::vector<double>> testFeatures;
            std::vector<std::vector<double>> testLabels;
            buildNormalizedSplitsInternal(
                trainFeatures,
                trainLabels,
                valFeatures,
                valLabels,
                testFeatures,
                testLabels,
                &cachedPreprocessor
            );
            isCached = true;
        }

        outNormalizedFeature = cachedPreprocessor.transform(inputFeature);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outNormalizedFeature.clear();
        outErrorMessage = ex.what();
        return 1;
    }
}
