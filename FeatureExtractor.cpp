#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class FeatureExtractor {
private:
    int expectedSampleCount;
    int frameCount;
    int frameSize;
    double epsilon;

public:
    FeatureExtractor(int sampleCount = 32000, int frames = 20, int samplesPerFrame = 1600)
        : expectedSampleCount(sampleCount), frameCount(frames), frameSize(samplesPerFrame), epsilon(1e-8) {}

    std::vector<double> extractFeatures(const std::vector<double>& samples) const {
        if (static_cast<int>(samples.size()) != expectedSampleCount) {
            throw std::runtime_error("FeatureExtractor expects exactly 32000 samples.");
        }
        if (frameCount * frameSize != expectedSampleCount) {
            throw std::runtime_error("Invalid frame configuration.");
        }

        const double n = static_cast<double>(expectedSampleCount);
        double totalEnergy = 0.0;
        double sum = 0.0;
        int zeroCrossings = 0;

        for (int i = 0; i < expectedSampleCount; ++i) {
            const double x = samples[static_cast<std::size_t>(i)];
            totalEnergy += x * x;
            sum += x;
            if (i > 0 && x * samples[static_cast<std::size_t>(i - 1)] < 0.0) {
                ++zeroCrossings;
            }
        }

        totalEnergy /= n;
        const double meanAmplitude = sum / n;

        double variance = 0.0;
        for (int i = 0; i < expectedSampleCount; ++i) {
            const double diff = samples[static_cast<std::size_t>(i)] - meanAmplitude;
            variance += diff * diff;
        }
        variance /= n;

        const double zcr = static_cast<double>(zeroCrossings) / static_cast<double>(expectedSampleCount - 1);

        std::vector<double> frameEnergies;
        frameEnergies.reserve(static_cast<std::size_t>(frameCount));

        for (int frame = 0; frame < frameCount; ++frame) {
            double frameEnergy = 0.0;
            const int start = frame * frameSize;
            for (int m = 0; m < frameSize; ++m) {
                const double x = samples[static_cast<std::size_t>(start + m)];
                frameEnergy += x * x;
            }
            frameEnergy /= static_cast<double>(frameSize);
            frameEnergies.push_back(frameEnergy);
        }

        double meanFrameEnergy = 0.0;
        for (double energy : frameEnergies) {
            meanFrameEnergy += energy;
        }
        meanFrameEnergy /= static_cast<double>(frameCount);

        double varianceFrameEnergyRaw = 0.0;
        for (double energy : frameEnergies) {
            const double diff = energy - meanFrameEnergy;
            varianceFrameEnergyRaw += diff * diff;
        }
        varianceFrameEnergyRaw /= static_cast<double>(frameCount);

        // Normalized frame-energy variance for better relative fluctuation separation.
        const double varianceFrameEnergy = varianceFrameEnergyRaw / (meanFrameEnergy + epsilon);

        std::vector<double> features = {
            totalEnergy,
            meanAmplitude,
            variance,
            zcr,
            meanFrameEnergy,
            varianceFrameEnergy
        };

        for (double value : features) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite feature value encountered.");
            }
        }

        return features;
    }
};

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
) {
    try {
        FeatureExtractor extractor(32000, 20, 1600);
        outFeatures = extractor.extractFeatures(samples);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outFeatures.clear();
        outErrorMessage = ex.what();
        return 1;
    }
}

int runFeatureExtractionDemo(const std::string& requestedPath) {
    std::vector<double> samples;
    std::string selectedPath;
    std::string errorMessage;

    if (loadValidatedAudioSamples(requestedPath, samples, selectedPath, errorMessage) != 0) {
        std::cerr << "FeatureExtraction error: " << errorMessage << '\n';
        return 1;
    }

    try {
        FeatureExtractor extractor(32000, 20, 1600);
        const std::vector<double> features = extractor.extractFeatures(samples);

        std::cout << "Loaded file: " << selectedPath << '\n';
        std::cout << "Sample count: " << samples.size() << '\n';
        std::cout << "Feature vector (6):\n";
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "1. Total energy: " << features[0] << '\n';
        std::cout << "2. Mean amplitude: " << features[1] << '\n';
        std::cout << "3. Signal variance: " << features[2] << '\n';
        std::cout << "4. Zero crossing rate: " << features[3] << '\n';
        std::cout << "5. Mean frame energy: " << features[4] << '\n';
        std::cout << "6. Variance frame energy: " << features[5] << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FeatureExtraction error: " << ex.what() << '\n';
        return 1;
    }
}
