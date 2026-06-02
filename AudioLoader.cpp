#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class AudioLoader {
private:
    int expectedSampleRate;
    int expectedDurationSeconds;
    int expectedChannels;

    [[nodiscard]] int expectedSampleCount() const {
        return expectedSampleRate * expectedDurationSeconds;
    }

    static std::uint16_t readU16(std::istream& stream) {
        std::uint16_t value = 0;
        stream.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }

    static std::uint32_t readU32(std::istream& stream) {
        std::uint32_t value = 0;
        stream.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }

public:
    AudioLoader(int sampleRate = 16000, int durationSeconds = 2, int channels = 1)
        : expectedSampleRate(sampleRate), expectedDurationSeconds(durationSeconds), expectedChannels(channels) {}

    std::vector<double> loadAudio(const std::string& filePath) const {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open WAV file: " + filePath);
        }

        char riff[4];
        file.read(riff, 4);
        if (std::string(riff, 4) != "RIFF") {
            throw std::runtime_error("Invalid WAV header: missing RIFF.");
        }

        (void)readU32(file);

        char wave[4];
        file.read(wave, 4);
        if (std::string(wave, 4) != "WAVE") {
            throw std::runtime_error("Invalid WAV header: missing WAVE.");
        }

        bool fmtFound = false;
        bool dataFound = false;
        std::uint16_t audioFormat = 0;
        std::uint16_t numChannels = 0;
        std::uint32_t sampleRate = 0;
        std::uint16_t bitsPerSample = 0;
        std::vector<char> audioBytes;

        while (file && (!fmtFound || !dataFound)) {
            char chunkId[4];
            file.read(chunkId, 4);
            if (file.gcount() != 4) {
                break;
            }

            const std::uint32_t chunkSize = readU32(file);
            const std::string id(chunkId, 4);

            if (id == "fmt ") {
                audioFormat = readU16(file);
                numChannels = readU16(file);
                sampleRate = readU32(file);
                (void)readU32(file);
                (void)readU16(file);
                bitsPerSample = readU16(file);

                if (chunkSize > 16) {
                    file.seekg(static_cast<std::streamoff>(chunkSize - 16), std::ios::cur);
                }
                fmtFound = true;
            } else if (id == "data") {
                audioBytes.resize(chunkSize);
                file.read(audioBytes.data(), static_cast<std::streamsize>(chunkSize));
                if (file.gcount() != static_cast<std::streamsize>(chunkSize)) {
                    throw std::runtime_error("WAV data chunk is truncated.");
                }
                dataFound = true;
            } else {
                file.seekg(static_cast<std::streamoff>(chunkSize), std::ios::cur);
            }

            if (chunkSize % 2 == 1) {
                file.seekg(1, std::ios::cur);
            }
        }

        if (!fmtFound || !dataFound) {
            throw std::runtime_error("Required WAV chunks (fmt/data) not found.");
        }
        if (audioFormat != 1) {
            throw std::runtime_error("Only PCM WAV is supported.");
        }
        if (numChannels != expectedChannels) {
            throw std::runtime_error("Expected mono WAV (1 channel).");
        }
        if (sampleRate != static_cast<std::uint32_t>(expectedSampleRate)) {
            throw std::runtime_error("Expected WAV sample rate: 16000 Hz.");
        }
        if (bitsPerSample != 16) {
            throw std::runtime_error("Expected 16-bit PCM WAV.");
        }

        const std::size_t totalSamples = audioBytes.size() / sizeof(std::int16_t);
        if (totalSamples != static_cast<std::size_t>(expectedSampleCount())) {
            throw std::runtime_error(
                "Expected exactly " + std::to_string(expectedSampleCount()) +
                " samples (2 seconds at 16kHz)."
            );
        }

        std::vector<double> samples;
        samples.reserve(totalSamples);

        const auto* pcm = reinterpret_cast<const std::int16_t*>(audioBytes.data());
        for (std::size_t i = 0; i < totalSamples; ++i) {
            double value = static_cast<double>(pcm[i]) / 32768.0;
            value = std::clamp(value, -1.0, 1.0);
            samples.push_back(value);
        }

        return samples;
    }
};

static std::vector<std::filesystem::path> collectDataDirectories() {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<std::filesystem::path> dirs = {
        cwd / "data",
        cwd / ".." / "data",
        cwd / ".." / ".." / "data",
        cwd / ".." / ".." / ".." / "data"
    };

    std::vector<std::filesystem::path> existingDirs;
    for (const auto& dir : dirs) {
        if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
            existingDirs.push_back(std::filesystem::weakly_canonical(dir));
        }
    }
    return existingDirs;
}

static bool isWavExtension(const std::filesystem::path& filePath) {
    std::string ext = filePath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".wav";
}

static std::vector<std::filesystem::path> collectWavFiles() {
    std::vector<std::filesystem::path> wavFiles;
    for (const auto& dataDir : collectDataDirectories()) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(dataDir)) {
            if (entry.is_regular_file() && isWavExtension(entry.path())) {
                wavFiles.push_back(entry.path());
            }
        }
    }
    std::sort(wavFiles.begin(), wavFiles.end());
    return wavFiles;
}

static std::optional<std::string> resolveProvidedPath(const std::string& requestedPath) {
    if (requestedPath.empty()) {
        return std::nullopt;
    }

    std::filesystem::path inputPath(requestedPath);
    if (std::filesystem::exists(inputPath) && std::filesystem::is_regular_file(inputPath)) {
        return std::filesystem::weakly_canonical(inputPath).string();
    }

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<std::filesystem::path> candidates = {
        cwd / inputPath,
        cwd / ".." / inputPath,
        cwd / ".." / ".." / inputPath,
        cwd / ".." / ".." / ".." / inputPath
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate) && std::filesystem::is_regular_file(candidate)) {
            return std::filesystem::weakly_canonical(candidate).string();
        }
    }

    return std::nullopt;
}

int loadValidatedAudioSamples(
    const std::string& requestedPath,
    std::vector<double>& outSamples,
    std::string& outSelectedPath,
    std::string& outErrorMessage
) {
    try {
        AudioLoader loader(16000, 2, 1);
        outSamples.clear();
        outSelectedPath.clear();
        outErrorMessage.clear();

        if (const auto provided = resolveProvidedPath(requestedPath); provided.has_value()) {
            outSelectedPath = provided.value();
            outSamples = loader.loadAudio(outSelectedPath);
            return 0;
        }

        if (const auto preferred = resolveProvidedPath("data/clap/clap_02.wav"); preferred.has_value()) {
            outSelectedPath = preferred.value();
            outSamples = loader.loadAudio(outSelectedPath);
            return 0;
        }

        const auto wavFiles = collectWavFiles();
        if (wavFiles.empty()) {
            outErrorMessage = "No WAV file provided and none found under data/.";
            return 1;
        }

        std::string lastError = "No valid WAV file found for Step 2 constraints.";
        for (const auto& wavPath : wavFiles) {
            try {
                outSamples = loader.loadAudio(wavPath.string());
                outSelectedPath = wavPath.string();
                return 0;
            } catch (const std::exception& ex) {
                lastError = ex.what();
            }
        }

        outErrorMessage = lastError;
        return 1;
    } catch (const std::exception& ex) {
        outErrorMessage = ex.what();
        return 1;
    }
}

int loadAudioFileStrict(
    const std::string& filePath,
    std::vector<double>& outSamples,
    std::string& outErrorMessage
) {
    try {
        AudioLoader loader(16000, 2, 1);
        outSamples = loader.loadAudio(filePath);
        outErrorMessage.clear();
        return 0;
    } catch (const std::exception& ex) {
        outSamples.clear();
        outErrorMessage = ex.what();
        return 1;
    }
}

int runAudioLoaderDemo(const std::string& requestedPath) {
    std::vector<double> samples;
    std::string selectedPath;
    std::string errorMessage;
    if (loadValidatedAudioSamples(requestedPath, samples, selectedPath, errorMessage) != 0) {
        std::cerr << "AudioLoader error: " << errorMessage << '\n';
        return 1;
    }

    std::cout << "Loaded file: " << selectedPath << '\n';
    std::cout << "Sample count: " << samples.size() << '\n';
    std::cout << "First 5 samples: ";
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < 5 && i < samples.size(); ++i) {
        std::cout << samples[i] << (i < 4 && i + 1 < samples.size() ? " " : "");
    }
    std::cout << '\n';
    return 0;
}
