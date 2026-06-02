#include <iostream>
#include <limits>
#include <string>

int runAudioLoaderDemo(const std::string& requestedPath);
int runFeatureExtractionDemo(const std::string& requestedPath);
int runDatasetBuilderDemo();
int runPreprocessorDemo();
int runNeuralNetworkDemo();
int runForwardLossDemo();
int runTrainingLoopDemo();
int runEvaluatorDemo();
int runEndToEndInferenceDemo(const std::string& requestedPath);

static int runStep1Demo() {
    std::cout << "Audio-Based Environmental Sound Classification System (skeleton)\n";
    return 0;
}

static void printStepMenu() {
    std::cout << "Select a step to run:\n";
    std::cout << "0. Exit\n";
    std::cout << "1. Project Skeleton Setup\n";
    std::cout << "2. WAV Audio Loader\n";
    std::cout << "3. Feature Extraction\n";
    std::cout << "4. Dataset Builder and Label Encoding\n";
    std::cout << "5. Preprocessor Normalization\n";
    std::cout << "6. Neural Network Core\n";
    std::cout << "7. Forward + Loss Math\n";
    std::cout << "8. Backpropagation + Training Loop\n";
    std::cout << "9. Evaluation Module\n";
    std::cout << "10. End-to-End Inference\n";
    std::cout << "Enter step number: ";
}

static int runSelectedStep(int step, const std::string& wavPath) {
    switch (step) {
        case 1:
            return runStep1Demo();
        case 2:
            return runAudioLoaderDemo(wavPath);
        case 3:
            return runFeatureExtractionDemo(wavPath);
        case 4:
            return runDatasetBuilderDemo();
        case 5:
            return runPreprocessorDemo();
        case 6:
            return runNeuralNetworkDemo();
        case 7:
            return runForwardLossDemo();
        case 8:
            return runTrainingLoopDemo();
        case 9:
            return runEvaluatorDemo();
        case 10:
            return runEndToEndInferenceDemo(wavPath);
        default:
            std::cerr << "Invalid step. Choose a step number from 1 to 10.\n";
            return 1;
    }
}

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        try {
            const int step = std::stoi(argv[1]);
            const std::string wavPath = (argc >= 3) ? argv[2] : "";
            return runSelectedStep(step, wavPath);
        } catch (const std::exception&) {
            std::cerr << "Usage: CP.exe <step-number> [wav-path]\n";
            std::cerr << "Example: CP.exe 3 data\\clap\\clap_02.wav\n";
            return 1;
        }
    }

    while (true) {
        printStepMenu();
        int step = 0;
        if (!(std::cin >> step)) {
            std::cerr << "Failed to read step number.\n";
            return 1;
        }

        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        if (step == 0) {
            std::cout << "Exiting.\n";
            return 0;
        }

        std::string wavPath;
        if (step == 2 || step == 3 || step == 10) {
            std::cout << "Enter WAV path or press Enter to use default: ";
            std::getline(std::cin, wavPath);
        }

        const int stepExitCode = runSelectedStep(step, wavPath);
        std::cout << "Step finished with exit code: " << stepExitCode << "\n\n";
    }
}
