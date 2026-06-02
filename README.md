# Audio-Based Environmental Sound Classification System (C++)

## Runner Update - Step Selection (2026-03-10)

### What was implemented
- Updated `main.cpp` so the single executable can run any completed step from `1` to `10`.
- Added both command-line step selection and interactive step selection.
- Added optional WAV-path input for Step `2` and Step `3`.

### Logic added
- `CP.exe <step-number> [wav-path]`
- If no arguments are given, the program shows a menu and asks which step to run.
- Interactive mode stays active after each step and prints the step exit code before returning to the menu.

### How to test
- Example commands:
  - `.\cmake-build-debug\CP.exe 1`
  - `.\cmake-build-debug\CP.exe 2 data\clap\clap_02.wav`
  - `.\cmake-build-debug\CP.exe 3 data\clap\clap_02.wav`
  - `.\cmake-build-debug\CP.exe 4`
  - `.\cmake-build-debug\CP.exe 5`
  - `.\cmake-build-debug\CP.exe 6`
  - `.\cmake-build-debug\CP.exe 7`
  - `.\cmake-build-debug\CP.exe 8`
  - `.\cmake-build-debug\CP.exe 9`
  - `.\cmake-build-debug\CP.exe 10 data\clap\clap_02.wav`
- Interactive menu:
  - `.\cmake-build-debug\CP.exe`
  - Then enter one step number from the menu

## Step 1 - Project Skeleton Setup (2026-03-07)

### What was implemented
- Created required folders: `data`, `models`, `results`, `docs`.
- Created required root `.cpp` files: `AudioLoader.cpp`, `FeatureExtractor.cpp`, `Preprocessor.cpp`, `NeuralNetwork.cpp`, `Trainer.cpp`, `Evaluator.cpp`, `Utils.cpp`.
- Simplified `main.cpp` to a minimal program that compiles.
- Verified execution from terminal using CLion-generated binary at `cmake-build-debug/CP.exe`.

### Logic added
- `main.cpp` prints a single line identifying the project skeleton.

### Assumptions
- Build tooling is provided by CLion toolchain (no standalone `g++`/`clang++` on system PATH).

### How to test the current stage
- Build once in CLion, then run from terminal:
  `.\cmake-build-debug\CP.exe`
- Expected output:
  `Audio-Based Environmental Sound Classification System (skeleton)`

## Step 2 - WAV Audio Loader (2026-03-07)

### What was implemented
- Implemented `AudioLoader` class in `AudioLoader.cpp` with constructor-based configuration.
- Refined module wiring so `main.cpp` is a thin entry point and Step 2 demo flow is handled by `runAudioLoaderDemo(...)` in `AudioLoader.cpp`.
- Added strict WAV parsing for `RIFF`/`WAVE`, `fmt ` chunk, and `data` chunk.
- Added validation checks:
  - PCM format only (`audioFormat = 1`)
  - Mono channel (`1`)
  - Sample rate `16000 Hz`
  - Bit depth `16-bit`
  - Exact sample count `32000` (2 seconds at 16kHz)
- Added normalization from signed 16-bit PCM to floating-point range `[-1, 1]`.
- Updated `main.cpp` to load a WAV file, print sample count, and print first 5 sample values.

### Logic added
- `AudioLoader::loadAudio()` now reads WAV metadata, extracts PCM data, validates format constraints, and returns `std::vector<double>` samples.
- `main.cpp` now delegates to the audio loader module; no `.cpp` file is included directly.
- `runAudioLoaderDemo(...)` supports CLI path input and fallback search for a WAV under `data/`.
- Run-configuration robustness added:
  - When no CLI argument is passed, it first tries `data/clap/clap_02.wav`.
  - It resolves relative paths from both project root and `cmake-build-debug` style working directories.

### Assumptions
- Dataset files expected for this step are 2-second, 16kHz, mono, 16-bit PCM WAV files.
- Some files in dataset may not match strict Step 2 constraints; those are intentionally rejected by validation.

### How to test the current stage
- Compile from terminal with CLion bundled toolchain:
  - `$env:Path='D:/CLion 2025.3.2/bin/mingw/bin;' + $env:Path`
  - `g++ -std=c++20 -O0 -g main.cpp -o cp_step2.exe`
- Run using a valid file:
  - `.\cp_step2.exe data\clap\clap_02.wav`
- Run with no argument (matches CLion Run behavior):
  - `.\cp_step2.exe`
- Expected behavior:
  - Prints loaded file path
  - Prints `Sample count: 32000`
  - Prints first 5 normalized sample values

## Step 3 - Feature Extraction (2026-03-07)

### What was implemented
- Implemented stateless `FeatureExtractor` class in `FeatureExtractor.cpp`.
- Implemented exact 6-feature extraction from `docs/Feature_extraction_formulas.docx`:
  - Total energy (normalized by `N`)
  - Mean amplitude
  - Signal variance
  - Zero crossing rate
  - Mean frame energy (20 frames, 1600 samples each)
  - Variance of frame energy
- Added input validation (`32000` samples required) and finite-value validation (reject NaN/Inf).
- Updated `main.cpp` to run Step 3 demo (`runFeatureExtractionDemo`) and print all 6 features.

### Logic added
- Frame setup is fixed at:
  - `N = 32000`
  - `frameCount = 20`
  - `frameSize = 1600`
- `FeatureExtractor` only performs signal feature math and does not contain WAV/file handling logic.
- Audio loading for Step 3 reuses `loadValidatedAudioSamples(...)` from `AudioLoader.cpp`.

### Assumptions
- Feature formulas follow the Word docs exactly and use double precision.
- Total energy uses the recommended normalized form: `(1/N) * sum(x[n]^2)`.

### How to test the current stage
- Compile from terminal with CLion bundled toolchain:
  - `$env:Path='D:/CLion 2025.3.2/bin/mingw/bin;' + $env:Path`
  - `g++ -std=c++20 -O0 -g main.cpp AudioLoader.cpp FeatureExtractor.cpp -o cp_step3.exe`
- Run with a known valid clip:
  - `.\cp_step3.exe data\clap\clap_02.wav`
- Expected behavior:
  - Prints loaded file path
  - Prints sample count `32000`
  - Prints 6 feature values (finite, no NaN/Inf)

## Step 4 - Dataset Builder and Label Encoding (2026-03-07)

### What was implemented
- Implemented dataset-building flow in `Trainer.cpp` through `Trainer` class.
- Added traversal of class folders:
  - `data/clap`
  - `data/knock`
  - `data/noise`
  - `data/silence`
- For each WAV file:
  - Load with strict validation via `AudioLoader` module wrapper (`loadAudioFileStrict`)
  - Extract 6 features via `FeatureExtractor` module wrapper (`extractFeatureVector`)
  - Build one-hot encoded label (`[1,0,0,0]`, `[0,1,0,0]`, `[0,0,1,0]`, `[0,0,0,1]`)
  - Store as dataset sample `{features, oneHotLabel, className, filePath}`
- Updated `main.cpp` to run Step 4 demo (`runDatasetBuilderDemo`).

### Logic added
- Class order is fixed as: `clap`, `knock`, `noise`, `silence`.
- Dataset path is resolved robustly from root or `cmake-build-debug`-style run directory.
- Invalid files are skipped during dataset build to keep pipeline execution stable.

### Assumptions
- One-hot mapping order remains fixed and will be reused in later training/evaluation steps.
- Some dataset files may violate strict Step 2 WAV constraints; these are skipped.

### How to test the current stage
- Build with CLion bundled CMake/Ninja:
  - `D:/CLion 2025.3.2/bin/cmake/win/x64/bin/cmake.exe -S . -B cmake-build-debug -G Ninja`
  - `D:/CLion 2025.3.2/bin/ninja/win/x64/ninja.exe -C cmake-build-debug`
- Run:
  - `.\cmake-build-debug\CP.exe`
- Expected behavior:
  - Prints dataset size (expected around `160-200`; current run: `177`)
  - Prints one sample file/class
  - Prints one sample 6-feature vector and one-hot label

## Step 5 - Preprocessor (Normalization) (2026-03-07)

### What was implemented
- Implemented `Preprocessor` class in `Preprocessor.cpp` with encapsulated normalization parameters.
- Added `fit()` to compute feature-wise mean and standard deviation from training features only.
- Added `transform()` and `transformDataset()` for z-score normalization.
- Added deterministic shuffle and split logic (`70/15/15`) in Step 5 demo flow.
- Added dataset export wrapper in `Trainer.cpp`:
  - `buildEncodedDataset(...)` returns features + one-hot labels + metadata.
- Updated `main.cpp` to run Step 5 demo (`runPreprocessorDemo`).

### Logic added
- Split is performed after shuffling with fixed seed `42` for reproducibility.
- Normalization is fitted only on training split and then applied to train/validation/test.
- Standard deviation uses epsilon guard (`1e-8`, fallback to `1.0`) to avoid divide-by-zero.
- Validation checks include:
  - normalized train feature means (approx `0`)
  - normalized train feature stddev (approx `1`)

### Assumptions
- Z-score normalization is used (allowed by docs under preprocessing module).
- Dataset size remains `177` because 3 WAV files fail strict Step 2 constraints and are skipped.

### How to test the current stage
- Build and run:
  - `D:/CLion 2025.3.2/bin/cmake/win/x64/bin/cmake.exe -S . -B cmake-build-debug -G Ninja`
  - `D:/CLion 2025.3.2/bin/ninja/win/x64/ninja.exe -C cmake-build-debug`
  - `.\cmake-build-debug\CP.exe`
- Expected behavior:
  - Prints dataset/split sizes
  - Prints normalized train means near `0`
  - Prints normalized train stddev near `1`
  - Prints one normalized feature vector and one one-hot label

## Step 6 - Neural Network Core (2026-04-24)

### What was implemented
- Implemented `NeuralNetwork` class in `NeuralNetwork.cpp` with constructor-based configuration.
- Added full architecture exactly as required:
  - Input layer: `6`
  - Hidden layer 1: `12` neurons with `ReLU`
  - Hidden layer 2: `8` neurons with `ReLU`
  - Output layer: `4` neurons with `Softmax`
- Added from-scratch forward propagation path:
  - affine transform (`W*x + b`) for each layer
  - ReLU activation for hidden layers
  - numerically-stable Softmax at output
- Added weight initialization (He-style normal initialization) and zero bias initialization.
- Added Step 6 demo function `runNeuralNetworkDemo()` and wired it into `main.cpp`.

### Logic added
- `NeuralNetwork::forward(...)` validates input dimension and returns class probabilities.
- `NeuralNetwork::predictClass(...)` returns argmax class index from Softmax output.
- Added wrapper `forwardNetworkSample(...)` for module-level forward inference with error reporting.
- Updated runner menu and dispatch logic to support step `6`.

### Assumptions
- Step 6 focuses on architecture and forward inference only.
- Loss computation and backpropagation are deferred to later training steps.
- Class index order for output remains aligned with dataset encoding: `clap`, `knock`, `noise`, `silence`.

### How to test the current stage
- Build:
  - `D:/CLion 2025.3.2/bin/cmake/win/x64/bin/cmake.exe -S . -B cmake-build-debug -G Ninja`
  - `D:/CLion 2025.3.2/bin/ninja/win/x64/ninja.exe -C cmake-build-debug`
- Run Step 6 directly:
  - `.\cmake-build-debug\CP.exe 6`
- Expected behavior:
  - Prints architecture `6 -> 12 (ReLU) -> 8 (ReLU) -> 4 (Softmax)`
  - Prints demo input feature vector
  - Prints 4 Softmax probabilities
  - Prints predicted class index and class name

## Step 7 - Forward + Loss Math (2026-04-24)

### What was implemented
- Extended `NeuralNetwork` class to include categorical cross-entropy loss computation.
- Added strict one-hot label validation for loss input.
- Added:
  - `categoricalCrossEntropy(...)` for sample-level probability/label loss
  - `computeSampleLoss(...)` for forward + loss on one sample
  - `computeBatchLoss(...)` for mini-batch average cross-entropy
- Added `runForwardLossDemo()` and wired it to step `7` in `main.cpp`.

### Logic added
- Loss formula implemented as:
  - `L = -sum(y_i * log(p_i))` with numerical clamping (`epsilon = 1e-12`) for stability.
- Mini-batch loss is implemented as arithmetic mean of per-sample cross-entropy values.
- Step 7 demo prints:
  - one sample Softmax probability vector
  - one sample cross-entropy loss
  - mini-batch average cross-entropy loss

### Assumptions
- Labels are strict one-hot vectors of size `4` (matching output classes).
- Step 7 covers forward + objective math only; parameter update/backprop is deferred to Step 8.

### How to test the current stage
- Build:
  - `D:/CLion 2025.3.2/bin/cmake/win/x64/bin/cmake.exe -S . -B cmake-build-debug -G Ninja`
  - `D:/CLion 2025.3.2/bin/ninja/win/x64/ninja.exe -C cmake-build-debug`
- Run Step 7 directly:
  - `.\cmake-build-debug\CP.exe 7`
- Expected behavior:
  - Prints Step 7 header (`Forward + Loss Math`)
  - Prints 4 probabilities for the sample
  - Prints finite sample loss
  - Prints finite mini-batch average loss

## Step 8 - Backpropagation and Training Loop (2026-04-24)

### What was implemented
- Extended `NeuralNetwork` internals in `NeuralNetwork.cpp` to support:
  - forward-pass cache for gradients
  - ReLU derivative
  - single-sample backpropagation with gradient descent updates
  - batch accuracy computation
  - model checkpoint saving to disk
- Added training-facing wrappers in `NeuralNetwork.cpp`:
  - `createNetworkHandle(...)`
  - `trainNetworkSample(...)`
  - `evaluateNetworkBatch(...)`
  - `saveNetworkModel(...)`
  - `destroyNetworkHandle(...)`
- Added reusable normalized split API in `Preprocessor.cpp`:
  - `buildNormalizedDatasetSplits(...)`
- Implemented Step 8 training workflow in `Trainer.cpp` through `runTrainingLoopDemo()`:
  - builds normalized train/val/test splits
  - trains for fixed epochs with shuffled SGD
  - tracks validation loss and saves best checkpoint
  - reports final test loss and accuracy
- Wired Step `8` into `main.cpp`.

### Logic added
- Backprop uses softmax + categorical cross-entropy gradient (`p - y`) at output layer.
- Hidden-layer deltas use chain rule with ReLU derivative.
- Weights and biases are updated each sample via learning rate configured at network construction.
- Best validation-loss checkpoint is saved to:
  - `models/nn_step8_best_model.txt`

### Assumptions
- Step 8 uses sample-wise SGD updates (batch size = 1) while reporting epoch-level train/validation metrics.
- Normalization and split policy remains consistent with Step 5 (`70/15/15`, seed `42`).
- Model save format is a plain-text checkpoint for transparency and easy debugging.

### How to test the current stage
- Build:
  - `D:/CLion 2025.3.2/bin/cmake/win/x64/bin/cmake.exe -S . -B cmake-build-debug -G Ninja`
  - `D:/CLion 2025.3.2/bin/ninja/win/x64/ninja.exe -C cmake-build-debug`
- Run Step 8 directly:
  - `.\cmake-build-debug\CP.exe 8`
- Expected behavior:
  - Prints training configuration and split sizes
  - Prints periodic epoch metrics (`train_loss`, `val_loss`, `val_acc`)
  - Prints best epoch and checkpoint path
  - Prints test loss and test accuracy

## Step 9 - Evaluation Module (2026-04-24)

### What was implemented
- Implemented `Evaluator` class in `Evaluator.cpp`.
- Added model-loading support in `NeuralNetwork.cpp` and wrapper API:
  - `loadNetworkModel(...)`
  - `predictNetworkClass(...)`
- Added Step 9 evaluation flow `runEvaluatorDemo()` and wired it into `main.cpp`.
- Added report export to:
  - `results/step9_evaluation_report.txt`

### Logic added
- Step 9 loads the best checkpoint from Step 8:
  - `models/nn_step8_best_model.txt`
- Reuses normalized validation/test splits from preprocessing module.
- Computes and prints:
  - validation loss and accuracy
  - test loss and accuracy
  - macro F1 score
- Builds test confusion matrix (rows=true class, cols=predicted class).
- Computes per-class:
  - precision
  - recall
  - F1 score
  - support
- Saves all metrics and confusion matrix to a text report under `results/`.

### Assumptions
- Step 8 checkpoint exists before running Step 9.
- Class order is unchanged: `clap`, `knock`, `noise`, `silence`.
- Evaluation uses the same deterministic split/normalization policy from Step 5/Step 8.

### How to test the current stage
- Build:
  - `D:/CLion 2025.3.2/bin/cmake/win/x64/bin/cmake.exe -S . -B cmake-build-debug -G Ninja`
  - `D:/CLion 2025.3.2/bin/ninja/win/x64/ninja.exe -C cmake-build-debug`
- Run Step 9 directly:
  - `.\cmake-build-debug\CP.exe 9`
- Expected behavior:
  - Prints validation and test metrics
  - Prints macro F1
  - Prints saved report path under `results/`

## Step 10 - End-to-End Inference (2026-04-24)

### What was implemented
- Added complete inference flow in `Evaluator.cpp` through:
  - `runEndToEndInferenceDemo(const std::string& requestedPath)`
- Added probability-prediction wrapper in `NeuralNetwork.cpp`:
  - `predictNetworkProbabilities(...)`
- Added single-feature normalization wrapper in `Preprocessor.cpp`:
  - `normalizeFeatureForInference(...)`
- Wired Step `10` into `main.cpp` menu and dispatch.
- Added Step 10 inference report export to:
  - `results/step10_inference_report.txt`

### Logic added
- Step 10 pipeline executes in strict module order:
  - Audio loading (`AudioLoader`)
  - Feature extraction (`FeatureExtractor`)
  - Training-consistent normalization (`Preprocessor`)
  - Model loading (`NeuralNetwork`, checkpoint from Step 8)
  - Softmax inference and class prediction
- Prints predicted class index, class name, and 4 class probabilities.
- Saves inference summary and probabilities to a text report in `results/`.

### Assumptions
- The trained checkpoint file exists at:
  - `models/nn_step8_best_model.txt`
- Input WAV follows existing dataset constraints:
  - PCM, mono, 16kHz, 16-bit, 2 seconds (32000 samples)
- If no WAV path is given, fallback file resolution from earlier steps is reused.

### How to test the current stage
- Build:
  - `D:/CLion 2025.3.2/bin/cmake/win/x64/bin/cmake.exe -S . -B cmake-build-debug -G Ninja`
  - `D:/CLion 2025.3.2/bin/ninja/win/x64/ninja.exe -C cmake-build-debug`
- Run Step 10 directly:
  - `.\cmake-build-debug\CP.exe 10 data\clap\clap_02.wav`
- Expected behavior:
  - Prints loaded input WAV path
  - Prints predicted class index and class name
  - Prints 4 Softmax probabilities
  - Prints saved inference report path under `results/`
