# MNIST Handwritten Digit Recognition Using a Multi-Layer Perceptron

**Language / Dil:** [English](README.md) | [Türkçe](README_TR.md)

**Deep Learning Course — Experimental Assignment**  
**Framework:** PyTorch  
**Model:** Multi-Layer Perceptron (MLP), trained from scratch

### Student Information

| Field | Value |
|---|---|
| Full name | Kerem YILMAZ |
| Student number | 23120205035 |
| Course | BIL485 Derin Öğrenme (Deep Learning) |

## 1. Project Overview

This project investigates handwritten digit recognition using the MNIST dataset and a fully connected Multi-Layer Perceptron (MLP). The objectives are to:

- Train an MLP with at least two hidden layers to classify digits from 0 to 9.
- Track training and validation loss and accuracy across epochs.
- Compare model capacity and learning rate while keeping the data split and other settings fixed.
- Evaluate the selected model on a separate MNIST test set and on a photograph of my own handwritten student number.
- Discuss overfitting, generalization, and the limitations of a small real-world handwriting sample.

**No convolutional neural networks (CNNs) or pretrained models are used.** The MLP weights are initialized and learned during training.

## 2. Dataset and Preprocessing

[MNIST](https://www.tensorflow.org/datasets/catalog/mnist) contains 70,000 grayscale images of handwritten digits (0–9), each with a resolution of 28 × 28 pixels.

| Subset | Number of images | Purpose |
|---|---:|---|
| Training | 50,000 | Update model weights |
| Validation | 10,000 | Monitor training and select a model checkpoint |
| Test | 10,000 | Report performance on held-out MNIST examples |

The training and validation subsets are obtained from MNIST's original 60,000-image training partition using a seeded random split (`seed = 42`). The official 10,000-image test partition is kept separate. The same split is used in all three experiments.

MNIST inputs are converted to floating-point tensors and normalized as follows:

```text
normalized_pixel = (pixel_in_[0,1] - 0.1307) / 0.3081
```

The MLP flattens each 28 × 28 image into a 784-dimensional input vector. The model does not use convolutional layers.

## 3. Model Architecture

The baseline network consists of two fully connected hidden layers with ReLU activation:

```text
Input: 28 × 28 grayscale image
        |
     Flatten
        |
   Linear(784, 256)
        |
       ReLU
        |
   Linear(256, 128)
        |
       ReLU
        |
    Linear(128, 10)
        |
  10 class logits (0–9)
```

The raw output logits are passed directly to PyTorch's `CrossEntropyLoss`, which handles the classification loss. At inference time, `softmax` is used to display per-digit confidence scores.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Batch size | 64 |
| Maximum epochs | 15 |
| Random seed | 42 |
| Baseline hidden layer sizes | 256, 128 |
| Baseline learning rate | 0.001 |
| Activation function | ReLU |
| Input normalization | Mean 0.1307, standard deviation 0.3081 |
| Checkpoint selection | Lowest validation loss |

For each epoch, training and validation loss/accuracy are recorded. The weights from the epoch with the lowest validation loss are saved as `best_model.pt`. MNIST test evaluation takes place after the training run, using the selected checkpoint; test data are not used for gradient updates or checkpoint selection.

## 4. Experiments

Three configurations were trained to study the influence of the number of hidden neurons and the learning rate. All other settings were kept the same.

| Experiment | Hidden layers | Learning rate | Trainable parameters |
|---|---|---:|---:|
| **1 — Baseline** | 256 → 128 | 0.001 | 235,146 |
| **2 — Smaller MLP** | 128 → 64 | 0.001 | 109,386 |
| **3 — Lower Learning Rate** | 256 → 128 | 0.0001 | 235,146 |

### 4.1 Quantitative Results

Training and validation metrics in the table below are measured **at each experiment's selected (lowest-validation-loss) epoch**, rather than necessarily at the final epoch.

| Metric | Baseline | Smaller MLP | Lower LR |
|---|---:|---:|---:|
| Best epoch | 5 | 6 | 15 |
| Training loss | 0.0428 | 0.0439 | 0.0319 |
| Validation loss | 0.0963 | 0.0932 | **0.0876** |
| Training accuracy | 98.63% | 98.61% | 99.19% |
| Validation accuracy | 97.38% | 97.29% | 97.16% |
| Test accuracy | **97.76%** | 97.61% | 97.73% |
| Training time | 83.6 s | 181.1 s | 95.9 s |

*Training times are single observed runs on a CUDA-enabled machine, not controlled performance benchmarks.* In particular, the smaller model took longer in this run, despite having fewer parameters. Timing differences may also reflect system load and other execution conditions.

### 4.2 Training Curves

The project saves training and validation plots for each experiment:

**Baseline — Loss**

![Baseline training and validation loss](results/baseline/loss.png)

**Baseline — Accuracy**

![Baseline training and validation accuracy](results/baseline/accuracy.png)

Additional plots:

- [Smaller MLP: loss](results/smaller/loss.png) · [accuracy](results/smaller/accuracy.png)
- [Lower learning rate: loss](results/lower_lr/loss.png) · [accuracy](results/lower_lr/accuracy.png)

Each experiment also saves its per-epoch metrics in `history.csv` and a summary in `summary.json`.

### 4.3 Discussion

**Baseline (256 → 128, learning rate 0.001).** The model learned quickly. Its minimum validation loss occurred at epoch 5 (`0.0963`). Training loss continued to decrease afterward, but validation loss fluctuated and was `0.1230` by epoch 15, while training loss reached `0.0132`. This growing gap indicates overfitting: improved fit to the training data did not translate into consistently better validation performance.

**Smaller MLP (128 → 64, learning rate 0.001).** Reducing the hidden layer sizes decreased the number of trainable parameters from 235,146 to 109,386 (approximately **53.5% fewer**). Nevertheless, test accuracy decreased by only 0.15 percentage points (97.76% to 97.61%). Thus, for this task, a substantially smaller network retained similar classification performance. Its best validation loss was `0.0932` at epoch 6, and later epochs also showed signs of overfitting.

**Lower learning rate (256 → 128, learning rate 0.0001).** With a learning rate ten times smaller, training began more slowly but validation loss generally continued to decrease throughout the 15 epochs. It reached the lowest validation loss of all three experiments (`0.0876`) at epoch 15. This run did not show the same sustained late-epoch increase in validation loss as the baseline. It is possible that further epochs would help, but this was not tested.

**Model selection.** The selection criterion was **minimum validation loss**, so the Lower LR configuration was chosen for the separate handwriting experiment. The baseline's test accuracy was marginally higher (97.76% versus 97.73%), but these results differ by only 0.03 percentage points (three predictions out of 10,000) and do not provide strong evidence of a meaningful difference. Test accuracies are reported for completeness; they were not used to choose the handwriting model.

## 5. Evaluation on My Own Handwriting

To evaluate performance outside the MNIST test set, I wrote my student number on paper and photographed it. The image is stored at [`handwriting/student_number.jpg`](handwriting/student_number.jpg).

### 5.1 Processing the Photograph

The photograph was processed using the following steps:

1. Rotate it 90° counterclockwise to make the writing horizontal.
2. Convert it to grayscale and invert the thresholded result so the digits appear light on a black background.
3. Identify separated digits using connected-component analysis and sort them in left-to-right order.
4. Preserve each digit's aspect ratio while resizing the foreground to fit within approximately 20 × 20 pixels.
5. Center the digit in a 28 × 28 canvas using its image moments.
6. Scale pixel intensities to `[0, 1]` and apply the same MNIST mean/std normalization used in training.

This photograph contained 11 separate digits. The connected-component method worked for this photograph because the handwritten characters were not touching. It may need adjustment for overlapping digits or different lighting conditions.

**Preprocessed digits:**

![Handwritten digits converted to 28x28 MNIST-like images](results/handwriting/mnist_preview.png)

**Predictions:**

![Predictions on handwritten student number](results/handwriting/prediction_preview.png)

### 5.2 Handwriting Results

| Evaluation | Result |
|---|---:|
| Handwritten digits evaluated | 11 |
| Correctly classified digits | 11 |
| Incorrectly classified digits | 0 |
| Handwriting accuracy | **100.00% (11/11)** |
| Selected model | Lower LR (256 → 128, lr = 0.0001) |
| Selected model's MNIST test accuracy | 97.73% |

All 11 digits were predicted correctly. The individual softmax confidence scores ranged from approximately 97.5% to 100.0%. **Softmax confidence is not necessarily a calibrated probability of correctness.**

These findings show that the selected MLP was able to recognize this particular real-world handwritten sample after preprocessing. However, **11 digits from a single photograph are too few to establish general real-world accuracy**. The 100% handwriting accuracy should not be interpreted as proof that the model outperforms its 97.73% MNIST test accuracy. A larger set of handwriting samples collected under different conditions would be required to evaluate distribution shift more reliably.

Per-digit predictions and labels are available in [`results/handwriting/predictions.csv`](results/handwriting/predictions.csv) and [`predictions.json`](results/handwriting/predictions.json).

## 6. Reproducing the Experiments

### Requirements

- Python (the experiments were run using Python 3.14 on Windows)
- PyTorch and torchvision (installed environment: `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`)
- NumPy, Matplotlib, Pillow, scikit-learn, OpenCV
- A CUDA-compatible NVIDIA GPU is optional; CPU execution is supported

Install PyTorch using the instructions appropriate to your system on the [official PyTorch website](https://pytorch.org/get-started/locally/). Then install the other project dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install torch and torchvision as directed at pytorch.org first.
python -m pip install -r requirements.txt
```

### Run the Three Training Experiments

```powershell
# Experiment 1: baseline
python train.py

# Experiment 2: smaller model
python train.py --hidden1 128 --hidden2 64 --output-dir results/smaller

# Experiment 3: lower learning rate
python train.py --lr 0.0001 --output-dir results/lower_lr
```

MNIST is downloaded automatically into `data/`. Each run generates its own graphs, epoch-by-epoch history, summary, and best checkpoint in the selected `results/` directory.

### Run Handwriting Recognition

Once the Lower LR experiment has generated `results/lower_lr/best_model.pt`, run:

```powershell
python predict_handwriting.py --image handwriting/student_number.jpg --rotate 90 --number 23120205035 --model results/lower_lr/best_model.pt
```

The `--number` argument supplies the reference labels for evaluation; the model does **not** receive these labels as input. Prediction artifacts are written to `results/handwriting/`.

### Repository Structure

```text
mnist-mlp/
├── model.py                         # Fully connected MLP architecture
├── train.py                         # Data loading, training, evaluation, plots
├── predict_handwriting.py           # Photo preprocessing and inference
├── requirements.txt                 # Python dependencies (except PyTorch)
├── .gitignore
├── README.md                        # English documentation
├── README_TR.md                     # Turkish documentation
├── handwriting/
│   └── student_number.jpg           # Original handwritten photograph
└── results/
    ├── baseline/
    │   ├── history.csv
    │   ├── summary.json
    │   ├── loss.png
    │   └── accuracy.png
    ├── smaller/                      # Same experiment files
    ├── lower_lr/                     # Same experiment files
    └── handwriting/
        ├── digits/                   # 11 processed 28x28 PNGs
        ├── mnist_preview.png
        ├── segmentation.png
        ├── prediction_preview.png
        ├── predictions.csv
        └── predictions.json
```

The downloaded dataset (`data/`), virtual environment (`.venv/`), and locally generated model checkpoints (`*.pt`) do not need to be committed: checkpoints can be regenerated using the commands above. For GitHub display, ensure that the plots and handwriting-result images referenced by this README **are committed**.

## 7. Conclusion and Limitations

The experiments demonstrate that a two-hidden-layer MLP can classify MNIST digits with approximately **97.6–97.8% test accuracy**, without convolutional layers or pretrained weights. Reducing the hidden layer widths produced a considerably smaller model with only a modest decrease in classification accuracy. Reducing the learning rate slowed early training but achieved the best validation loss after 15 epochs.

The training curves illustrate the importance of validation monitoring: continuously decreasing training loss does not guarantee improved performance on unseen data. The Lower LR model correctly recognized all 11 digits in the submitted handwritten photograph after MNIST-style preprocessing, but this limited sample is not sufficient to make a broad claim about generalization or robustness to distribution shift.

Further investigation could include repeated runs with different seeds, learning-rate scheduling, early stopping, and a larger, more varied collection of handwritten samples. These are potential future extensions, **not experiments performed in this submission**.
