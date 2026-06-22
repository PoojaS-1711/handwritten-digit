# CNN Handwritten Digit Recognizer Playground

An interactive Convolutional Neural Network (CNN) playground to train, test, and visualize deep learning models in real-time. Built using **PyTorch** for model training and **Streamlit** for the web interface.

---

## ✨ Features

- **🎨 Interactive Canvas**: Draw digits (0-9) on a canvas and get real-time classification predictions and confidence charts.
- **🔬 Activation Map Inspector**: Visualize exactly what the convolutional layers (Conv1 and Conv2) "see" (extracting edges, orientations, and shapes) using your drawings as input.
- **⚙️ Live Hyperparameter Sandbox**: Configure epochs, batch size, and learning rate directly from the UI, run PyTorch training, and watch live training curves (loss/accuracy) update in real-time.
- **🧠 CNN Architecture Explainer**: Educational step-by-step breakdown of how Convolutions, ReLU activation, Max Pooling, and Dense layers work.

---

## 🛠️ Model Architecture

The network is defined in PyTorch as follows:
1. **Input**: Grayscale image, `28 x 28 x 1` (Standard MNIST format).
2. **Conv Layer 1**: `16` filters of size `3x3` (with padding `1` to maintain resolution) + **ReLU** activation.
3. **Max Pool 1**: `2x2` pooling (strided), downsampling resolution to `14 x 14`.
4. **Conv Layer 2**: `32` filters of size `3x3` (with padding `1`) + **ReLU** activation.
5. **Max Pool 2**: `2x2` pooling (strided), downsampling resolution to `7 x 7`.
6. **Flatten**: Reshapes the `32 x 7 x 7` tensor into a `1568`-element 1D vector.
7. **Dense Layer 1**: Fully connected mapping `1568` to `128` nodes + **ReLU** activation + **Dropout** (`0.25` rate to prevent overfitting).
8. **Output Layer**: Fully connected mapping `128` to `10` logits (representing digits 0-9).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or newer installed on your system.

### One-Click Launch (Windows)
Double-click the **`run.bat`** file. It will automatically check your Python environment, install required packages (using the lightweight CPU version of PyTorch), and start the Streamlit web application.

### Manual Setup

1. **Clone/Move into the folder**:
   ```bash
   E:
   cd E:\handwritten-digit-recognizer
   ```

2. **Install Dependencies**:
   It is recommended to install the CPU-only version of PyTorch for quick setup:
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

3. **Run Web Interface**:
   ```bash
   streamlit run app.py
   ```

4. **CLI Training (Optional)**:
   If you want to train the network from the command line instead of the UI:
   ```bash
   python train.py --epochs 5 --batch-size 64 --lr 0.001
   ```

---

## 📄 File Listing

- `model.py` - Network architecture definition and activation extraction helper.
- `train.py` - MNIST training code pipeline.
- `app.py` - Streamlit visual interface.
- `run.bat` - Automates installation & execution.
- `requirements.txt` - Required python packages.
- `data/` - MNIST dataset storage directory.
- `mnist_cnn.pth` - Saved PyTorch weights (pre-trained to **98.86% accuracy**).
- `training_curves.png` - Loss and accuracy curves generated during training.
