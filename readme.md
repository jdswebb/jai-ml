# MLP Neural Network Trainer & Visualizer

An interactive, from-scratch implementation of a multilayer perceptron (MLP) neural network in Jai, with real-time DX11 visualization, training, and inference. Developed as a personal learning project while following the excellent [3Blue1Brown neural network series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). Since I wrote this while learning myself, the code should be fairly easy to follow with the videos but isn’t meant to be optimal or a reference implementation.

![Neural Network Visualization](emnist.gif)

## Features

- **Neural Network Implementation**
  - Forward and backpropagation from scratch
  - Customizable layer configurations and activation functions
  - MSE and cross-entropy loss support

- **Dataset Support**
  - Supports MNIST (digits) and EMNIST balanced (digits + characters) datasets
  - Save/load models at runtime
  - Pre-trained examples included for both datasets

- **Real-time Visualization**
  - Interactive network structure and activation display
  - Customizable themes and rendering options
  - Live accuracy tracking

## Build and run:

At the repository root:

```
jai main.jai
./main
```

There are significant performance benefits to building with the `VERY_RELEASE` configuration in Jai, which does a better job of auto-vectorizing some of the loops (even though they were written to try and trigger this).

## Training

The repository includes pre-trained networks ready for immediate use. To train your own models:

### 1. Download MNIST Dataset
From [Kaggle MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv):
- Download `mnist_test.csv` and `mnist_train.csv`
- Copy both files to the `data` directory

### 2. Download EMNIST Dataset
From [Kaggle EMNIST](https://www.kaggle.com/datasets/crawford/emnist):
- Download `emnist_balanced_train.csv` and `emnist_balanced_test.csv`
- Copy both files to the `data` directory

Click 'Rebuild and train' in the GUI to rebuild for the configuration specified on the GUI, or 'Continue training' to continue running training on the current network.

## License

MIT

**Third party:**
- SDL2 - Licensed under zlib license
- Dear ImGui - Licensed under MIT license
- Contains modified snippets from the Jai standard library

See respective library licenses for full terms.