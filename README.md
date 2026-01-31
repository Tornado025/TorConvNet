# TorConvNet

This repo contains a simple implementation of a Convolutional Neural Network built from scratch using NumPy. It was not designed with testing or performance kept in mind and was only made for understanding purposes.

## Prerequisites

- Python 3.7+
- NumPy
- TensorFlow/Keras (for MNIST dataset)

## Features

- Conv3x3 convolutional layer with customizable padding and stride
- Pool2 max pooling layer
- Softmax output layer with cross-entropy loss
- Forward and backpropagation implementation
- Testing of the same on the MNIST dataset

## Layers

### Conv3x3
- 3x3 convolutional filters
- Supports padding and stride parameters
- Xavier initialization for filter weights

### Pool2
- Max pooling with configurable pool size and stride
- Default 2x2 pooling with stride 2

### Softmax
- Fully connected softmax layer for classification
- Includes backpropagation with gradient computation

## Limitations

- Only supports 3x3 filters in convolutional layer
- Single epoch training can be slow
- No validation set evaluation

## Future Tweaks

- Add support for variable filter sizes
- Implement batch processing
- Add data augmentation
- Support for different activation functions
- Add model checkpointing and evaluation on test set
- Performance optimizations

## Installation

Follow these steps to set up the project locally:

- **Clone the repository**
  ```bash
  git clone https://github.com/Tornado025/TorConvNet.git
  cd TorConvNet
  ```

- **Install dependencies**
  ```bash
  pip install numpy tensorflow
  ```

## Usage

Run the training script:
```bash
python cnn.py
```

This will train the CNN on the MNIST dataset for 3 epochs. This is configurable but it takes a lot of time 

## File Structure

- `conv.py` - Conv3x3 convolutional layer implementation
- `pool.py` - Pool2 max pooling layer implementation
- `softmax.py` - Softmax output layer implementation
- `cnn.py` - Main training script

## License

This project is licensed under the MIT License. See the LICENSE file for details.
