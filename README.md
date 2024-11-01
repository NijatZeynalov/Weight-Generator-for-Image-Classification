# Weight Generator for Image Classification

## Objective
This project provides a framework for compressing neural networks and generating adaptive weights for different network architectures. It contains tools for compressing pre-trained models and retraining them for optimal performance


## 1. Training the Full Model

The train_full_model.py script is used to train a full convolutional neural network on the CIFAR-10 dataset. The script defines a simple CNN architecture with convolutional and pooling layers to classify CIFAR-10 images.

To run the training script:

```
python3 train_full_model.py
```
The trained model weights will be saved to full_model.pth.

Example Output

```
Epoch 1, Loss: 1.9487
Epoch 2, Loss: 1.7352
Epoch 3, Loss: 1.5218
...
Epoch 20, Loss: 0.6354
```

This output shows the loss progression during training.

## 2. Compressing the Model

Once the full model has been trained, you can compress it using the compress_model.py script. This script uses the NetworkCompressor class to reduce the number of parameters in the model, making it suitable for deployment on devices with limited computational power.

To compress and retrain the model:

```
python3 compress_model.py
```
The compressed model weights will be saved to compressed_model.pth.

Example Output

```
Compressing model with 50% compression factor...
Epoch 1, Loss: 1.8234
Epoch 2, Loss: 1.6210
...
Epoch 20, Loss: 0.7234
```

By compressing pre-trained models and retraining them efficiently, this project aims to make deep learning models accessible and deployable in low-resource environments while maintaining high performance.