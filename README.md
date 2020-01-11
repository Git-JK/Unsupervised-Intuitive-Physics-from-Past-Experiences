# Visual Dynamics: Probabilistic Future Frame Synthesis via Cross Convolutional Networks
An implementation based on [TensorLayer](https://github.com/tensorlayer/tensorlayer).

## Current Stage
**The model is not converging and subject to future investigation.**

To test the model, just point the paths in `config.py` to where your data lives, and execute `python3 train.py`. You will see a friendly progress indicator.

By default, this program will log important info per epoch to `result/current/logs`. You can launch a TensorBoard and make use of that log. It will generate one image every 8 epochs.

A new branch is created for debugging model architecture. (While I suspect that it's not an issue related to the architecture.)

## Overall Hierarchy
![demonstration of hierarchy of this paper](hierarchy.jpg)
