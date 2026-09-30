# Counting Connected Components with Neural Networks

Coursework for the **Foundations of Machine Learning** module, completed as part of my Master's degree — awarded a mark of **96%**. This repository contains the Jupyter notebook with all the code I wrote to complete the assignment tasks and produce the results for the accompanying written report.

## The task

The assignment was to design a synthetic image dataset from scratch and train neural networks to classify images by the number of disconnected shapes they contain, where connectivity is defined by 4-neighbour adjacency (up, down, left, right).

## What the notebook covers

1. **Data generation** — a procedural image generator that draws simple shapes on a 32×32 grid and verifies the label (number of connected components) using 4-connectivity, wrapped in a PyTorch `Dataset` class. Includes sanity checks and visual exploration of the generated data.
2. **Model design** — two models implemented in PyTorch:
   - a fully connected **multi-layer perceptron (MLP)**
   - a **convolutional neural network (CNN)**
3. **Training and evaluation** — training both models on disjoint train/validation/test splits, with training-loss curves, test accuracy comparison, and confusion matrices.
4. **Distribution shift** — constructing shifted test sets (e.g. added salt noise) to probe how well each model generalises beyond the training distribution.
5. **Improving generalisation** — data augmentation plus regularised model variants (dropout and batch normalisation) to improve performance on the shifted data while training only on the original distribution.

## Tools

Python, PyTorch, NumPy, SciPy, Matplotlib — all in a single Jupyter notebook: [`msc-neural-network-classification.ipynb`](msc-neural-network-classification.ipynb)

The notebook is saved with its outputs, so all plots and results render directly on GitHub without needing to re-run it.
