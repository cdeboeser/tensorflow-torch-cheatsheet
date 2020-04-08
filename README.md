# Tensorflow Torch Cheat Sheet

This repository contains side-by-side comparisons of equal operations in Tensorflow 2.x and Pytorch 1.3+. It is meant to facilitate the transition from in both directions and serve as a quick reference how to do similar things on both frameworks.


## Table of Contents

| Asset | Description |
|:--|:--|
| [Basic Tensor Operations](BasicTensorOperations.ipynb) | Comparing the most fundamental tensor operations like reshaping, splitting, adding, or multiplying matrices |
| [Seq2Seq Comparison](seq2seq_tutorial/ModelTutorial.ipynb) | Walkthrough and comparison of a seq2seq model with Bahdanau attention for both torch and tensorflow with direct comparison |
| [tf.data Dataloader](seq2seq_tutorial/DataloaderTutorial.ipynb) | Example dataloader with tf.data API from (1) generator that reads json-lines files and (2) csv files. |

## Contributing

I am happy about any contributions to extend this cheat sheet. Some todos are:
- Neural network layers
- Custom neural network modules
- Training loops
- Data loading (for various data types)
