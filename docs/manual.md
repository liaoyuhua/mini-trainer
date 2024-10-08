<div align="center">

# 📚 Manual

</div>

## 🤔 Intuition

Currently, there are many libraries that provide ability to train neural networks like PyTorch Lightning, HuggingFace transformers, and other personal projects. If we evaluate them from power and learning curve perspective, they are really located in extreme positions.

Thus, we need a trainer that can be as simple as possible while meeting our needs.

## 🔠 Concepts

### Trainer

4 scenarios:

* train by epoch

* train by iteration

* train with validation

* train without validation

2 ways to train:

* Train from scratch

* Train from checkpoint

### Extension

**Trainer Life Cycle**

* `on_train_start`
* `on_epoch_start`
* `on_batch_start`
* `on_batch_end`
* `on_epoch_end`
* `on_train_end`
* `on_validation_start`
* `on_validation_end`

### Template

## 🛠️ Installation & Usage

## 🌟 Customization


## 🥇 Best Practices

## 👨‍💻 Contributing

First of all, contributing is always welcome. You can contribute to this project in many ways like reporting a bug, requesting an extension or feature, or even implementing an extension (pushing a PR or build a python package on PyPI like `trainer-extension-<name>`).