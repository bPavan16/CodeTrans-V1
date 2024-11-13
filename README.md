# Code-to-Code Translation from Java to Python using T5-small

## Overview

This project demonstrates a method of translating Java code into Python using the T5-small transformer model, trained on a custom-generated dataset. Unlike previous approaches that rely on language-specific libraries or attributes, this method avoids such inbuilt functionalities, promoting the creation of algorithmically pure, fully translatable code. The dataset used for training was generated using the Gemini model, which translated sample text into pseudocode before coding it into both Java and Python. The resulting dataset was then used to train the T5-small model to learn generalized patterns and structures for code translation.

### Key Features
- **Algorithmically Pure Translation**: The model avoids using language-specific shortcuts, ensuring full translatability.
- **T5-small Model**: A transformer-based model specifically trained for code-to-code translation.
- **Cross-Language Code Translation**: The approach is aimed at translating Java code into Python.
- **Generalized Patterns**: The model learns generalized structures, improving the robustness of code translation.
- **Evaluation**: Results indicate high accuracy in generating semantically equivalent and syntactically correct Python code from Java.

## Dataset

The dataset was generated in the following manner:
1. Sample text was initially translated into pseudocode.
2. The pseudocode was then implemented in both Java and Python, ensuring no language-specific shortcuts were used.
3. This dataset, without any built-in language-specific attributes, was then used to train the T5-small transformer model.

## Model

The model used for code translation is T5-small, a transformer model fine-tuned to learn the patterns and structures required for converting Java code to Python. The model was trained using the dataset described above.

## Evaluation

The model was evaluated on its ability to translate Java code into Python while maintaining semantic equivalence and syntactical correctness. The evaluation results indicate promising performance for use in real-world scenarios.

## Installation

To use this model, follow these steps:

### Requirements
- Python 3.7+
- PyTorch
- Hugging Face Transformers library
- Datasets library from Hugging Face
- Tokenizers library

### Install Dependencies

```bash
pip install torch transformers datasets tokenizers
