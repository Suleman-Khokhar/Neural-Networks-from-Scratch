# Neural Network Fundamentals — From Neurons to Gradient Descent

> **A first-principles study of neural networks using Python and NumPy — built to understand the mathematics, matrix operations, forward propagation, loss, backpropagation, and parameter updates behind modern neural networks.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Learning_Project-8A2BE2?style=for-the-badge)

## Overview

This notebook is a hands-on exploration of **neural network fundamentals from the lowest level upward**.

Instead of starting with a high-level deep-learning framework, the notebook builds intuition around the operations that a neural network performs internally:

**Inputs → Weights → Bias → Dot Product → Layer Output → Activation → Loss → Gradients → Parameter Updates**

The goal is not simply to use a neural-network library, but to understand **why the computations work, how the matrix shapes fit together, and how a model learns by changing its parameters**.

---

## What This Notebook Covers

### 1. Neural Network Parameters

Introduces the two primary learnable parameters of a neuron:

- **Weights** — determine how strongly each input influences the neuron.
- **Biases** — provide a learnable offset to the weighted sum.

The core neuron calculation is:

\[
z = x_1w_1 + x_2w_2 + \dots + x_nw_n + b
\]

A simple Python implementation is used before moving to vectorized NumPy operations.

---

### 2. Single Neuron / Perceptron

The notebook starts with individual neuron calculations to make the underlying computation explicit.

Example structure:

```text
Inputs
  │
  ├── x₁ × w₁
  ├── x₂ × w₂
  ├── x₃ × w₃
  │
  └────────────── + bias
                     │
                     ▼
                  output
```

This establishes the foundation for understanding larger dense layers.

---

### 3. Dot Products

The notebook connects the manual neuron calculation to the mathematical concept of a **dot product**.

For example:

\[
X \cdot W =
(x_1w_1) + (x_2w_2) + \dots + (x_nw_n)
\]

This is then implemented with NumPy:

```python
np.dot(weights, inputs)
```

The notebook explicitly relates the NumPy operation back to the individual multiplications and additions performed by a neuron.

---

### 4. Matrix Shapes

A major focus of the notebook is understanding **matrix dimensions and compatibility**.

It explains:

- Rows and columns
- Matrix shapes
- Weight matrices
- Input batches
- Transposition
- Matrix multiplication
- Bias broadcasting
- Why the inner dimensions of matrix multiplication must align

For example:

```text
Input batch       Weights
(3, 4)      ×     (4, 5)
                         │
                         ▼
                    Output
                     (3, 5)
```

Understanding these shapes is essential for debugging neural-network code and moving from single examples to batches of data.

---

### 5. Batch Processing

The notebook progresses from one input vector to multiple input samples.

Instead of processing:

```text
one sample → one calculation
```

it works with:

```text
multiple samples → matrix operations → multiple outputs
```

This introduces the practical structure used by neural-network implementations.

The notebook demonstrates matrix multiplication with transposed weights when working with the initial representation, then moves toward a cleaner object-oriented dense-layer implementation where the weight matrix is stored in a multiplication-friendly shape.

---

### 6. Building Dense Layers

A reusable `Layer_Dense` class is implemented with NumPy.

The class contains:

- Randomly initialized weights
- Zero-initialized biases
- A forward-pass method
- Matrix multiplication
- Bias addition

Core structure:

```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
```

The notebook then connects multiple dense layers:

```text
Input
(3, 4)
   │
   ▼
Dense Layer 1
4 inputs → 5 neurons
   │
   ▼
(3, 5)
   │
   ▼
Dense Layer 2
5 inputs → 2 neurons
   │
   ▼
(3, 2)
```

---

## Activation Functions

The notebook introduces activation functions as the mechanism that adds **non-linearity** to a neural network.

It discusses:

- **Sigmoid**
- **Tanh**
- **ReLU**
- **Softmax**

### Sigmoid

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The notebook works through a concrete example where a negative weighted sum produces a probability close to zero.

It also discusses common uses of sigmoid, including binary classification and gating mechanisms.

### ReLU

\[
\text{ReLU}(z)=\max(0,z)
\]

The notebook explains why non-linear activations are important: without them, stacking linear layers would still result in a linear transformation and would severely limit what the network can represent.

---

## Forward Propagation

The notebook connects the individual concepts into a forward pass:

```text
Input Data
    │
    ▼
Weighted Sum
    │
    ▼
Bias Addition
    │
    ▼
Activation
    │
    ▼
Next Layer
    │
    ▼
Prediction
```

The output of one layer becomes the input to the next layer.

---

## Loss and Cross-Entropy

After producing a prediction, the notebook introduces the concept of measuring how wrong that prediction is.

For binary classification, it discusses cross-entropy loss:

\[
L(a,y)=-y\log(a)-(1-y)\log(1-a)
\]

The notebook demonstrates how the true label determines which part of the expression contributes to the loss.

This establishes the connection between:

```text
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Parameter Update
```

---

## Backpropagation

The notebook then moves from forward computation to the mathematics behind learning.

It derives the gradient of the loss with respect to the prediction:

$$
\frac{\partial L}{\partial a}
=
-\frac{y}{a}
+
\frac{1-y}{1-a}
$$

It continues toward the familiar binary-classification result:

$$
dz = a-y
$$

The gradients with respect to the weights and bias are:

$$
dw_i = (a-y)x_i
$$

$$
db = a-y
$$

These gradients tell the model how its parameters contributed to the prediction error.

---

## Gradient Descent

The notebook concludes the learning loop by introducing parameter updates:

$$
w_i := w_i - \alpha \frac{\partial J}{\partial w_i}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

where:

- $w_i$ = weight
- $b$ = bias
- $\alpha$ = learning rate
- $J$ = cost function

The model moves in the opposite direction of the gradient because the goal is to **minimize the cost function**.
The central learning cycle is summarized as:

```text
        ┌───────────────┐
        │   Input Data  │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ Forward Pass  │
        │ Weights+Bias  │
        └───────┬───────┘
                ▼
           Prediction
                │
                ▼
        ┌───────────────┐
        │     Loss      │
        │ "How wrong?"  │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ Backpropagation│
        │   Gradients   │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │    Update     │
        │ Weights/Biases│
        └───────┬───────┘
                │
                └──────► Repeat
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **NumPy** | Vectorized numerical and matrix operations |
| **Jupyter Notebook** | Interactive learning and experimentation |
| **LaTeX / Markdown** | Mathematical explanations and documentation |

---

## Notebook Structure

The learning progression is intentionally layered:

```text
01. Neural Network Introduction
        ↓
02. Parameters: Weights & Biases
        ↓
03. Single Neuron
        ↓
04. Dense Layer
        ↓
05. Matrix Shapes
        ↓
06. Dot Products
        ↓
07. Batch Processing
        ↓
08. Multiple Layers
        ↓
09. Dense Layer Class
        ↓
10. Activation Functions
        ↓
11. Forward Pass
        ↓
12. Loss / Cross-Entropy
        ↓
13. Backpropagation
        ↓
14. Gradient Descent
```

---

## Running the Notebook

### 1. Clone the repository

```bash
git clone https://github.com/Suleman-Khokhar/neural-network-fundamentals-from-scratch.git
cd neural-network-fundamentals-from-scratch
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / Kali

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy jupyter
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
Intro_and_Neuron_Code.ipynb
```

and run the cells from top to bottom.

---

## Learning Outcomes

After working through this notebook, you should be able to explain:

- What weights and biases represent
- How a single neuron calculates an output
- How a dot product relates to neuron computation
- Why matrix shapes matter
- How batches of samples are represented
- How dense layers perform forward propagation
- Why transposition is sometimes required
- How multiple layers are connected
- Why activation functions introduce non-linearity
- How sigmoid converts a raw score into a probability
- How cross-entropy measures prediction error
- What gradients represent
- How backpropagation provides parameter gradients
- How gradient descent updates weights and biases

---

## Design Philosophy

This notebook follows a **from-scratch, concept-first approach**.

Rather than hiding neural-network operations behind a high-level framework, the implementation deliberately exposes the underlying mechanics:

> **Understand the multiplication before using the matrix.  
> Understand the matrix before using the layer.  
> Understand the layer before using the network.  
> Understand the gradient before trusting the training process.**

This makes the notebook suitable as a foundation for progressing toward more advanced topics such as optimizers, regularization, convolutional neural networks, and full neural-network training implementations.

---

## Current Scope

This notebook primarily focuses on **fundamentals, forward computation, mathematical intuition, and the derivation of learning concepts**.

It should not be presented as a complete deep-learning framework or a production-ready training system. The emphasis is understanding the mechanics that later implementations build upon.

---

## Next Steps

Planned natural extensions include:

- [ ] Implement ReLU in Python
- [ ] Implement Softmax in Python
- [ ] Implement loss functions directly in NumPy
- [ ] Implement full backpropagation in code
- [ ] Implement gradient descent training
- [ ] Add trainable parameters
- [ ] Add accuracy evaluation
- [ ] Train a small classification model
- [ ] Implement optimizers
- [ ] Build a complete neural network from scratch

---

## Author

**Suleman Khokhar**

Computer Science Student · Neural Networks · AI · EdTech

GitHub: [@Suleman-Khokhar](https://github.com/Suleman-Khokhar)

---

## License

This project is intended primarily for **learning, experimentation, and educational reference**.

If you build upon the material, attribution is appreciated.
