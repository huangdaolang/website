---
title: "Amortized Inference"
draft: false

lightgallery: true
---

Amortized inference leverages neural networks to rapidly approximate and execute inferential tasks by reusing information from past data and simulations. This approach significantly speeds up the Bayesian inference process by training models that can efficiently predict the property of interests without the need for repetitive simulations or computationally intensive optimizations.

**Traditional inference**:
```mermaid
   [Data]  --->  [Complex Computation] ---> [Result]
     |                    |
     v                    v
 [New Data] ---> [Complex Computation] ---> [Result]
     |                    |
     v                    v
    ...                  ...
[More Data] ---> [Complex Computation] ---> [Result]
```


**Amortized inference**:
```mermaid
   [Data]  --->  [Neural Network]  --->  [Result]
```

![Illustration](/amortization.png)