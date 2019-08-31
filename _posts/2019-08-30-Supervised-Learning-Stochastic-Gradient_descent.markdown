---
layout: post
title:  "Implementing Stochastic Gradient Descent in C++ for Supervised Learning on Data with Sparse Features"
date: 2019-08-29
categories: Machine Learning
use_math: true
description: In this post, we will implement Stochastic Gradient Descent  to solve a supervised (binary classification) learning problem. The features are sparse and we will use C++ to implement it. 
header-includes:
---

This post cannot be read on its own. One need to read the previous blogpost about implementing Gradient Descent where we learn how to implement Gradient Descent to solve a supervised (binary classification) learning problem. The coding up of the loss function has been explained in good detail there. The post also talks about how to read the sparse classification datasets into compressed row storage sparse matrices and how to use these data structures to solve the supervised learning problem using Gradient Descent. I assume you have taken a  look at the previous post and I will jump right into implementing the stochastic gradient solver part.

Let us try to use the following steps of **Stochastic Gradient Descent** at an epoch $t$ (as usual we choose the first guess as an all zero vector).

**Step 0:** Given $x^{t}$

**Step 1:**  for iter = 1, ... , m, do the following:  (Remember, there are m samples):

​	   **Step 1a:** choose $i \in \{1, \dots m\} $ randomly

​	   **Step 1b:** compute the gradient of $L$ at the current observation and current $x$. i.e. $g = L'(x:a_{i},y_{i})$  

​	   **Step 1c:** We update the learning rate as follows $\eta = \frac{1}{1+t}$

​	   **Step 1d:** update $x \gets x −\eta (g+\lambda x)$ 

**Step 2**:      $x^{t+1} \gets x$

The above pseudocode is taken (and slightly modified) from lectures notes of Martin Takac.

The SGDSolver.hpp file looks like this

The next blogpost will have the Stochastic Gradient Descent (SGD) Implementation. I will try to compare this with SGD. You can find the full source at [github](https://github.com/CGudapati/BinaryClassification)
