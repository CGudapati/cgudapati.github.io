---
layout: post
title:  "Storing Sparse Matrices"
date: 2018-04-09
categories: SparseMatrices
use_math: true
---
In this post, we will talk about sparse matrices and the data structures that can be used to store them. We will mainly discuss the Triplet Fomr and the Compressed Column Storage form. 

Let us take an example of a simple sparse matrix:

$$
   \begin{bmatrix}a & b\\c & d\end{bmatrix}
$$
