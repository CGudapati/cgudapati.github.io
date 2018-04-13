---
layout: post
title:  "Storing Sparse Matrices"
date: 2018-04-09
categories: SparseMatrices
use_math: true
---
In this post, we will talk about sparse matrices and the data structures that can be used to store them. We will mainly discuss the Triplet Fomr and the Compressed Column Storage form. 

Let us take an example of a simple sparse matrix with $m = 4$ rows  and $n = 5$ columns:
$$A =   \begin{bmatrix}2 & 0 & 1& 1 & 0\\0  & 1 & 0 & 2 &0\\0 &0 &1 &0 &3\\2 &0 &4  &0 &0 \end{bmatrix} $$























