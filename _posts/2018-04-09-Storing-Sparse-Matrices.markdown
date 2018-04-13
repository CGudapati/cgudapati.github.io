---
layout: post
title:  "Storing Sparse Matrices"
date: 2018-04-09
categories: SparseMatrices
use_math: true
description: In this post, we will talk about sparse matrices and the data structures that can be used to store them. 
---
In this post, we will talk about sparse matrices and the data structures that can be used to store them. We will mainly discuss the Triplet Form and the Compressed Column Storage form. 

Let us take an example of a simple sparse matrix with $m = 4$ rows  and $n = 5$ columns:


$$
A =   \begin{bmatrix}2 & 0 & 1& 1 & 0\\0  & 1 & 0 & 2 &0\\0 &0 &1 &0 &3\\2 &0 &4  &0 &0 \end{bmatrix}
$$

We will be using 0-based indexing for coding and 1-based indexing whenever we are discussing linear algebraic notation. 

The simplest way to represet sparse matrix is by using triplet form. We will simply list the row index column index and the value for each non-zero entry in the matrix $A$

<pre>
<b>row_index col_index value </b>
    0         0       2     
    3         0       2
    1         1       1
    0         2       1
    2         2       1
    3         2       4
    0         3       1
    1         3       2
    2         4       3
</pre>
It should be noted that  we need **not** have the elements in the triplet format to in sorted format. 
























