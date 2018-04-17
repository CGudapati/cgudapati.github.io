---
layout: post
title:  "Storing Sparse Matrices: C++"
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

The simplest way to represent sparse matrix is by using triplet form. We will simply list the row index column index and the value for each non-zero entry in the matrix $A$

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
It should be noted that  we need **not** have the elements in the triplet format to in sorted format. We clearly see that we need less space to represent the sparse matrix. But we can do better by using Compressed Column Storage \(CCS\). 

### Compressed Column Storage (CCS)

A matrix stored in CCS format needs three vectors to represent it, namely two `<int>` vectors, `col_ptr` and `row_index` and one `<double>` vector, `vals`. The `vals` store all the non-zero values of matrix $A$ in a contiguous piece of memory. We can traverse the matrix $A$ column by column. The size of `vals` vector is `nnz(A)` (number of non-zeros in matrix $A$ ). The size of `row_index` vector is also `nnz(A)`. The `row-index` contains the row index of each non-zero element of $A$.

Going column by column, we can write the `vals` vectors as follows

$$vals = [2,2,1,1,1,4,1,2,3] $$

The corresponding row indexes of each of the above value in `vals` vector is (we are using 0 based index )

$$row\_index = [0,3,1,0,2,3,0,1,2] $$

It should be noted that we have the same values of `row_index` and `vals` vectors in both triplet and CCS format. 

The `col_ptr` vector has the locations in the `vals` vector where each column starts. Its size is `n+1` It can also be thought as the cumulative sum of column counts. The `col_ptr` starts with 0 and since the matrix $A$'s first column has two elements, the second element in `col_ptr` is 2. The next element in `col_ptr` will be `col_ptr[1] + nnz(A(:,2) = 2 + 1 = 3`. The last element of `col_ptr` is `nnz+1`. 

$$col\_ptr = [0, 2, 3, 6, 8, 9]$$
 
The data structure to represent compressed column storage is given below.

~~~ c++
struct SSparseMat{
    int nzmax;  //maximum number of non-zeros entries
    int m;  //number of rows
    int n;  //number of cols
    std::vector<int> col_ptr;
    std::vector<int> row_index;
    std::vector<double> vals;
    int nz;  // flag for triplet or CCS
};
~~~

We can use the above struture to store both the triplet matrices and CCS matrices.



Most of this discussion is based on my readings of the Excellent book by Tim Davis. I encourage eveyone to take a look at the book. 

### References
- - -
1. Tim Davis, [_Direct Methods for Sparse Linear Systems_](Direct Methods for Sparse Linear Systems
), 2006, SIAM 



















