---
layout: post
title:  "Implementing Gradient Descent in C++ for Supervised Learning on Data with Sparse Features"
date: 2019-08-29
categories: SparseMatrices
use_math: true
description: In this post, we shall learn how to implement Gradient Descent to solve a supervised (binary classification) learning problem. The features are sparse and we will use C++ to implement it.
header-includes:
---

It has been very long and in this post, we shall learn how to implement Gradient Descent to solve a supervised (binary classification) learning problem. The features are sparse and we will use C++ to implement it. The focus is going to be more on the implementation side and less on the mathy side. So, at times I might do things in a handwavy manner but I will provide links on where you can get more (and better) information. I presume you have some rudimentary knowledge about Supervised Learning, Optimization and Gradient Descent. I will still go through them quickly for the sake of completeness.

**Supervised Learning:** In simplest terms, supervised learning involves trying to find a mapping function, which can predict the output based on the input. We find the function based on a set of labeled data (input: a vector of features-output: some sort of label). For simplicity's sake, we assume we are only looking at two  output labels. Essentially, we are trying to predict on of the output from the given set of input features. We learn the mapping function by looking at the already labelled data called "Training Data". Then once the function has been found, we can vouch for its credibility by using the function to predict the output from the input on data  which has been not used in training  and is called "Testing Data".   The [Supervised Learning Wikipedia](https://en.wikipedia.org/wiki/Supervised_learning) page is pretty good for more information.



**Optimization:** In the most simplistic terms, optimization involves finding the best value of a function over a given domain. Let us define a few terms. The first is **Loss Function**. The Loss function $L(f(x), y)$  tells us how much we "lose" by predicting the output $f(x)$ instead of the true output $y$.  A lot has been spoken about [Empirical Risk Minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization) is the literature and I won't spend any time here. We use the Logistic Loss function to minimize the empirical risk.  

Our data looks like this 

$$
y : A
$$

where $y$ is an $m$-dimensional vector of labels and $A$ is a collection of $m$ observations with each observation having $n$ features. Let $a_{kj}$ be the $k$th observation's $j$th feature. 

 The Logistic Loss is given by:

$$
L(x; a,y) = \log(1+\exp(-yx^{T}a)),
$$

where $x$ is the set of variables (some folks might use the variable name $w$) which we are trying to optimize, $y$ is the label of an observation, and $a$ is the feature vector of that particular observation.  Now when we have $i = m$ samples, $ (y_{i}: a_{i}) $, where $a_{i} \in R^{n}$ and $y_i \in {(+1, -1)}$, our average loss can be obtained by summing up the individual losses for each sample and then dividing it my $m$. We will try to minimize this average loss. 


$$
\min_{x \in R^{n}} P(x) = \frac{1}{m}\sum_{i=1}^{m}L(x;a_i,y_i)).
$$


We usually add a regularization term to prevent overfitting and more can be learnt at the wikipedia article on [Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)). So the final optimization problem is:

$$
\min_{x \in R^{n}} P(x) = \frac{1}{m}\sum_{i=1}^{m}L(x;a_i,y_i)) + \frac{\lambda}{2}||x||^2
$$

This is an unconstrained optimization problem and there are numerous algorithms to solve it. A simple one is Gradient Decent. Before we go into the details of Gradient Descent, let us talk about the general framework of an optimization algorithm. Consider the following  general optimization problem. 

$$
\min f(x) \\ \text{s.t.}\quad  x \in \mathcal{F}
$$


where $\mathcal{F}$ is a convex set of feasible solutions.  When $\mathcal{F}$ is $\mathbb{R}^n$, then we have an unconstrained optimization problem. 

**Step 0:** Given $x^{0}$ is the first feasible solution.

**Step 1:** Find a search direction, $s^{k}$ which satisfies $\delta f(x^{k}, s^{k}) < 0$, where $\delta f(x^{k}, s^{k}). $ is the directional derivative: $\delta f(x^{k}, s^{k})= \nabla f(x)^{T}s$. If we cannot find such a direction, then **STOP!** Optimum found.

**Step 2:**  Perform a 1 dimensional **Line Search** to obtain that $\alpha^{k}$ which minimizes $f(x^{k} + \alpha_{k} s^{k})$

**Step 3:** Update $x^{k}$ as $x^{k+1} \gets x^{k} + \alpha_{k}s^{k}$ and $k \gets k+1$

**Step 4:** Check for stopping criteria and if they are met, **STOP**, else, **GOTO** **Step 1**

The above steps are taken and modified from the excellent and free "Nonlinear Optimization" text book written by my advisor and his colleagues (E. De Klerk, C. Roos, **T. Terlaky**)

Let us try to use the above framework for **Gradient Descent**.

**Step 0:** Let us choose $x^{0} = \mathbf{0}$ (an all 0 vector)  as the first guess

**Step 1:** For the directional derivative, $\delta f(x^{k}, s^{k})$ to be strictly less than 0, we can choose $s^{k} = -\nabla f(x^{k})$. For first iteration , it would be $ s^{0} = -\nabla f(x^{0})$ and    $\delta f(x^{k}, s^{k}) = \nabla f(x^{0})^{T}(-\nabla f(x^{0})) = -\lvert\lvert\nabla f(x^{0})\rvert\rvert^{2} < 0$ 

**Step 2:** Now we know that if we go in the direction of negative gradient, we can reduce the value of the objective function. But how much should be advance? Too little, we might not get to the optimal solution fast enough. Too much, we might overplay our hands and miss the optimum. So, we need to find the best possible step length, which as you have guessed it sets up another optimization problem and it is called Line Search. Though this is just a 1-dimensional optimization problem, depending on the method we choose to optimize we might have to do a lot extra calculations like say objective function evaluations.  Can we get the convergence for free? The answer is **Yes**, but under some specific conditions -- namely the gradient of $P(x)$ is Lipschitz continuous. [Mark Schmidt](https://www.cs.ubc.ca/~schmidtm/Courses/540-W18/L4.pdf) at UBC explains it much better than I can do. And luckily, P(x)'s gradient is Lipschitz continuous. So what does this mean for the gradient descent? It means that we have a guarantee for convergence when the step size is constant $\alpha_{k} = \frac{1}{L}$ where $L$ is the Lipschitz's constant (largest Eigen value of the Hessian of $P'(x)$ for all values of $x$  also, not very important for the implementation). Now, we are getting the exact amount to move for a guaranteed decrease in objective function value. 

**Step 3**:      $$x^{k+1} \gets x^{k} -\alpha_{k}\nabla f(x^{k})$$ 

**Step 4:** The stopping condition could be $\lvert\lvert\nabla f(x^{k})\rvert\rvert \le \epsilon$ 

Now, Let us get to the exciting world of coding!

We shall use the [ParseSVM](cgudapati.com/sparsematrices,/svm,/libsvm,/machinelearning/2019/01/19/A-Simple-C++-Library-to-Parse-LIBSVM-files.html) to read the classification data. The ParseSVM has undergone some changes since that post was written, namely normalization of each observation and the return data-structure names. The updated code can be found at [github](https://github.com/CGudapati/ParseSVM). 

Let us refresh out memory about Sparse Matrices -- specifically about the Compressed Row Storage. Let us assume that we have a sparse matrix $A$ with no special structure. We then create 	3 vectors: $values$, $col\_index$, and $row\_ptr$. The $values$ vector is  stores the non-zero `floating point`  elements of $A$ taken row wise.   The $col\_index$ or column index vector is an `int` vector which contains the position of each element in $values$ vector and the $row\_ptr$ or row pointer vector stores the positions in $col\_index$ where a new row starts.  More information can be found on [netlib](http://netlib.org/linalg/html_templates/node91.html)

Let us take an example of a simple sparse matrix, $B$ with $m = 4$ rows  and $n = 5$ columns:
$$
B =   \begin{bmatrix}2 & 0 & 1& 1.2 & 0\\0  & 1 & 0 & 2 &0\\0 &0 &1.3 &0 &3\\2 &0 &4  &0 &0 \end{bmatrix}
$$

We will be using 0-based indexing for coding and 1-based indexing whenever we are discussing linear algebraic notation. 

Let us traverse $A$ row by row and add it to the $values$ vector.

$$values = [2,1,1.2,1,2,1.3,3,2,4] $$

The corresponding column indexes of each of the above value in $values$ vector is (we are using 0 based index )

$$col\_index = [0,2,3,1,3,2,4,0,3] $$

The $row\_ptr$ vector has the locations in the $values$  vector where each column starts. Its size is $n+1$. The $row\_ptr$ starts with 0 and since the matrix $A$'s first row has three elements, the second element in $row\_ptr$ is 3. The next element in $row\_ptr$  will be  $ 3 + 2 = 5$.  

$$row\_ptr = [0, 3, 5, 7, 9]$$

Let us iterate through a row: say, we choose the third row **(i = 3)**, then we can do the following:

~~~c++
for (k = B.row_ptr[i]; k < B.row_ptr[i+1]; k++){
 	std::cout << B.values[k] << " "   
}
~~~

The above snippet of code will produce `2 4` as output 

```c++
for (k = B.row_ptr[i]; k < B.row_ptr[i+1]; k++){
 	std::cout << B.col_index[k] << " "   
}
```

The above snippet of code will produce `0 2`  as output . The above two snippets will be key to implementing the sparse updates. We are saying that row 3 has non-zero numbers at indices $0, 2$ and they respectively are $2, 4$

The ParseSVM can load the the libsvm file. It is a very simple call (provided one has the libsvm file properly encoded).  

 

```c++
const std::string file_path = "data/enron.libsvm";
    
    Classification_Data_CRS A;
    //We will store the problem data in variable A as compressed row storage and the data is going to be normalized
    get_CRSM_from_svm(A, file_path);
```

Let us create a LogLoss Function namespace which contains all the methods that will be used for solving the classification problem. This loss function class can be used by other optimization algorithms (we do use this loss function in the next post for stochastic gradient descent ). 

~~~c++
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>

#include "../ParseSVM/Matrix.h"
#include "../helper.h"

namespace LogLoss {
    
    //This function calculates the A*x vector
    inline void compute_data_times_vector( const Classification_Data_CRS&  A, const std::vector<double>& x, std::vector<double> & ATx){...}
    
    
    // This function calculates the objective value
    inline double compute_obj_val(const std::vector<double> &ATx, const Classification_Data_CRS &A, const std::vector<double>& x,  double lambda){...}
    
    
    
    //Gradient computation at a given point, x
    inline void compute_grad_at_x(const std::vector<double> &ATx, const Classification_Data_CRS &A, const std::vector<double> &x, double lambda, std::vector<double> &grad){..}
    
    
    //Computing the training error
    inline double compute_training_error(const Classification_Data_CRS& A, const std::vector<double>& ATx){...}
    
}
~~~

There are four functions that have been implemented here. We  will go through them one by one in some detail

~~~c++
   1  inline void compute_data_times_vector( const Classification_Data_CRS&  A, const std::vector<double>& x, std::vector<double> & ATx){
   2     
   3     std::fill(ATx.begin(), ATx.end(), 0.0);  //Setting the ATx vector to 0;
   4 
   5     //The number of elements in ATX is the same as number of observations in the original data.
   6     
   7     for(auto i = 0; i < A.m; ++i){
   8         for( auto j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j){
   9             ATx[i] += A.values[j]*x[A.col_index[j]];
  10         }
  11     }
  12     
  13 }
	
~~~

We shall set `ATx` ​vector to zero so that it can be filled with the values of  $A*x$. Since $A$ is the feature vector of $m$ samples, it has the dimension of $m\times n$ and $x$ has the dimension of $n\times1$ and hence we can see why `ATx` is a $m\times 1$ vector.  In line 7 of the above code snippet, we iterate though every single sample and for each sample we get the indices where the non-zeros will be present  (Line 8 from the `A.row_ptr` values) and then multiply them with the numbers present at the same index in $x$ as the other positions in that particular observation are 0 anyway. The best way to do this is to do it by hand on a small toy example to see what is happening.

The next function is implemented here

```c++
   1 inline double LogLoss::compute_obj_val(const std::vector<double> &ATx, const Classification_Data_CRS &A, const std::vector<double>& x,  double lambda){
   2     
   3     //The objective function is
   4     
   5     //      1    m             -yi*ai*x      λ       2
   6     //     --- * Σ  log(1 + e ^       )   + -—-*||x||
   7     //      m   i=1                          2
   8     //  where ai is the i_th row of the A matrix
   9     
  10     //Set everything to 0
  11     double obj_val = 0.0;
  12     
  13     //calculating the term 1
  14     for(auto i = 0; i < A.m; i++){  //This is the Σ_i=1..m
  15         obj_val += log(1+exp(-1*A.y_label[i]*ATx[i]));
  16     }
  17     
  18     obj_val /= A.m;  //dividing by 1/m
  19     
  20     obj_val += 0.5*lambda*l2_norm_sq(x);
  21     
  22     return obj_val;
  23 }
```

The regularized objective function is 

$$
\min_{x \in R^{n}} P(x) = \frac{1}{m}\sum_{i=1}^{m}\log(1+\exp(-y_{i}x^{T}a_{i})) + \frac{\lambda}{2}||x||^2
$$

We can see that we are summing $m$ terms and then dividing by and finally adding the regularization term. So we will do things in that order. In Line 11, we initialize `obj_val` to 0 as we accumulate the value of the objective function after considering each sample. Now for an $i^{th}$ sample, we need to calculate the $\log(1+\exp(-y_{i}x^{T}a_{i}))$ and we know that $x^{T}a_{i}$  or $a_{i}^{T}x$ is the $i^{th}$ coordinate of `ATx​` vector. Line 14 and 15 will get us the sum of all the log losses for each sample. Then we have to divide by $m$ which is achieved in Line 18 and then we add the value of the regularization in line 20. The objective function is finished. 

Now let us move on to the gradient. 

~~~cpp
   1 void LogLoss::compute_grad_at_x(const std::vector<double> &ATx, const Classification_Data_CRS &A, const std::vector<double> &x, double lambda, std::vector<double> &grad){
   2     
   3     // Set the gradient to zero as you have to caclulate it from scratch. The gradient is a vector of n (number of features) elements.
   4     // We loop through each row in the Data matrix and update the correspondong coordinates of the gradient.
   5     // i.e say the first row of the data matrix has non zeros in a11, a13 and a1n
   6     // positions: we then update the graident in only those three corordinates.
   7     // After we are though all the rows, then we will scale it by m and then add the "x" vector.
   8     
   9     std::fill(grad.begin(), grad.end(), 0.0);  //Setting the gradient vector to 0;
  10     
  11     //We go through each row in the A matrix and then update the corresponding coordinates in the gradient. We are not buildning corordinate by corrdinate gradient.
  12     for (auto i = 0; i < A.m; i++) {
  13         auto accum = exp(-1*A.y_label[i]*ATx[i]);
  14         accum = accum/(1.0 + accum);
  15         for(auto j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j){
  16             auto temp = -1.0*accum*A.values[j]*A.y_label[i];
  17             grad[A.col_index[j]] += temp;
  18         }
  19     }
  20 
  21 //                                                    1                                               1
  22 //Using c++ lambdas to scale the current gradient by ---, scale x by λ and add them together: grad = --- * gradient +  λ*x
  23 //                                                    m                                               m
  24 // This is a classic daxpy operation
  25     
  26 //This finishes the gradient
  27     
  28     std::transform(grad.begin(), grad.end(), x.begin(), grad.begin(),[&A, &lambda](auto grad_i, auto x_i){return (1.0/A.m)*grad_i + lambda*x_i;});
  29     
  30 }

~~~

We need to do a sparse gradient update.  Though it could be trivial for most folks, the below process helped me a few years ago when I was implementing the sparse gradient update. 

Let us start by writing out the  expanded form of the $G(x)$. I left out the regularization term for now.

$$
G(x) = \log(1+e^{-y_1x^{T}a_{1}}) + \log(1+e^{-y_2x^{T}a_{2}})+ ... + \log(1+e^{-y_mx^{T}a_{m}})
$$

Let us expand those terms now.

$$
G(x) = \log(1+e^{-y_{1}(a_{11}x_{1} + a_{12}x_{2} + ... + a_{1n}x_{n})})+ \log(1+e^{-y_{2}(a_{21}x_{1} + a_{22}x_{2} + ... + a_{2n}x_{n})}) + ... + \log(1+e^{-y_{m}(a_{m1}x_{1} + a_{m2}x_{2} + ... + a_{mn}x_{n})})
$$

To get the gradient, we have to take the partial derivative with respect to each and every coordinate of $x$.

$$
\frac{\partial G}{\partial x_{1}} = \frac{1}{1+e^{-y_{1}x^{T}a_{1}}}.e^{-y_{1}x^{T}a_{1}}.-y_{1}a_{11} + \frac{1}{1+e^{-y_{2}x^{T}a_{2}}}.e^{-y_{2}x^{T}a_{2}}.-y_{2}a_{21} + ... + \frac{1}{1+e^{-y_{m}x^{T}a_{m}}}.e^{-y_{m}x^{T}a_{m}}.-y_{m}a_{m1}
$$

more compactly,

$$
\frac{\partial G}{\partial x_{1}} = \frac{e^{-y_{1}x^{T}a_{1}}}{1+e^{-y_{1}x^{T}a_{1}}}.-y_{1}a_{11} + \frac{e^{-y_{2}x^{T}a_{2}}}{1+e^{-y_{2}x^{T}a_{2}}}.-y_{2}a_{21}+ ... + \frac{e^{-y_{m}x^{T}a_{m}}}{1+e^{-y_{m}x^{T}a_{m}}}.-y_{m}a_{m1}
$$
we have,
$$
\begin{bmatrix}
\frac{\partial G}{\partial x_{1}}\\ \frac{\partial G}{\partial x_{2}}\\.\\.\\.\\ \frac{\partial G}{\partial x_{n}}
\end{bmatrix}
= 
\begin{bmatrix}
\frac{e^{-y_{1}x^{T}a_{1}}}{1+e^{-y_{1}x^{T}a_{1}}}.-y_{1}a_{11} + \frac{1}{1+e^{-y_{2}x^{T}a_{2}}}.e^{-y_{2}x^{T}a_{2}}.-y_{2}a_{21} + ... + \frac{1}{1+e^{-y_{m}x^{T}a_{m}}}.e^{-y_{m}x^{T}a_{m}}.-y_{m}a_{m1} \\
\frac{e^{-y_{1}x^{T}a_{1}}}{1+e^{-y_{1}x^{T}a_{1}}}.-y_{1}a_{12} + \frac{1}{1+e^{-y_{2}x^{T}a_{2}}}.e^{-y_{2}x^{T}a_{2}}.-y_{2}a_{22} + ... + \frac{1}{1+e^{-y_{2}x^{T}a_{m}}}.e^{-y_{2}x^{T}a_{m}}.-y_{2}a_{m2}\\
.\\
.\\
.\\
\frac{e^{-y_{1}x^{T}a_{1}}}{1+e^{-y_{1}x^{T}a_{1}}}.-y_{m}a_{1n} + + \frac{1}{1+e^{-y_{2}x^{T}a_{2}}}.e^{-y_{2}x^{T}a_{2}}.-y_{2}a_{2n} + ... + \frac{1}{1+e^{-y_{m}x^{T}a_{m}}}.e^{-y_{m}x^{T}a_{m}}.-y_{m}a_{mn}
\end{bmatrix}
$$


Now we need to add the vector $\lambda x$  to the $\partial G$ to get the final gradient ($\frac{d}{dx}\frac{\lambda}{2}\lvert\lvert x\rvert\rvert^2 = \lambda x$) 

I tried to keep the above gradient calculation formula as error free as possible but I might have still made an error. The code works correctly anyway. 

So we need to create the above vector. The below code snippet does exactly that. It is fairly well commented.


```c++
   1 inline void compute_grad_at_x(const std::vector<double> &ATx, const Classification_Data_CRS &A, const std::vector<double> &x, double lambda, std::vector<double> &grad){
   2         
   3         // Set the gradient to zero as you have to caclulate it from scratch. The gradient is a vector of n (number of features) elements.
   4         // We loop through each row in the Data matrix and update the correspondong coordinates of the gradient.
   5         // i.e say the first row of the data matrix has non zeros in a11, a13 and a1n
   6         // positions: we then update the graident in only those three corordinates.
   7         // After we are though all the rows, then we will scale it by m and then add the "x" vector.
   8         
   9         std::fill(grad.begin(), grad.end(), 0.0);  //Setting thegradient vector to 0;
  10         
  11         //We go through each row in the A matrix and then update the corresponding coordinates in the gradient. We are not buildning corordinate by corrdinate gradient.
  12         for (auto i = 0; i < A.m; i++) {
  13             auto accum = exp(-1*A.y_label[i]*ATx[i]);   
  14             accum = accum/(1.0 + accum); // This calculates the terms with e^{-y_ixTa}
  15             for(auto j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j){
  16                 auto temp = -1.0*accum*A.values[j]*A.y_label[i];
  17                 grad[A.col_index[j]] += temp;      //Updating only those coordinates of the gradient where there are non zeros in a_{i}
  18             }
  19         }
  20         
  21         //                                                    1                                               1
  22         //Using c++ lambdas to scale the current gradient by ---, scale x by λ and add them together: grad = --- * gradient +  λ*x
  23         //                                                    m                                               m
  24         // This is a classic daxpy operation
  25         
  26         //This finishes the gradient
  27         
  28         std::transform(grad.begin(), grad.end(), x.begin(), grad.begin(),[&A, &lambda](auto grad_i, auto x_i){return (1.0/A.m)*grad_i + lambda*x_i;});
  29         
  30     }

```

It is assumed that we get the value of `ATx` vector before we calculate the gradient. Alternatively, we can do that calculation in this function too. So we have the various values of $x^{T}a_{i}$ (or  $a_{i}^{T}x$)  in the vector `ATx`.  We then iterate through the rows of A and change the corresponding coordinates in the gradient when we encounter a non zero in the feature vector of an observation. And finally we do the daxpy operation of adding two different vectors using some lambdas. Further lines 16 and 17 can be combined (which I hope the compiler already does)

We now code the training error function. It will return the fraction of wrong predictions made by the model. As we run more and more epochs, the training error should ultimately reduce. 

~~~ c++
   1 inline double compute_training_error(const Classification_Data_CRS& A, const std::vector<double>& ATx){
   2         
   3         // We calculate the training errors as follows
   4         // When a new observation is provided, we apply the logistic function
   5         //
   6         //                1
   7         //     f(z) = ------------
   8         //                   -z
   9         //            (1 + e^  )
  10         // where z = a_i*x and a_i is the feature vector of the new observation
  11         
  12         // If f(z) >= 0.5, then its predicted label is +1 and f(z) < 0.5, the predicetd label is -1;
  13         // More information about logistic regression can be found on wikipedia
  14         
  15         //As the algorithm progresses, we hope that the training error reduces.
  16         
  17         
  18         double train_error = 0.0;
  19         
  20         std::vector<int> z(ATx.size(), 0);  //This will hold the predictions
  21         
  22         double prediction = 0.0;
  23         int corrent_predictions = 0;
  24         
  25         for (std::size_t i =0; i < ATx.size(); ++i) {
  26             prediction = 1.0/(1.0 + exp(-1.0*ATx[i]));
  27             if (prediction >= 0.5){
  28                 z[i] = 1;
  29             }
  30             else{
  31                 z[i] = -1;
  32             }
  33             
  34             if (A.y_label[i]== z[i]) {
  35                 corrent_predictions++;
  36             }
  37         }
  38         
  39         return 1.0-(corrent_predictions*1.0/(A.m)); // total number of correct predictions/ total number of observations.
  40         
  41         return train_error;
  42     }

~~~



The above code is fairly straightforward. Find the value of $f(z)$ for every observation with the correct value of $x$ and then see if it >= 0.5 or < 0.5. Update the corresponding counts and then calculate the training error. 

Now we have finished the loss function. Let us dive into the solvers.

We define a `CoreSolver` which has all the necessary functions that need to implemented. The following code's OOP nature can be further improved.

The `CoreSolver.hpp` is 

~~~ c++
#ifndef CoreSolver_hpp
#define CoreSolver_hpp

#include <stdio.h>
#include <vector>
#include <iomanip>
#include <iomanip>

#include "../ParseSVM/Matrix.h"
#include "../LossFunctions/LogLoss.h"

class CoreSolver{
public:
//    CoreSolver() {}
//    virtual ~CoreSolver() {}
    virtual void init(const Classification_Data_CRS &A ,double lam, double alfa, int max_iter) = 0; //Implement this in the child classes
    virtual void run_solver(const Classification_Data_CRS& A) = 0; //Implement this in the child classes
    virtual void run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad) = 0;
    virtual double get_vector_norm(const std::vector<double>& v);
};


#endif /* CoreSolver_hpp */
~~~

As you can see, there are 4 functions that have to implemented. We will only implement the last function which gets the vector norm in the `CoreSolver.cpp` and let the child classes implement the other functions.

```c++
#include "CoreSolver.hpp"

double CoreSolver::get_vector_norm(const std::vector<double>& v){
    double accum = 0.0;
    
    for( double x : v){
        accum += x*x;
    }
    return sqrt(accum);
}
```

The above function is fairly straightforward. We get the norm of a vector by multiplying each element and then storing the result in `accum` and then taking the square root. 

Now let us take a look at the `GradientDescent` which is a child class of `CoreSolver`

The `GradientDescent.hpp` looks like this:

```c++
#ifndef GradientDescent_hpp
#define GradientDescent_hpp

#include <stdio.h>
#include "CoreSolver.hpp"

class GradientDescent : public CoreSolver{
public:
    
    std::vector<double> x;
    std::vector<double> grad;
    std::vector<double> ATx;
    double lambda;
    double alpha;
    int iters;
    
    virtual void init(const Classification_Data_CRS &A ,double lam, double alfa, int max_iter);
    virtual void run_solver(const Classification_Data_CRS& A);
    virtual void run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad);
    
    
};

#endif /* GradientDescent_hpp */
```

I didn't pay too much heed to the public-private nature of the member variables and functions (mainly because this is for private use and  I am slightly lazy)

The following are the steps for running the **gradient descent algorithm**.

Let $f(x)$ be a convex, smooth and differentiable function. The following general steps will get you the optimal solution.*

Given $f(x)$ and $ x_0$

**Step 1:**  Find a descent direction (search direction) $s = -\nabla P(x)$

​		If no such direction exists, STOP!!!

**Step 2:** Update $x := x + \frac{1}{L} s$

**Step 3:** Check for stopping conditions; if they are met, then you can stop. if not continue steps 1-2

The first init function just initializes the various values of the solver

```c++
void GradientDescent::init(const Classification_Data_CRS &A, double lam, double alfa, int max_iter){
    x.resize(A.n, 0); //We are creating a vector of all 0s of size n (number of features)
    grad.resize(A.n, 0); //The gradient also is of size n
    ATx.resize(A.m, 0); //This vector holds the value of A*x. (T is times and not transpose)
    lambda = lam;
    alpha = alfa;
    iters = max_iter;
}
```

`x` is the variable that we are optimizing (in hindsight, I should have called it `w`). `grad` is the variable which stores the gradient vector. `ATx` has the matrix vector product of $A$ and $x$. $\lambda$ is the regularization factor and $L$ is the Lipschtz constant for step size. Now onto the`run_ solver`.

```c++
       1 void GradientDescent::run_solver(const Classification_Data_CRS &A){
   2     //Let us first set up the variables
   3     LogLoss::compute_data_times_vector(A, x, ATx);
   4     double obj_val =  LogLoss::compute_obj_val(ATx, A, x, lambda);
   5     double train_error = LogLoss::compute_training_error(A, ATx);
   6     
   7     //Setting up the output that would be visible on screen"
   8     std::cout << "   Iter  " <<  "    obj. val  " << "     training error "  <<  "\n";
   9     std::cout << std::setw(10) << std::left << 0 << std::setw(20) << std::left << std::setprecision(10) << obj_val << std::setw(20) << std::left << train_error  << "\n";
  10     
  11     // Now we have to update x. From the general algorithm, we have to find a descent direction,  perform a line search to get
  12     // the appropriate value of α and the update the value of x. In gradient descent, the negative gradient vector will
  13     // always give you the descent direction.  We will use a constant step size of 1/L (L is Lipschiz constant)
  14     //                                                      1
  15     // We will use std::transform to do this job: x := x - ---*g; (simple daxpy operation)
  16     //                                                      L
  17     
  18     //Now let us run the solver for all the iterations
  19     for(int k = 1; k <= this->iters; k++){
  20         
  21         this->run_one_iter(A, x, ATx, grad);
  22         
  23         LogLoss::compute_data_times_vector(A, x, ATx);
  24         double obj_val =  LogLoss::compute_obj_val(ATx, A, x, lambda);
  25         double train_error = LogLoss::compute_training_error(A, ATx);
  26         LogLoss::compute_grad_at_x(ATx, A,x, lambda, grad);
  27         
  28         std::cout << std::setw(10) << std::left << k << std::setw(20) << std::left << std::setprecision(10) << obj_val << std::setw(20) << std::left << train_error << "\n" ;
  29         
  30     }
  31 }
  32 
  33 void GradientDescent::run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad){
  34     
  35     LogLoss::compute_grad_at_x(ATx, A, x, lambda, grad);
  36     std::transform(x.begin(), x.end(), grad.begin(), x.begin(), [=](double x_i, double grad_i){return x_i - this->alpha*grad_i;});
  37     
  38     
  39 }



```

I have some rudimentary logging set up in lines 8, 9 and 28 for us to observe the progress of the solution. We always choose the starting value of as $x = 0$. First we calculate the values of `ATx` (line 3), objective value (line 4, purely for logging) and training error (line 5 and purely for logging). Then in lines (8,9) we log the obj. function value and training error value at $x=0$. We then run the solver for the number of iterations specified by the user, i.e me. In each iteration, we run one epoch of gradient descent (lines 21, 33-39). Each epoch of gradient descent has two steps --- calculating the gradient of the logloss function at $x$, there by getting the search direction $s$ (line 35) and updating $x := x + \frac{1}{L} s$ (line 36). This is a simple daxpy operation and can be achieved by `std::transform`. We ask the user to supply $\frac{1}{L}$ value directly.   After this one iteration is finished running, we set calculate the new values of $ATx$ for the next iterate and also calculate the objective function value, $P(x)$ and training error value for logging. This finishes the Gradient Descent algorithm. 

The above code can be simplified a bit. We can stop calculating the value of $ATx$ outside the `run_one_iter` function and calculate it inside. There is probably no need to pass the  `grad`  and lines 3-5 and 24-26 are kind of duplicate. We can optimize it further. To maintain parity between this blog post and the source code, I will not make these changes and if I do, I will try to update it here too. 

The main function is fairly straightforward. We get the data into `A` variable by using the `get_CRSM_from_svm` function in ParseSVM library.  We choose the values of $\lambda = 0.0001$ and $\frac{1}{L} = 10$. We will run it for 100 iteration

```c++
int int main(int argc, const char * argv[]) {
    
    const std::string file_path = argv[1];
    
    Classification_Data_CRS A;
    
    //We will store the problem data in variable A and the data is going to be normalized
    get_CRSM_from_svm(A, file_path);
    
    std::cout << "GD: " << "\n";
    GradientDescent GD;
    double lambda = 0.0001;
    double Lips = 10.0;
    int iters = 100;
    GD.init(A, lambda, Lips, iters);
    GD.run_solver(A);
}

```

The next blogpost will have the Stochastic Gradient Descent (SGD) Implementation. I will try to compare this with SGD. You can find the full source at [github](https://github.com/CGudapati/BinaryClassification)
