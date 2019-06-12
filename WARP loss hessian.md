

```python
import numpy as np
import math
from scipy.optimize import approx_fprime
from matplotlib import pyplot as plt
%matplotlib inline
```


```python
num_rows = 500
num_cols = 100

#generate actual categories
y = np.zeros((num_rows,num_cols))
rows = np.arange(num_rows)
cols = np.random.randint(0,num_cols,num_rows)
y[rows,cols]+=1

#generate output of predict_proba_ type function
y_hat = np.random.exponential(1,(num_rows,num_cols))
y_hat = ((1/np.sum(y_hat,axis=1))*y_hat.T).T
```


```python
#get positive indices
p = np.argmax(y_hat, axis=1)
positive_indices = np.argmax(y_hat, axis=1)
#positive_indices = np.zeros((num_rows,num_cols))
#rows = np.arange(num_rows)
#positive_indices[rows,p]+=1

#get negative indices and L
negative_indices = np.zeros(num_rows)
L=np.zeros(num_rows)
for i in range(num_rows):
    msk = np.full(num_cols,True)
    j = p[i]
    msk[j]=0
    all_labels_idx = np.arange(num_cols)
    neg_labels_idx = all_labels_idx[msk]
    num_trials = 1
    while num_trials<num_cols:
        neg_idx = np.random.choice(neg_labels_idx)
        msk[neg_idx] = False
        neg_labels_idx = all_labels_idx[msk]

        margin = 1 + y_hat[j,neg_idx] - y_hat[i,j]

        if margin > 0:
            loss_weight = np.log(math.floor((num_cols-1)/(num_trials)))
            L[i] = loss_weight
            negative_indices[i] = neg_idx
            break
        else:
            num_trials+=1
            continue
negative_indices = np.array(negative_indices, dtype=int)
```


```python
def loss(x):
    return(
        L.T@(1+
            x[np.arange(num_rows),negative_indices] -
            x[np.arange(num_rows),positive_indices]))

def loss_helper(x):
    return(
        (1+
            x[np.arange(num_rows),negative_indices] -
            x[np.arange(num_rows),positive_indices]))

def loss_grad(x):
    epsilon_scalar = 1e-10
    loss_helper(y_hat)
    epsilon_mat = np.full(y_hat.shape,epsilon_scalar)
    f = (loss_helper(y_hat+epsilon_mat) - loss_helper(y))/epsilon_scalar
    return L*f
```


```python
def hessian ( x0, loss_fn, epsilon=1e-5, linear_approx=False, *args ):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)
    """
    # ``calculate_cost_function`` is the cost function implementation
    # The next line calculates an approximation to the first
    # derivative
    
    #f1 = approx_fprime( x0, loss_fn, epsilon = epsilon, *args)
    f1 = loss_grad(y_hat)

    # This is a linear approximation. Obviously much more efficient
    # if cost function is linear
    if linear_approx:
        f1 = np.matrix(f1)
        return f1.transpose() * f1    
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = np.zeros ( ( n, n ) )
    # The next loop fill in the matrix
    xx = x0
    for j in range( n ):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        #f2 = approx_fprime( x0, loss_fn,epsilon = epsilon, *args)
        f2 = loss_grad(y_hat)
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0        
    return hessian
```


```python
h = hessian(y_hat, loss, 1e-1)
```
