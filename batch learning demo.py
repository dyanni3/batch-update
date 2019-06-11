#%%
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from matplotlib import pyplot as plt
#%%

# data generating process:
# y = theta_1 x_1 + theta_2 x_2 + epsilon*N(0,1)
theta1 = 2
theta2 = 8
full_size = 10000
x1 = np.random.normal(0,1,full_size)
x2 = np.random.normal(0,1,full_size)
epsilon = 1e-1
noise = epsilon*np.random.normal(0,1,full_size)
y = theta1*x1 + theta2*x2 + noise

#%%
plt.plot(x1,y,'o')
plt.plot(x2,y,'o')

#%%
Xtrain = np.stack([x1,x2],axis=1)
ytrain = y
lr = LR()
lr.fit(Xtrain,ytrain)
lr.coef_
#%%
ypred = lr.predict(Xtrain)
J2 = mse(ytrain,ypred)
qstarstar_true = lr.coef_
print(J2)
#%%
def dq(x2, y2, qstar, hess1):
    H2 = x2.T @ x2
    M = hess1+H2
    left = np.linalg.inv(M)
    right1 = (y2-x2@qstar)
    right = (x2.T)@right1
    return left@right

#%%
X1 = Xtrain[:9800]; y1 = y[:9800]
X2 = Xtrain[9800:]; y2 = y[9800:]
lr1 = LR()
lr1.fit(X1,y1)
qstar = lr1.coef_
hess1 = X1.T@X1
delta_q = dq(X2,y2,qstar,hess1)

#%%
qstarstar_pred = qstar+delta_q
nrmse = ((((qstarstar_pred - qstarstar_true)**2).sum())**.5)/qstarstar_true.mean()
nrmse_zeroth_order = ((((qstar - qstarstar_true)**2).sum())**.5)/qstarstar_true.mean()
perc_improve = round(100*nrmse/nrmse_zeroth_order,2)
print("NRMSE (normalized root mean squared error):  %s"%str(nrmse))
print("only %s percent the normalized error as zeroth order prediction"%str(perc_improve))