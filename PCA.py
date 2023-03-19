#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# загрузка данных
iris = load_iris()
X = iris.data

# стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# вычисление матрицы ковариации
covariance_matrix = np.cov(X_scaled.T)

# вычисление собственных значений и собственных векторов
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# сортировка собственных значений по убыванию
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# проекция данных на новое пространство признаков
X_pca = X_scaled.dot(eigenvectors)

# построение графика
plt.scatter(X_pca[:,0], X_pca[:,1], c=iris.target)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# In[ ]:




