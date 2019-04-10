import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

from scipy import sparse
eye = np.eye(4)
print("Numpy array:\n{}".format(eye))

# 将NumPy数组转换成CSR格式的SciPy稀疏矩阵
# 只保存非零元素
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))


data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))



import matplotlib.pyplot as plt

# 在-10和10之间生成一个数列，共100个数
x = np.linspace(-10, 10, 100)
# 用正弦函数创建第二个数组
y = np.sin(x)
# plot函数绘制一个数组关于另一个数组的折线图
plt.plot(x, y, marker = "x")

import pandas as pd
from IPython.display import display

# 创建关于人的简单数据集
data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Location': ['New York', 'Paris', 'Berlin', 'London'],
        'Age': [24,  13, 53, 33]}
data_pandas = pd.DataFrame(data)
# IPython.display可以在Jupyter Notebook中打印出”美观的“DataFrame
display(data_pandas)


