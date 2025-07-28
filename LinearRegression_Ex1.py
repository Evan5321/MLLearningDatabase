import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#通过本示例我们可以直接使用Sklearn库来训练一元线性回归方程

def read_data():
    # 读取CSV文件
    data = pd.read_csv('DecissionTreeData.csv')
    #print(data,type(data))
    return data['年龄（岁）'],data['薪资（万元 / 年）'],data['是否晋升']

def linear_regression(x,y):
    model = LinearRegression()
    model.fit(x,y)
    w = model.coef_[0]  # 斜率
    b = model.intercept_  # 截距
    print(f"回归方程: y = {w:.2f}x + {b:.2f}")
    pred = model.predict(x)
    return pred,w,b

if __name__ == '__main__':
    age,salary,upper = read_data()
    age = age.to_numpy()
    age = age.reshape(-1,1)
    salary = salary.to_numpy()
    pred,w,b = linear_regression(age,salary)
    plt.figure(figsize=(10, 6))
    plt.scatter(age, salary, color='blue', label='True read_data')  # 散点图表示实际数据
    plt.plot(age, pred, color='red', linewidth=2, label=f'LR Line: y = {w:.2f}x + {b:.2f}')  # 绘制回归线
    plt.xlabel('age')
    plt.ylabel('salary')
    plt.title('L$R')
    plt.legend()
    plt.grid(True)
    plt.show()

