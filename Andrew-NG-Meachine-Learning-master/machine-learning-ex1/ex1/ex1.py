#!/usr/bin/env python
# coding: utf-8

# In[9]:


# 导入绘图包
import matplotlib.pyplot as plt
# 导入numpy包
import numpy as np

#导入数据
datafile = open('ex1data1.txt')
data = datafile.readlines()
dataCount = len(data)
tx = []
ty = []
for i in range(dataCount):
    tx.append(float(data[i].split(',')[0]))
    ty.append(float(data[i].split(',')[1]))
plt.scatter(tx, ty)
plt.show()


# In[10]:


# 创建自变量矩阵
x = np.matrix(tx).T
# 插入1
x = np.insert(x,0,np.ones(x.shape[0]),1)
# print(x)
y = np.matrix(ty).T


# In[11]:


# 损失函数
def J_theta(theta,x,y):
    return np.sum(np.power(x * theta - y,2 )) / (2 * len(x))

# 预测函数
def h_theta(x,theta):
    return x * theta


# In[12]:


# 初始化
theta  = [0.0,0.0] # theta0 = 0,theta1 = 0
theta = np.matrix(theta).T
# 输出初始化时候的损失数
print(J_theta(theta,x,y))
alpha = 0.01  # 定义学习速度
iterations = 15000 #迭代次数


# In[13]:


# 梯度下降
lastJ = 35 #定义上一次的损失函数值，偏大一点方便运算
minx1 = 100 #定义theta 0 的最小值
maxx1 = -100 #定义theta 1 的最大值
minx2 = 100 #定义theta 0 的最小值
maxx2 = -100 #定义theta 1 的最大值
j = [] #记录每次迭代产生的损失函数值
for time in range(iterations):
    tempSum = x * theta - y # 计算预测函数的值
    minx1 = min(minx1,theta[0,0])  #更新
    maxx1 = max(maxx1,theta[0,0])  #更新
    minx2 = min(minx2,theta[1,0])  #更新
    maxx2 = max(maxx2,theta[1,0])  #更新
    J = J_theta(theta,x,y)
    j.append(J)
    if J > lastJ: # 如果j 反倒变大了，说明学习速率太快了
        alpha /= 2
    elif J == lastJ: # 如果几乎没变化，说明差不多已经优化到极限了，输出迭代次数
        print(time)
        break
    lastJ = J
    for i in range(2): #更新theta
        theta[i, 0] -= alpha * np.sum(np.multiply(tempSum, x[:, i])) / len(x) # np.multipy 是矩阵对应位置相乘，而不是使用矩阵乘法

print(minx1,maxx1,minx2,maxx2)


# In[14]:


# 绘图
lx = range(5,25)
ly = [theta[0,0] + theta[1,0] * i for i in lx]

print(lastJ)

plt.scatter(tx, ty)
plt.plot(lx,ly)
plt.show()


# In[15]:


#绘制损失函数随着次数的曲线
plt.plot([i for i in range(len(j))],j)
plt.show()


# In[16]:


# 绘制损失函数随着参数的曲线
imaged3x = np.linspace(minx1,maxx1, num=100)
image3dy = np.linspace(minx2,maxx2,num=100)
imaged3x,image3dy = np.meshgrid(imaged3x,image3dy)

loss_grid = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        tmp_theta = np.array([[imaged3x[i,j]],[image3dy[i,j]]])
        loss_grid[i,j] = J_theta(tmp_theta,x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(imaged3x, image3dy, loss_grid,cmap='rainbow')
ax.set_xlabel('theta1')
ax.set_ylabel('theta2')
ax.set_zlabel('loss')
ax.view_init( azim=45)
plt.show()

