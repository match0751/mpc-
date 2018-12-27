import numpy as np
from scipy.optimize import minimize
import mpc 

#------------初始化相关数值------------

prediction_horizon = 30 #预测范围是10 

#state_number = 3 #状态量有两个

state_number = 1 #状态量有两个

input_number = 1 #输入量只有两个，线速度以及角速度

#x_d = 10*np.random.rand( prediction_horizon,state_number ) #初始化理想状态量（列向量）随机数


sin=np.linspace(2, 5, num=prediction_horizon)
#sin_1=np.linspace(2, 5, num=prediction_horizon)
#sin_2=np.linspace(3, 6, num=prediction_horizon)
#sin_3=np.linspace(4, 7, num=prediction_horizon)

#print("sin_1")
#print(np.sin(sin_1))

#print("sin_2")
#print(np.sin(sin_2))

#print("sin_3")
#print(np.sin(sin_3))

#x_d = 10*np.sin( [sin_1,sin_2,sin_3] ) #初始化理想状态量（列向量）随机数
#x_d = x_d.T #构造30*3的矩阵

x_d = 10*np.sin( sin ) #初始化理想状态量（列向量）随机数

#x_d = 10*np.random.rand( prediction_horizon,state_number ) #初始化理想状态量（列向量）随机数

print("x_d")
print(x_d)

#x=np.zeros((prediction_horizon,state_number))#初始化实际状态量（列向量)为零

x= np.random.rand(prediction_horizon,state_number)#初始化实际状态量（列向量)为随机数

#print("x")
#print(x)

Q=1000*np.eye(prediction_horizon) #初始化Q（需要半正定）

#print("Q")
#print(Q)

R=10*np.eye(prediction_horizon) #初始化R（需要正定）

#print("R")
#print(R)

#u = np.zeros((prediction_horizon,input_number)) #初始化实际状态量（列向量）
u = np.random.rand(prediction_horizon,input_number) #初始化实际状态量（列向量）为随机数

#print("u")
#print(u)


x = mpc.MPC(x,x_d,prediction_horizon,state_number,input_number)

#print("x.Q")
#print(x.Q)

#print("x.R")
#print(x.R)

temp = x.optimize()
