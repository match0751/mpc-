#-------------导入相关安装包-------------
import numpy as np
from scipy.optimize import minimize
import time


#------------定义目标代价函数------------
def objective(J,x_d,x,u,Q,R):
    x_d,x,u,Q,R = args


    # sum（）把所有误差量加起来，J此时是一个标量
    J = np.sum((x_d-x).T*Q*(x_d-x)) + np.sum(u.T*R*u) # 计算代价函数实时数值

    return J

#------------定义运动学函数------------
def f(x,u):

    f = 2*x  # 实际函数可能需要改变

    return f


#------------定义约束------------

# 1.输入变量符合运动学方程 （等式）
def constraint1(x,u):
    #取数据的size
    lenth = len(x[:,0]) # 将prediction horizon取出来
    ##将运动学约束定义出来(由于有for循环，暂时写这部分）
    #for i in range(len(x[:,0])):
    #    x[i+1,0] = 

    ## 将运动学约束用矩阵的形式定义出来
    #x[1:lenth,:] = f( x[0:lenth-1,:],u[0:lenth-1,:] ) # 具体的公式推导看onenote 
    return f( x[0:lenth-2,:],u[0:lenth-2,:] ) - x[1:lenth-1,:] 

## 2.初始状态满足要求(等式）
#def constraint2(x,x_init):
#    return x[0,:]-x_init

# 3.最终状态达到理想数值(等式）
def constraint3(x,x_d):
    #取数据的size
    lenth = len(x[:,0]) # 将prediction horizon取出来

    return x[lenth-1,:] - x_d[lenth-1,:] # 最终的数值要为零

#localtime = time.asctime( time.localtime(time.time()) )
#print ("本地时间为 :", localtime)

#------------初始化相关数值------------

J = 0 #代价函数初始值为零

prediction_horizon = 10 #预测范围是10 

state_number = 3 #状态量有两个

input_number = 2 #输入量只有两个，线速度以及角速度

x_d = np.random.rand( prediction_horizon,state_number ) #初始化理想状态量（列向量）随机数

x=np.zeros((prediction_horizon,state_number))#初始化实际状态量（列向量)为零

Q=10*np.eye(prediction_horizon) #初始化Q（需要半正定）

R=10*np.eye(prediction_horizon) #初始化R（需要正定）

u = np.zeros((prediction_horizon,input_number)) #初始化实际状态量（列向量）

#------------状态量初始化------------

x_init = x_d[0,:] # 将状态量初始化成x_d的第一个量

x[0,:] = x_d[0,:]

# 显示初始状态
#print('Initial Objective: ' + str(objective(x[0,:])))


#------------开始进行优化------------

# 定义边界
b = (1.0,10.0)
bnds = (b, b, b, b)

# 定义约束
con1 = {'type': 'eq', 'fun': constraint1} 
#con2 = {'type': 'eq', 'fun': constraint2}
con3 = {'type': 'eq', 'fun': constraint3}

#cons = ([con1,con2,con3])
cons = ([con1,con3])
solution = minimize(objective,x_init,args = ( x_d,x,u,Q,R ),method='SLSQP',
                    bounds=bnds,constraints=cons)
x = solution.x

#localtime = time.asctime( time.localtime(time.time()) )
#print ("本地时间为 :", localtime)

# show final objective
#print('Final Objective: ' + str(objective(x)))

## print solution
#print('Solution')
#print('x1 = ' + str(x[0]))
#print('x2 = ' + str(x[1]))
#print('x3 = ' + str(x[2]))
#print('x4 = ' + str(x[3]))
