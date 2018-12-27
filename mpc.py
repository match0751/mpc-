#-------------导入相关安装包-------------
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt


# 定义一个模型预测控制的类 
class MPC:
    

    #------------1.定义数据传入函数------------
    def __init__(self,x,x_d,prediction_horizon,state_number,input_number):

        self.x_d=x_d #将理想轨迹传入

        self.x=x #将实际轨迹传入

        self.prediction_horizon=prediction_horizon #预测范围

        self.state_number=state_number #状态量

        self.input_number=input_number #输入量

    #------------初始化相关数值------------

        self.Q=1000*np.eye(prediction_horizon) #初始化Q（需要半正定）

        self.R=10*np.eye(prediction_horizon) #初始化R（需要正定）

        self.u = np.random.rand(prediction_horizon,input_number) #初始化实际状态量（列向量）为随机数

        #调试代码
        #print("x_d")
        #print(self.x_d)

        #print("x")
        #print(self.x)

        #print("prediction_horizon")
        #print(self.prediction_horizon)

        #print("state_number")
        #print(self.state_number)

        #print("input_number")
        #print(self.input_number)

        #print("Q")
        #print(self.Q)

        #print("R")
        #print(self.R)

        #print("u")
        #print(self.u)
        #solution = minimize(self.objective,self.x,args=(self.x_d,self.u,self.Q,self.R),method='SLSQP',constraints=cons) #求解过程放在初始化里面就可以

    #------------定义目标代价函数------------
    #def objective(self,*args):
    def objective(self,*args):
        #args的unpack
        x,x_d,u,Q,R = args

        ##调试代码
        #print("x.shape")
        #print(x.shape)

        #print("x_d.shape")
        #print(x_d.shape)

        #x数据的reshape
        #x=np.reshape(self.x,(self.prediction_horizon,self.state_number) )
        #x=np.reshape(x,(self.prediction_horizon,self.state_number) ) #貌似是不需要reshape

        ##将数值赋值给其他变量
        ##x_d,u,Q,R = args
        #x_d = self.x_d
        #u = self.u       
        #Q = self.Q 
        #R = self.R

        #print("R")
        #print(R)

        #print("x_d")
        #print(x_d)

        #print("u")
        #print(u)

        #print("Q")
        #print(Q)

        #计算代价函数--第一部分
    
        #计算所有项的误差的平方
        error_x = np.power(x-x_d,2)
       
        #调试代码
        #print("x")
        #print(x)

        #print("x_d")
        #print(x_d)
        
        #print("error_x")
        #print(error_x)        

        #将每一项加起来
        sum_x=np.sum(error_x)
       
        #计算代价函数总的部分
        J = sum_x 
        
        #print("J")
        #print(J)

        return J 
    

    #------------定义运动学函数------------
    def f(self,x,u):
        #f = 2*x  # 实际函数可能需要改
        
        f = x+u
        return f

    # 1.输入变量符合运动学方程 （等式）
    #def constraint1(self,*args):
    def constraint1(self,*args):
        #数据unpack
        x,u,num = args

        ##调试代码
        #print("x[num,:]")
        #print(x[num])

        #print("u44")
        #print(u)

        #print("num44")
        #print(num)

        

        #将x来reshape成（10,3）矩阵
        #x=np.reshape(self.x,(self.prediction_horizon,self.state_number) )
        x=np.reshape(x,(self.prediction_horizon,self.state_number) )

        #print("x")
        #print(x)

        #print("self.f(x[num,:],u) - x[num+1,:]")
        #print(self.f(x[num,:],u) - x[num+1,:])

        #对u的数据处理
        return self.f(x[num,:],u) - x[num+1,:]


    # 2.初始状态达到理想数值(等式）
    def constraint2(self,*args):
        #args的unpack
        x,x_d=args

        #将x来reshape成（10,3）矩阵
        #x=np.reshape(self.x,(self.prediction_horizon,self.state_number) )
        x=np.reshape(x,(self.prediction_horizon,self.state_number) )   

        #对x_d的处理
        x_d = np.array(x_d,dtype=float) # 将元组转化为数组
        x_d = np.reshape(x_d,(self.prediction_horizon,self.state_number) )  # 将数组reshape

        #取数据的size
        lenth = len(x[:,0]) # 将prediction horizon取出来

        return x[0,:] - x_d[0,:] # 初始数值要为零


    # 3.最终状态达到理想数值(等式）
    def constraint3(self,*args):
        #args的unpack
        x,x_d=args

        #print("x666")
        #print(x)

        #print("x_d666")
        #print(x_d)

        #print("x.shape666")
        #print(x.shape)

        #print("x_d.shape666")
        #print(x_d.shape)

        #将x来reshape成（10,3）矩阵
        #x=np.reshape(self.x,(self.prediction_horizon,self.state_number) )
        x=np.reshape(x,(self.prediction_horizon,self.state_number) )
        ##检查数据传入是否有问题
        #print("x_1")
        #print(x_1)
        #print("x")
        #print(x)

        #print("x_d")
        #print(x_d)

        #print("self.x_d")
        #print(self.x_d)
        #对x_d的处理
        x_d = np.array(x_d,dtype=float) # 将元组转化为数组
        x_d = np.reshape(x_d,(self.prediction_horizon,self.state_number) )  # 将数组reshape

        #取数据的size
        lenth = len(x[:,0]) # 将prediction horizon取出来

        tem = x[lenth-1,:] - x_d[lenth-1,:]
       
        #print("tem99")
        #print(tem)


        return x[lenth-1,:] - x_d[lenth-1,:] # 最终的数值要为零

    #------------开始进行优化------------
    
    def optimize(self):
        # 定义空的list 
        cons= []

        # 1.定义运动学约束
        for num in range(self.prediction_horizon-1):
            con = {'type': 'eq', 'fun': self.constraint1,'args':(self.u[num,:],num)}
            cons.append (con)
        
        # 2.定义初始状态约束
        con10 = {'type': 'eq', 'fun': self.constraint2,'args':(self.x_d,)} 
        cons.append(con10)

        # 3.定义最终状态约束
        con11 = {'type': 'eq', 'fun': self.constraint3,'args':(self.x_d,)} 
        #cons.append(con11)

        # 总约束：cons

        # 求解
        #solution = minimize(objective,x,args=(x_d,u,Q,R),method='SLSQP',constraints=cons)
        #solution = minimize(self.objective,self.x,args=(self.x_d,self.u,self.Q,self.R),method='SLSQP',constraints=cons)
        solution = minimize(self.objective,self.x,args=(self.x_d,self.u,self.Q,self.R),method='SLSQP',constraints=cons) # 会把初始状态导入数据
        x = solution.x 

       
        print("x")
        print(x)

        print("u")
        print(self.u)

        return x #将控制数据返回给矩阵



   


