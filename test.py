#-------------导入相关安装包-------------
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt

#localtime = time.asctime( time.localtime(time.time()) )
#print ("本地时间为 :", localtime)

##------------定义目标代价函数------------
#def objective(x,*args):
#    x=np.reshape(x,(prediction_horizon,state_number) )

#    #将数值赋值给其他变量
#    x_d,u,Q,R = args
#    #print("x")
#    #print(x)

#    #print("x_d")
#    #print(x_d)

#    #print("u")
#    #print(u)

#    #print("Q")
#    #print(Q)

#    #print("R")
#    #print(R)

#    #print("x-x_d")
#    #计算代价函数--第一部分
#    sum_x=np.dot((x-x_d).T,Q)
#    sum_x=np.dot(sum_x,(x-x_d))
#    sum_x=np.sum(sum_x)
#    #print("sum_x")
#    #print(sum_x)
#    #计算代价函数--第二部分
#    sum_u=np.dot(u.T,R)
#    sum_u=np.dot(sum_u,u)
#    sum_u=np.sum(sum_u)
#    #print("sum_u")
#    #print(sum_u)
#    #计算代价函数总的部分
#    J = (sum_x + sum_u)

#    print("代价函数 J：")
#    print(J)
    
#    return J 

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 



#------------定义目标代价函数------------
def objective(x,*args):
    x=np.reshape(x,(prediction_horizon,state_number) )

    #将数值赋值给其他变量
    x_d,u,Q,R = args
    
    #print("x")
    #print(x)

    #print("x_d")
    #print(x_d)

    #print("u")
    #print(u)

    #print("Q")
    #print(Q)

    #print("R")
    #print(R)

    #print("x-x_d")

    #计算代价函数--第一部分
    
    #计算所有项的误差的平方
    error_x = np.power(x-x_d,2)

    #print("error")
    #print(error)

    #将每一项加起来
    sum_x=np.sum(error_x)

    #print("sum_x")
    #print(sum_x)
    
    #计算代价函数--第二部分
    
    #计算所有项的误差的平方
    error_x = np.power(x-x_d,2)

    #print("error")
    #print(error)

    #将每一项加起来
    sum_x=np.sum(error_x)

    #print("sum_x")
    #print(sum_x)


    #计算代价函数总的部分
    J = sum_x 

    #print("代价函数 J：")
    #print(J)
    
    return J 


##------------定义运动学函数------------
#def f(x,u):
#    lenth = len(x[:,0])

#    func = np.zeros((prediction_horizon,state_number)) #定义变量

#    #print ("func")
#    #print (func)

#    #print ("x")
#    #print (x)

#    func[0:lenth-1,:] = 2*x[0:lenth-1,:] #除了最后一行，都符合运动学解结果
    
#    #print ("func")
#    #print (func)

#    #func[lenth-1,:] = x[lenth-1,:] #最后一行保持原样
   
#    #print ("func")
#    #print (func)

#    # 实际函数可能需要改变
#    return func

#------------定义运动学函数------------
def f(x,u):

    #f = 2*x  # 实际函数可能需要改变
    f = x-u
    return f


#------------定义约束------------

## 1.输入变量符合运动学方程 （等式）
#def constraint1(x,*args):
#    #将x来reshape成（10,3）矩阵
#    x=np.reshape(x,(prediction_horizon,state_number) )
    
#    #对u的数据处理
#    u = args
#    u = np.array(u,dtype=float) # 将元组转化为数组
#    u = np.reshape(u,(prediction_horizon,input_number) )  # 将数组reshape

#    print("u")
#    print(u)

#    print("u.shape")
#    print(u.shape)

#    #取数据的size
#    lenth = len(x[:,0]) # 将prediction horizon取出来

#    print("lenth")
#    print(lenth)

#    print("x")
#    print(x)

#    print("f( x[0:lenth-1,:],u[0:lenth-1,:] )")
#    print(f( x[0:lenth-1,:],u[0:lenth-1,:] ))

#    print("x[1:lenth,:]")
#    print(x[1:lenth,:])

#    print("f( x[0:lenth-1,:],u[0:lenth-1,:] ) - x[1:lenth,:]")
#    print(f( x[0:lenth-1,:],u[0:lenth-1,:] ) - x[1:lenth,:])

#    ##将运动学约束定义出来(由于有for循环，暂时写这部分）
#    #for i in range(len(x[:,0])):
#    #    x[i+1,0] = 
#    ## 将运动学约束用矩阵的形式定义出来
#    #x[1:lenth,:] = f( x[0:lenth-1,:],u[0:lenth-1,:] ) # 具体的公式推导看onenote 
#    return f( x[0:lenth-1,:],u[0:lenth-1,:] ) - x[1:lenth,:] 

# 1.输入变量符合运动学方程 （等式）
def constraint1(x,*args):
    #将x来reshape成（10,3）矩阵
    x=np.reshape(x,(prediction_horizon,state_number) )
    
    #对u的数据处理
    u,num = args
    
    #f(x[num,:],u)

    return f(x[num,:],u)- x[num+1,:]


# 2.初始状态达到理想数值(等式）
def constraint2(x,*args):
    #将x来reshape成（10,3）矩阵
    x=np.reshape(x,(prediction_horizon,state_number) )
    
    #对x_d的处理
    x_d = args
    x_d = np.array(x_d,dtype=float) # 将元组转化为数组
    x_d = np.reshape(x_d,(prediction_horizon,state_number) )  # 将数组reshape
    #print("x_d")
    #print(x_d.shape)

    #取数据的size
    lenth = len(x[:,0]) # 将prediction horizon取出来

    return x[0,:] - x_d[0,:] # 初始数值要为零


# 3.最终状态达到理想数值(等式）
def constraint3(x,*args):
    #将x来reshape成（10,3）矩阵
    x=np.reshape(x,(prediction_horizon,state_number) )
    
    #对x_d的处理
    x_d = args
    x_d = np.array(x_d,dtype=float) # 将元组转化为数组
    x_d = np.reshape(x_d,(prediction_horizon,state_number) )  # 将数组reshape
    #print("x_d")
    #print(x_d.shape)

    #取数据的size
    lenth = len(x[:,0]) # 将prediction horizon取出来

    return x[lenth-1,:] - x_d[lenth-1,:] # 最终的数值要为零



#------------初始化相关数值------------

prediction_horizon = 100 #预测范围是10 

#state_number = 3 #状态量有两个

state_number = 1 #状态量有两个

input_number = 1 #输入量只有两个，线速度以及角速度

#x_d = 10*np.random.rand( prediction_horizon,state_number ) #初始化理想状态量（列向量）随机数

sin=np.linspace(2, 5, num=prediction_horizon)

x_d = 10*np.sin( sin ) #初始化理想状态量（列向量）随机数

#x_d = 10*np.random.rand( prediction_horizon,state_number ) #初始化理想状态量（列向量）随机数

#print("x_d")
#print(x_d)

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


#------------开始进行优化------------

# 定义空的list 
cons= []

#print("cons")
#print(cons)

# 1.定义运动学约束
for num in range(prediction_horizon-1):
    print("num")
    print(num)

    con = {'type': 'eq', 'fun': constraint1,'args':(u[num,:],num)}
    
    cons.append (con)

    #print("cons")
    #print(cons)


# 2.定义初始状态约束
con10 = {'type': 'eq', 'fun': constraint2,'args':(x_d,)} 

# 3.定义最终状态约束
con11 = {'type': 'eq', 'fun': constraint3,'args':(x_d,)} 
cons.append(con11)

# 总约束


# 求解
#solution = minimize(objective,x,args=(x_d,u,Q,R),method='SLSQP',constraints=cons)
solution = minimize(objective,x,args=(x_d,u,Q,R),method='SLSQP',constraints=cons)

x = solution.x 

x=np.reshape(x,(prediction_horizon,state_number) )

# 数据可视化

# 将两个数值画在一张图片上面 
X = np.linspace(1, prediction_horizon, prediction_horizon, endpoint=True)

plt.plot(X,x,label="x")
plt.plot(X,x_d,label="x_d")
plt.show()

print("x_d")
print(x_d.T)

print("x")
print(x)

print("u")
print(u)

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


