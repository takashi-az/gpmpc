import numpy as np

class GP():
    def __init__(self):
        self.DT = 0.2
        self.Kffinv = np.array([])
        self.buffer = np.array([]) #[x,y,w,vx,vy,r,u1,u2,u3]
        self.x = np.array([]) #[vx,vy,r,u1,u2,u3]
        self.y = np.array([]) #[state_size,data_size]
        self.episode_idx = np.array([])
        self.var = 0.0
        
    def create_gram_matrix(self):
        if len(self.buffer) > 3000:
            self.buffer = self.buffer[-3000:]
        self.x,self.y = self.calc_x_y(self.buffer)
        var = np.var(self.x)
        self.var = var
        x_len = len(self.x)
        Kff = np.zeros((x_len,x_len))
        for i in range(x_len):
            for j in range(x_len):
                Kff[i][j] = self.kernel(self.x[i],self.x[j],theta1=var)
                Kff[j][i] = Kff[i][j].copy()
        Kff = Kff + np.diag(np.ones(len(self.x))/5) 
        self.Kffinv = np.linalg.inv(Kff)
    
    def log(self):
        self.episode_idx = np.append(self.episode_idx,len(self.buffer)-1)
    
    def memory(self,x,u):
        T = np.array([[0.707,0.707,-0.707,-0.707],
                      [-0.707,0.707,-0.707,0.707],
                      [-0.1888,0.1888,0.1888,-0.1888]])
        new_u = T@u
        if len(self.buffer.shape)==1:
            self.buffer = np.append(self.buffer,np.concatenate([x,new_u]))
            self.buffer = self.buffer.reshape(1,-1)
        elif len(self.buffer.shape)==2:
            self.buffer = np.concatenate([self.buffer,np.concatenate([x,new_u]).reshape(1,-1)])
        
    def calc_x_y(self,buffer):
        x_traj = np.array([])
        y = np.array([])
        for i in range(len(buffer)-1):
            if i in self.episode_idx:
                continue
            else:
                x,u = buffer[i][:6],buffer[i][-3:]
                if len(x_traj.shape)==1:
                    x_traj = np.append(x_traj,x)
                    x_traj = x_traj.reshape(1,-1)
                elif len(x_traj.shape)==2:
                    x_traj = np.concatenate([x_traj,x.reshape(1,-1)])
                next_state = buffer[i+1][:6]
                y_pre = (next_state-x)/self.DT - self.f_nom(x,u)
                if len(y.shape)==1:
                    y = np.append(y,y_pre)
                    y = y.reshape(1,-1)
                elif len(y.shape)==2:
                    y = np.concatenate([y,y_pre.reshape(1,-1)])
        return x_traj,y.T 
    
    def f_nom(self,curr_x,inp):
        x,y,psi,u,v,r = curr_x
        m = 11.5;Iz = 0.16
        Xudot = -5.5;Yvdot = -12.7;Nrdot = -0.12
        Mrb = np.array([[ m , 0 , 0 ],
                      [ 0 , m , 0 ],
                      [ 0 , 0 , Iz]])
        Ma = -np.array([[Xudot,0,0],
                        [0,Yvdot,0],
                        [0,0,Nrdot]])
        g = np.array([[0],
                      [0],
                      [0]])
        Minv = np.linalg.inv(Mrb+Ma)
        T = np.array([[0.707,0.707,-0.707,-0.707],
                      [-0.707,0.707,-0.707,0.707],
                      [-0.1888,0.1888,0.1888,-0.1888]])
        J = np.array([[np.cos(psi),-np.sin(psi),0],
                      [np.sin(psi),np.cos(psi),0],
                      [0,0,1]])
        xdot = np.block([[np.zeros([3,3]),J],[np.zeros([3,3]),np.zeros([3,3])]])@curr_x.T+np.block([[np.zeros([3,3])],[Minv]])@inp
        return xdot

    
    def kernel(self,x,y,theta1=1):
        Linv = np.diag(np.ones(len(x))/np.array([0.2,0.5,0.5,3,3,3]))
        return theta1*np.exp(-(x-y)@Linv@(x-y).T)
    
    def predict(self,x_):
        '''
        x_ : [vx,vy,r,u1,u2,u3]
        '''
        kfs = np.zeros([len(self.x),len(x_)])
        for i in range(len(self.x)):
            for j in range(len(x_)):
                kfs[i][j] = self.kernel(self.x[i],x_[j],self.var)
        kss = np.zeros([len(x_),len(x_)])
        for i in range(len(x_)):
            for j in range(len(x_)):
                kss[i][j] = self.kernel(x_[i],x_[j],self.var)
        newdot = np.zeros(6)
        for i in range(len(x_)):
            newdot[i] = kfs.T@self.Kffinv@self.y[i]
        # sigma = kss-kfs.T@Kinv@kfs
        return newdot #,sigma

# class FITC(GP):
#     def __init__(self,inducing_point=5,x_max=[],x_min=[]):
#         '''
#         Kfu : 今までのデータと補助変数とのグラム行列
#         Ksu : 新しく得られたデータと補助変数のグラム行列
#         Kuu : 補助変数のグラム行列
#         inducing_point : 一次元の格子点
#         lattice_point : 格子点
#         '''
#         super().__init__()
#         self.Kfu = np.array([])
#         self.inducing_point = inducing_point
#         self.lattice_point = np.zeros(inducing_point**len(x_max)) 
#         for i,ix in enumerate(np.linspace(x_min[0]*0.95,x_max[0]*0.95,inducing_point)):
#             for j,jx in enumerate(np.linspace(x_min[1]*0.95,x_max[1]*0.95,inducing_point)):
#                 for k,kx in enumerate(np.linspace(x_min[2]*0.95,x_max[2]*0.95,inducing_point)):
#                     self.lattice_point[25*i+5*j+k] = np.array([ix,jx,kx])
#         self.Kuu = np.zeros((self.inducing_point**3,self.inducing_point**3))
#         for i in range(self.inducing_point**3):
#             for j in range(self.inducing_point**3):
#                 self.Kuu[i][j] = self.kernel(self.lattice_point[i],self.lattice_point[j])
#         self.Kuuinv = np.linalg.inv(self.Kuu)
#         self.

#     def create_gram_matrix(self,x):
#         super().create_gram_matrix()

#     def predict(self,x_):
#         Kss = np.zeros([len(x_),len(x_)])
#         Ksu = np.zeros([len(x_),self.inducing_point**3])
#         for i in range(len(x_)):
#             for j in range(len(x_)):
#                 Kss[i][j] = self.kernel(x_[i],x_[j])
#             for j in range(self.inducing_point**3):
#                 Ksu[i][j] = self.kernel(x_[i],self.lattice_point[j])
#         Qss = Ksu@self.Kuuinv@Ksu.T
#         Qff = self.Kfu@self.Kuuinv@self.Kfu.T
#         var = np.var(x)
#         A = np.diag(self.Kff-Qff+np.diag([var for i in range(len(x))]))
#         Ainv = np.diag(1./A)
#         sig = np.linalg.inv(self.Kfu.T@Ainv@self.Kfu+self.Kuu)
#         mean = Ksu@sig@self.Kfu.T@Ainv@y.T
#         sigma = Kss - Qss + Ksu@sig@Ksu.T
#         return mean,sigma