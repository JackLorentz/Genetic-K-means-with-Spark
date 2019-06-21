import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import random
import copy
from math import sqrt

random.seed(0)

class Kmeans:
    
    def cen_init(self, k, x):
        center = np.empty([k, x.shape[1]])
        selected = random.sample(range(x.shape[0]), k)
        for i in range(len(selected)):
            center[i] = x[selected[i]]            
        return center


    def distance(self, t, c):
        return sqrt(np.sum((t-c)*(t-c))),np.sum((t-c)*(t-c))


    def fit(self, X_train=None, nIter=10, num_of_centers=3):
        tic = time.perf_counter()

        #Step1: 隨機初始化
        #資料數
        m = X_train.shape[0]
        c_init = self.cen_init(num_of_centers, X_train)
        #print(c_init)
        cen=copy.deepcopy(c_init)
        #cen=c_init
        dim=X_train.shape[1]
        #行星數
        k = cen.shape[0]
        #每筆資料跟行星距離
        dis = np.zeros([m,k])
        cen_ass = np.zeros([m,])
        sse_dis = np.zeros([m,k])
        #行星位置歷史紀錄
        cen_his = cen
        temp = np.zeros([k,dim])
        count = 0
        #
        sse=[]
        for t in range(nIter):
            #Step2: 計算每一筆資料跟行星的距離   
            for r in range(0,m):
                for c in range(0,k):
                    dis[r][c],sse_dis[r][c] = self.distance(X_train[r],cen[c])
            cen_ass = (np.argmin(dis, axis=1)).reshape((-1,))
            #記錄每次迭代的行星位置
            cen_his = np.concatenate((cen_his, cen))
            sse.append(0)
            for i in range(m):
                sse[t]=sse[t]+sse_dis[i][cen_ass[i]]
        
            print('[the', t+1, 'th iteration]  SSE:', sse[t])
            #print()
            #print("center:",cen)
            temp = np.zeros([1,dim])
            count = 0
            for i in range(num_of_centers):
                temp = np.zeros([1,dim])
                count = 0
                for r in range(0,m):
                    temp = temp + 1.0*(cen_ass[r]==i)*X_train[r]#+(0.02)*(cen_ass[r]!=c)*X_train[r]
                    count = count + 1.0*(cen_ass[r]==i)#+(0.02)*(cen_ass[r]!=c)
                cen[i] = temp.reshape(-1,)/count
        
        toc = time.perf_counter()
        spend_time = str(1000*(toc - tic))
        print("Comuptation Time: " + spend_time + "ms")
        
        return cen, cen_ass, cen_his, sse, c_init, spend_time