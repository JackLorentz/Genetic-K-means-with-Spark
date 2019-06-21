import pandas as pd
import numpy as np
import random
import math
import time
import matplotlib.colors as colors
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score

random.seed(0)

class Cluster():
    def __init__(self, data):
        self.data
        self.dim = data.shape[1]
        self.centroid = np.zeros([1, self.dim]).reshape(-1,)
        self.data_points = []
    

    def distance(self, a, b):
        return np.sum((a-b)*(a-b))


    # this method finds the average distance of all elements in cluster to its centroid
    def computeS(self):
        points = np.array(self.data_points)
        num = points.shape[0]
        dis_sum = 0.0
        #計算每一筆資料跟行星的距離   
        for i in range(num):
            dis_sum += math.sqrt(self.distance(self.data[i], self.centroid))
        return dis_sum / num



class Chromosome():
    def __init__(self, data, num_cluster):
         #初始化參數
        self.data = data
        self.kmax = num_cluster
        self.data_num = data.shape[0]
        self.dim = data.shape[1]
        self.center = self.init_center()
        self.sol = self.init_sol()
        #每次都會更新的參數
        self.fitness = 0.0
        self.mutation_rate = [0.0] * data.shape[0]
        self.sse = 0.0


    #隨機選n個中心點   
    def init_center(self):
        center = np.empty([self.kmax, self.dim])
        selected = random.sample(range(self.data_num), self.kmax)
        for i in range(self.kmax):
            center[i] = self.data[selected[i]]            
        return center
    
    
    #對於每一個染色體，隨機產生一組解 => 每一個等位基因代表對應的群 => shape=(150, 1)
    def init_sol(self):
        dis = np.zeros([self.data_num, self.kmax])
        #計算每一筆資料跟行星的距離   
        for r in range(self.data_num):
            for c in range(self.kmax):
                dis[r][c] = self.distance(self.data[r], self.center[c])
        #分群 
        sol = (np.argmin(dis, axis=1)).reshape((-1,))
        return sol
         

    def distance(self, a, b):
        return np.sum((a-b)*(a-b))
    

    def cal_SSE(self):
        sse = 0.0
        for i in range(len(self.sol)): 
            square_error = self.distance(self.data[i], self.center[self.sol[i]])
            sse += square_error
        return sse
    

    def cal_fitness_v2(self, max_fitness, c):
        return c*max_fitness - self.fitness
    

    def cal_fitness(self):
        '''
        clusters = [Cluster()]*self.kmax
        
        for c in range(self.kmax):
            clusters[c] = Cluster(data)
        
        for c in range(self.kmax):
            clusters[c].centroid = self.center[c]
            for r in range(self.data_num):
                if self.sol[r] == c:
                    clusters[c].data_points.append(self.data[r])
        '''
        #dbi = davies_bouldin_score(data, self.sol)
        silhouette = silhouette_score(self.data, self.sol)
        #ari = adjusted_rand_score(labels, self.sol)
        
        return silhouette
    

    def daviesBouldin(self, clusters):
        sigmaR = 0.0
        nc = self.kmax
        for i in range(nc):
            sigmaR += self.computeR(clusters, i, clusters[i])
            #print(sigmaR)
        DBIndex = float(sigmaR) / float(nc)
        #print('DBI:', DBIndex)
        return DBIndex
    

    def computeR(self, clusters, i, iCluster):
        listR = []
        for j, jCluster in enumerate(clusters):
            if(i != j):
                temp = self.computeRij(iCluster, jCluster)
                listR.append(temp)
        #print(max(listR))
        return max(listR)


    def computeRij(self, iCluster, jCluster):
        Rij = 0
        d = math.sqrt(self.distance(iCluster.centroid, jCluster.centroid))
        Rij = (iCluster.computeS() + jCluster.computeS()) / d

        return Rij
    

    def mutation(self):
        sol_distribution = sorted(self.sol)
        '''step 1 : choose the value to mutate'''
        for i in range(self.data_num):
            self.mutation_rate[i] = 0.02
            #print('mutation rate:', self.mutation_rate[i])
            sign = random.uniform(0.0, 1.0)
            if(sign < self.mutation_rate[i]):
                '''step 2 : do mutation on chosen point by wheel'''
                self.sol[i] = random.choice(sol_distribution)
                
        '''step 3 : update center node'''
        for c in range(self.kmax):
            temp = np.zeros([1, self.dim])
            count = 0
            for r in range(self.data_num):
                temp += int(self.sol[r] == c)*self.data[r]
                count += int(self.sol[r] == c)
            self.center[c] = temp.reshape(-1,) / count


    def KMO(self):
        dis = np.zeros([self.data_num, self.kmax])
        #計算每一筆資料跟行星的距離   
        for r in range(self.data_num):
            for c in range(self.kmax):
                dis[r][c] = self.distance(self.data[r], self.center[c])
        #分群 
        #self.sol = (np.argmin(dis, axis=1)).reshape((-1,))
        sol_tmp = (np.argmin(dis, axis=1)).reshape((-1,))
        silhouette = silhouette_score(self.data, sol_tmp)
        if silhouette > self.fitness:
            self.sol = sol_tmp
            self.fitness = silhouette
            
        #更新每一群重心
        for c in range(self.kmax):
            temp = np.zeros([1, self.dim])
            count = 0
            for r in range(self.data_num):
                temp += int(self.sol[r] == c)*self.data[r]
                count += int(self.sol[r] == c)
            self.center[c] = temp.reshape(-1,) / count
        #更新SSE
        self.SSE = self.cal_SSE()



class Genetic_Kmeans:
    def __init__(self, population_size=10, num_cluster=3, MAX_GEN=10, Ps=0.8, Pc=0.8):
        self.population_size = population_size
        self.num_cluster = num_cluster
        self.MAX_GEN = MAX_GEN
        self.Ps = Ps
        self.Pc = Pc


    def fit(self, data, labels):
        self.data = data
        self.labels = labels

        tic = time.perf_counter()

        fitness_his = []
        sse_his = []
        ari_his = []
        cen_his = []
        population = []
        for i in range(self.population_size):
            population.append(Chromosome(data, self.num_cluster))
            
        #for each generation
        for i in range(self.MAX_GEN):
            print('[the', i+1, 'th generation]')
            population = self.selection(population, self.Ps)
            population = self.crossover(population, self.Pc)

            print('After Crossover:')
            for i in range(len(population)):
                print('chromosome', i, "'s fitness value", population[i].fitness)
                    
            #for each chromosome
            for j in range(self.population_size):
                #population[j].mutation()
                population[j].KMO()
            #找出最大適應值的染色體
            population = sorted(population, reverse=True, key=lambda elem: elem.fitness)
            ari = adjusted_rand_score(labels, population[0].sol)
            print('Fitness value:', population[0].fitness)
            print('Sum of Square Error:', population[0].SSE)
            print('Adjusted Rand Index:', ari)
            print('=======================================')
            fitness_his.append(population[0].fitness)
            sse_his.append(population[0].SSE)
            ari_his.append(ari)
            cen_his.append(population[0].center)
        toc = time.perf_counter()
        spend_time = str(1000*(toc - tic))
        print("Comuptation Time: " + spend_time + "ms")

        return population[0], fitness_his, sse_his, spend_time


    #適者生存
    def selection(self, chromosomes, Ps):
        size = len(chromosomes)
        new_populations = []
        
        #計算個染色體的適應值,並統計存活率
        for i in range(size):
            chromosomes[i].fitness = chromosomes[i].cal_fitness()
        #存活率
        chosen_rate = Ps
        print('survival rate:', chosen_rate*100, '%')

        print('Before Selection:')
        chromosomes = sorted(chromosomes, reverse=True, key=lambda elem: elem.fitness)
        for i in range(len(chromosomes)):
            print('chromosome', i, "'s fitness value", chromosomes[i].fitness)
    
        #找出(存活率*個體數)個適應值的染色體
        #適應值越大越容易存活
        for i in range(8):
            new_populations.append(chromosomes[i])
    
        #填滿染色體數
        while len(new_populations) < size:
            idx = random.randint(0, 4)
            new_populations.append(chromosomes[idx])
        
        print('After Selection:')
        new_populations = sorted(new_populations, reverse=True, key=lambda elem: elem.fitness)
        for i in range(len(new_populations)):
            print('chromosome', i, "'s fitness value", new_populations[i].fitness)
    
        return new_populations

    #交配
    def crossover(self, chromosomes, Pc):
        numOfInd = len(chromosomes)
        #根據交配得到數量並隨機選出染色體
        index = random.sample(range(0, numOfInd - 1), int(Pc * numOfInd))
    
        new_chromosomes = []
        for i in range(len(index)):  # do how many time
            new_chromosomes = self.doCrossover(chromosomes, i, index)
        
        return new_chromosomes


    def doCrossover(self, chromosomes, i, index):
        length = chromosomes[0].sol.shape[0]
        cut = random.randint(1, length - 2)
        #依取樣順序跟隔壁交換基因(每一筆資料的分群) => sol為基因
        parent1 = chromosomes[index[i]]
        parent2 = chromosomes[index[(i + 1) % len(index)] % length]
        child1 = Chromosome(self.data, self.num_cluster)
        child2 = Chromosome(self.data, self.num_cluster)
        
        p1 = list(parent1.sol)
        p2 = list(parent2.sol)
        c1 = p1[0:cut] + p2[cut:length]
        c2 = p1[cut:length] + p2[0:cut]
        child1.sol = np.array(c1)
        child2.sol = np.array(c2)
        
        # 計算child適應值
        child1.fitness = child1.cal_fitness()
        child2.fitness = child2.cal_fitness()
        
        #父子兩代在競爭一次,留下適應值大的
        listA = []
        listA.append(parent1)
        listA.append(parent2)
        listA.append(child1)
        listA.append(child2)
        #依適應值反向排序
        listA = sorted(listA, reverse=True, key=lambda elem: elem.fitness)
        
        #留下最大的兩個
        chromosomes[index[i]] = listA[0]
        chromosomes[index[(i + 1) % len(index)] % length] = listA[1]

        return sorted(chromosomes, reverse=True, key=lambda elem: elem.fitness)