#import findspark
#findspark.init()
import sys
import pyspark
from pyspark import SparkContext, SparkConf
#from pyspark.sql import SparkSession
conf = SparkConf().setAppName("GKA_test").setMaster("local[*]")
sc = SparkContext(conf=conf)

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

iris = datasets.load_iris()
data = iris['data']
labels = iris['target']

m = data.shape[0]
dim = data.shape[1]
population_size = 10
num_cluster = 3
MAX_GEN = 10

random.seed(0)

def load_abalone():
    df = pd.read_csv('./abalone.csv')
    data = np.zeros([4177, 8])
    data[:, 0] = df['sex'].values
    data[:, 1] = df['length'].values
    data[:, 2] = df['diameter'].values
    data[:, 3] = df['height'].values
    data[:, 4] = df['whole_weight'].values
    data[:, 5] = df['shucked_weight'].values
    data[:, 6] = df['viscera_weight'].values
    data[:, 7] = df['shell_weight'].values
    labels = df['label'].values

    for i in range(len(labels)):
        if labels[i] == 28:
            labels[i] -= 1

    return data, labels

class Chromosome():
    def __init__(self, data, num_cluster):
        #初始化參數
        self.kmax = num_cluster
        self.data_num = data.shape[0]
        self.dim = data.shape[1]
        self.center = self.init_center(data)
        self.sol = None
    
    #隨機選num_cluster個中心點    
    def init_center(self, data):
        center = []
        selected = random.sample(range(self.data_num), self.kmax)
        for i in range(self.kmax):
            center.append(data[selected[i]])            
        return center
        
    #對於每一個染色體，隨機產生一組解 => 每一個等位基因代表對應的群 => shape=(150, 1)
    def cal_solution(self, rdd):
        self.sol = np.array(rdd.map(lambda p: self.closestPoint(p, self.center)).take(self.data_num))

    def distance(self, a, b):
        return np.sum((a-b)*(a-b))
    
    #這個函數的目的就是求取該點應該分到哪個中心點的集合去，返回的是序號
    def closestPoint(self, p, centers):
        bestIndex = 0
        closest = float("+inf")
        for i in range(len(centers)):
            tempDist = np.sum((p - centers[i]) ** 2)
            if tempDist < closest:
                closest = tempDist
                bestIndex = i
        return bestIndex
    
    def cal_fitness(self, data):
        return silhouette_score(data, self.sol)
    
    def cal_SSE(self, data):
        sse = 0.0
        for i in range(len(self.sol)): 
            square_error = self.distance(data[i], self.center[self.sol[i]])
            sse += square_error
        return sse
    
    def KMO(self, rdd):
        #計算每一筆資料跟行星的距離
        #對所有資料執行map過程，最終生成的是(index, (point, 1))的rdd
        closest = rdd.map(lambda p: (self.closestPoint(p, self.center), (p, 1)))
        sol_tmp = closest.take(self.data_num)
        #執行reduce過程，該過程的目的是重新求取中心點，生成的也是rdd
        pointStats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        #生成新的中心點
        newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()
       
        #設置新的中心點
        for (iK, p) in newPoints:
            self.center[iK] = p
        #更新分群解
        for i in range(len(sol_tmp)):
            self.sol[i] = sol_tmp[i][0]
        #更新SSE
        #self.SSE = self.cal_SSE()

#適者生存
def selection(chromosomes, Ps, data):
    size = len(chromosomes)
    new_populations = []
        
    #計算個染色體的適應值,並統計存活率
    for i in range(size):
        chromosomes[i].fitness = chromosomes[i].cal_fitness(data)
    #存活率
    print('survival rate:', Ps*100, '%')

    print('Before Selection:')
    chromosomes = sorted(chromosomes, reverse=True, key=lambda elem: elem.fitness)
    for i in range(len(chromosomes)):
        print('chromosome', i, "'s fitness value", chromosomes[i].fitness)

    #找出(存活率*個體數)個適應值的染色體
    #適應值越大越容易存活
    for i in range(int(population_size*Ps)):
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
def crossover(data, chromosomes, Pc):
    numOfInd = len(chromosomes)
    #根據交配得到數量並隨機選出染色體
    index = random.sample(range(0, numOfInd - 1), int(Pc * numOfInd))
    
    new_chromosomes = []
    for i in range(len(index)):  # do how many time
        new_chromosomes = doCrossover(data, chromosomes, i, index)
        
    return new_chromosomes


def doCrossover(data, chromosomes, i, index):
    length = chromosomes[0].sol.shape[0]
    cut = random.randint(1, length - 2)
    #依取樣順序跟隔壁交換基因(每一筆資料的分群) => sol為基因
    parent1 = chromosomes[index[i]]
    parent2 = chromosomes[index[(i + 1) % len(index)] % length]
    child1 = Chromosome(data, num_cluster)
    child2 = Chromosome(data, num_cluster)
        
    p1 = list(parent1.sol)
    p2 = list(parent2.sol)
    c1 = p1[0:cut] + p2[cut:length]
    c2 = p1[cut:length] + p2[0:cut]
    child1.sol = np.array(c1)
    child2.sol = np.array(c2)
        
    # 計算child適應值
    child1.fitness = child1.cal_fitness(data)
    child2.fitness = child2.cal_fitness(data)
        
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

if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print("Usage: kmeans <iris/abalone> <k> <iterations>", file=sys.stderr)
        sys.exit(-1)

    fitness_his = []
    sse_his = []
    ari_his = []
    cen_his = []
    population = []
    if sys.argv[1] == 'abalone':
        data, labels = load_abalone()
    num_cluster = int(sys.argv[2])
    MAX_GEN = int(sys.argv[3])

    tic = time.perf_counter()

    rdd = sc.parallelize(data)
    rdd.cache()
    for i in range(population_size):
        population.append(Chromosome(data, num_cluster))
        #population[i].center = rdd.takeSample(False, num_cluster, 1)
        population[i].cal_solution(rdd)
        print('<the', i+1, 'th chromosome has been initialized.>')

    #for each generation
    for i in range(MAX_GEN):
        print('[the', i+1, 'th generation]')
        population = selection(population, 0.8, data)
        population = crossover(data, population, 0.8)
            
        print('After Crossover:')
        for i in range(len(population)):
            print('chromosome', i, "'s fitness value", population[i].fitness)
                        
        #for each chromosome
        for j in range(population_size):
            #population[j].mutation()
            population[j].KMO(rdd)
            print('the', j+1, "'s KMO has been finished.")
        #找出最大適應值的染色體
        population = sorted(population, reverse=True, key=lambda elem: elem.fitness)
        print()
        print('Fitness value:', population[0].fitness)
        print('=======================================')
    
    toc = time.perf_counter()
    ari = adjusted_rand_score(labels, population[0].sol)
    spend_time = str(1000*(toc - tic))
    print('=======================================')
    print('Adjusted Rand Index:', ari)
    print('Sum of Square Error:', population[0].cal_SSE(data))
    print("Comuptation Time: " + spend_time + "ms")
    print('=======================================')