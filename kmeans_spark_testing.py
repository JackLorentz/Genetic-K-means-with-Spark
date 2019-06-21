from __future__ import print_function

import sys
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkConf
def parseVector(line):
    return np.array([float(x) for x in line.split(',')])


def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

def dist(center_point,p):
    return np.sum(np.square(center_point[p[0]]-p[1][0]))
        


if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print("Usage: kmeans <file> <k> <iteration>", file=sys.stderr)
        sys.exit(-1)
    
    temp=np.loadtxt(sys.argv[1],dtype=np.float,delimiter=",")

    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
    
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    #sys.argv[1]sys.argv[2]
    '''
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    data = lines.map(parseVector).cache()
    '''
    data=sc.parallelize(temp).persist()
    K =int(sys.argv[2])#28
    iteration = int(sys.argv[3])#10
    tic = time.perf_counter()
    kPoints = data.takeSample(False, K, 1)
    #tempDist = 1.0
    
    for i in range(iteration):
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p,1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()      
        
        #tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)
        #dist(newPoints,closest)
        #sse=np.sum(newPoi)
        for (iK, p) in newPoints:
            kPoints[iK] = p
    toc = time.perf_counter()
    spend_time = str(1000*(toc - tic))
    print("==================================================================================================================")
    print("Comuptation Time: " + spend_time + "ms")
    print("==================================================================================================================")    
    temp=closest.collect()
    dists=[dist(kPoints,temp[i]) for i in range(len(temp))]
    sse=np.sum(dists)
    print("==================================================================================================================")
    print("sse:",sse)
    print("==================================================================================================================")
    print("Final centers: " + str(kPoints))
    print("==================================================================================================================")
    spark.stop()
