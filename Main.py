from Kmeans import Kmeans
from Genetic_Kmeans import Genetic_Kmeans, Chromosome

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, fowlkes_mallows_score, jaccard_similarity_score, silhouette_score, adjusted_rand_score

random.seed(0)
ALGO = 0

def load_abalone():
    df = pd.read_csv('data/abalone.csv')
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



def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)



def show_performance_indices(data, labels, predicts, sse, spend_time, name):
    output = "Final SSE: " + str(sse)
    output += "\nSpend Time: " + spend_time
    output += "\n[External Index]"
    output += "\nAdjusted Rand Index: " + str(adjusted_rand_score(labels, predicts))
    output += "\nJaccard Index: " + str(jaccard_similarity_score(labels, predicts)) 
    output += "\nFowlkes Mallows Index: " + str(fowlkes_mallows_score(labels, predicts))
    output += "\n[External Index]"
    output += "\nSilhouette Index: " + str(silhouette_score(data, predicts))
    output += "\nDavies Bouldin Index:" + str(davies_bouldin_score(data, predicts))
    print(output)

    file = open("reports/" + name + "_performance_report.txt", "w")
    file.write(output)
    file.close()


def main(argv):
    if len(argv) < 4:
        print('Please select your clustering algorithm (ka / gka) , dataset (iris / abalone) , and iteration !')   
        quit()
 
    #選擇演算法
    if argv[1] == "KA" or argv[1] == "ka":
        ALGO = 0
    elif argv[1] == "GKA" or argv[1] == "gka":
        ALGO = 1
    else:
        print("You can't select this algorithm .")
        quit()

    #設定散布圖顏色
    color = ['r', 'g', 'b']
    palette = []
    for i,c in enumerate(colors.cnames):
        palette.append(c)
    color_index = random.sample(range(100), 100)
    for i in range(len(color_index)):
        if palette[color_index[i]] != 'red' and palette[color_index[i]] != 'green' and palette[color_index[i]] != 'blue':
            color.append(palette[color_index[i]])
            
    iterations = int(argv[3])
    #選擇分群數
    if ALGO == 0:
        KA = Kmeans()
        if argv[2] == 'iris':
            iris = datasets.load_iris()
            data = iris['data']
            labels = iris['target']
            num_cluster = 3

            cen, predicts, cen_his, sse, c_init, spend_time = KA.fit(X_train=data, num_of_centers=num_cluster, nIter=iterations)
            
            label_centers = np.zeros([3, data.shape[1]])
            #更新每一群重心
            for c in range(num_cluster):
                temp = np.zeros([1, data.shape[1]])
                count = 0
                for r in range(data.shape[0]):
                    temp += int(labels[r] == c)*data[r]
                    count += int(labels[r] == c)
                label_centers[c] = temp.reshape(-1,) / count

            pca = PCA(n_components=2, random_state=1)
            pca.fit(data)
            new_data = pca.transform(data)
            final_center = pca.transform(cen)
            new_label_centers = pca.transform(label_centers)

            target_color = ['a' for i in range(len(labels))]
            predict_color = ['a' for i in range(len(labels))]
            for c in range(num_cluster):
                for r in range(len(labels)):
                    if labels[r] == c:
                        target_color[r] = color[c]
                        
                    if predicts[r] == c:
                        predict_color[r] = color[c]

            print("[Predict]")
            #畫預測分群
            print(predicts)
            show_performance_indices(data, labels, predicts, sse[len(sse)-1], spend_time, "ka_iris")

            #畫SSE
            plt.plot(range(iterations), sse)  
            plt.ylabel('Sum of Square Error')
            plt.xlabel('Iterations')

            #畫散布圖
            plt.figure(figsize=(21,7))
            #121: 共有1列2行, 這個圖表在第一個位置
            plt.subplot(121)
            plt.scatter(new_data[:,0], new_data[:,1], c=target_color)
            yellow = plt.scatter(new_label_centers[:,0], new_label_centers[:,1], s=75, c='black', label='label cen')
            black = plt.scatter(final_center[:,0], final_center[:,1], s=100, c='orange', label='predict cen')
            plt.legend(handles=[yellow, black])
            plt.title('Actual labels and final centroids')

            plt.subplot(122)
            plt.scatter(new_data[:,0], new_data[:,1], c=predict_color)
            plt.scatter(new_label_centers[:,0], new_label_centers[:,1], s=75, c='black', label='label cen')
            plt.scatter(final_center[:,0], final_center[:,1], s=100, c='orange', label='predict cen')
            plt.legend(handles=[yellow, black])
            plt.title('Clustered labels and final centroids')

            plt.show()

        elif argv[2] == 'abalone':
            data, labels = load_abalone()
            num_cluster = 28
            cen, predicts, cen_his, sse, c_init, spend_time = KA.fit(X_train=data, num_of_centers=num_cluster, nIter=iterations)

            label_centers = np.zeros([28, data.shape[1]])
            #更新每一群重心
            for c in range(num_cluster):
                temp = np.zeros([1, data.shape[1]])
                count = 0
                for r in range(data.shape[0]):
                    temp += int(labels[r] == c)*data[r]
                    count += int(labels[r] == c)
                label_centers[c] = temp.reshape(-1,) / count

            pca = PCA(n_components=2, random_state=1)
            pca.fit(data)
            new_data = pca.transform(data)
            final_center = pca.transform(cen)
            new_label_centers = pca.transform(label_centers)

            target_color = ['a' for i in range(len(labels))]
            predict_color = ['a' for i in range(len(labels))]
            for c in range(num_cluster):
                for r in range(len(labels)):
                    if labels[r] == c:
                        target_color[r] = color[c]
                        
                    if predicts[r] == c:
                        predict_color[r] = color[c]

            print("[Predict]")
            #畫預測分群
            print(predicts)
            show_performance_indices(data, labels, predicts, sse[len(sse)-1], spend_time, "ka_abalone")

            #畫SSE
            plt.plot(range(iterations), sse)  
            plt.ylabel('Sum of Square Error')
            plt.xlabel('Iterations')

            #畫散布圖
            plt.figure(figsize=(21,7))
            #121: 共有1列2行, 這個圖表在第一個位置
            plt.subplot(121)
            plt.scatter(new_data[:,0], new_data[:,1], color = target_color)
            yellow = plt.scatter(new_label_centers[:,0], new_label_centers[:,1], s=75, color='black', label='label cen')
            black = plt.scatter(final_center[:,0], final_center[:,1], s=100, color='orange', label='predict cen')
            plt.legend(handles=[yellow, black])
            plt.title('Actual labels and final centroids')

            plt.subplot(122)
            plt.scatter(new_data[:,0], new_data[:,1], color = predict_color)
            plt.scatter(new_label_centers[:,0], new_label_centers[:,1], s=75, color='black', label='label cen')
            plt.scatter(final_center[:,0], final_center[:,1], s=100, color='orange', label='predict cen')
            plt.legend(handles=[yellow, black])
            plt.title('Clustered labels and final centroids')

            plt.show()


    elif ALGO == 1:
        if argv[2] == 'iris':
            iris = datasets.load_iris()
            data = iris['data']
            labels = iris['target']
            num_cluster = 3
            GKA = Genetic_Kmeans(num_cluster=num_cluster, MAX_GEN=iterations)
            chromosome, fitness_his, sse_his, spend_time = GKA.fit(data, labels)

            label_centers = np.zeros([3, data.shape[1]])
            #更新每一群重心
            for c in range(num_cluster):
                temp = np.zeros([1, data.shape[1]])
                count = 0
                for r in range(data.shape[0]):
                    temp += int(labels[r] == c)*data[r]
                    count += int(labels[r] == c)
                label_centers[c] = temp.reshape(-1,) / count

            pca = PCA(n_components=2, random_state=1)
            pca.fit(data)
            new_data = pca.transform(data)
            final_center = pca.transform(chromosome.center)
            new_label_centers = pca.transform(label_centers)

            target_color = ['a' for i in range(len(labels))]
            predict_color = ['a' for i in range(len(labels))]
            for c in range(num_cluster):
                for r in range(len(labels)):
                    if labels[r] == c:
                        target_color[r] = color[c]
                        
                    if chromosome.sol[r] == c:
                        predict_color[r] = color[c]

            print("[Predict]")
            #畫預測分群
            print(chromosome.sol)
            show_performance_indices(data, labels, chromosome.sol, sse_his[len(sse_his)-1], spend_time, "gka_iris")

            plt.figure(figsize=(21, 7))
            #畫SSE
            plt.subplot(121)
            plt.plot(range(iterations), sse_his)  
            plt.ylabel('Sum of Square Error')
            plt.xlabel('Generation')
            #畫適應值
            plt.subplot(122)
            plt.plot(range(iterations), fitness_his)
            plt.ylabel('Fitness value')
            plt.xlabel('Generation')

            #畫散布圖
            plt.figure(figsize=(21, 7))
            #121: 共有1列2行, 這個圖表在第一個位置
            plt.subplot(121)
            plt.scatter(new_data[:,0], new_data[:,1], c=target_color)
            yellow = plt.scatter(new_label_centers[:,0], new_label_centers[:,1], s=75, c='black', label='label cen')
            black = plt.scatter(final_center[:,0], final_center[:,1], s=100, c='orange', label='predict cen')
            plt.legend(handles=[yellow, black])
            plt.title('Actual labels and final centroids')

            plt.subplot(122)
            plt.scatter(new_data[:,0], new_data[:,1], c=predict_color)
            plt.scatter(new_label_centers[:,0], new_label_centers[:,1], s=75, c='black', label='label cen')
            plt.scatter(final_center[:,0], final_center[:,1], s=100, c='orange', label='predict cen')
            plt.legend(handles=[yellow, black])
            plt.title('Clustered labels and final centroids')

            plt.show()

        elif argv[2] == 'abalone':
            data, labels = load_abalone()
            num_cluster = 28
            GKA = Genetic_Kmeans(num_cluster=num_cluster, MAX_GEN=iterations)
            chromosome, fitness_his, sse_his, spend_time = GKA.fit(data, labels)

            label_centers = np.zeros([28, data.shape[1]])
            #更新每一群重心
            for c in range(num_cluster):
                temp = np.zeros([1, data.shape[1]])
                count = 0
                for r in range(data.shape[0]):
                    temp += int(labels[r] == c)*data[r]
                    count += int(labels[r] == c)
                label_centers[c] = temp.reshape(-1,) / count

            pca = PCA(n_components=2, random_state=1)
            pca.fit(data)
            new_data = pca.transform(data)
            final_center = pca.transform(chromosome.center)
            new_label_centers = pca.transform(label_centers)

            target_color = ['a' for i in range(len(labels))]
            predict_color = ['a' for i in range(len(labels))]
            for c in range(num_cluster):
                for r in range(len(labels)):
                    if labels[r] == c:
                        target_color[r] = color[c]
                        
                    if chromosome.sol[r] == c:
                        predict_color[r] = color[c]

            print("[Predict]")
            #畫預測分群
            print(chromosome.sol)
            show_performance_indices(data, labels, chromosome.sol, sse_his[len(sse_his)-1], spend_time, "gka_abalone")

            #畫SSE
            plt.plot(range(iterations), sse_his)  
            plt.ylabel('Sum of Square Error')
            plt.xlabel('Generation')

            #畫適應值
            plt.plot(range(iterations), fitness_his)
            plt.ylabel('Fitness value')
            plt.xlabel('Generation')

            #畫散布圖
            plt.figure(figsize=(21,7))
            #121: 共有1列2行, 這個圖表在第一個位置
            plt.subplot(121)
            plt.scatter(new_data[:,0], new_data[:,1], color = target_color)
            yellow = plt.scatter(new_label_centers[:,0], new_label_centers[:,1], s=75, color='black', label='label cen')
            black = plt.scatter(final_center[:,0], final_center[:,1], s=100, color='orange', label='predict cen')
            plt.legend(handles=[yellow, black])
            plt.title('Actual labels and final centroids')

            plt.subplot(122)
            plt.scatter(new_data[:,0], new_data[:,1], color = predict_color)
            plt.scatter(new_label_centers[:,0], new_label_centers[:,1], s=75, color='black', label='label cen')
            plt.scatter(final_center[:,0], final_center[:,1], s=100, color='orange', label='predict cen')
            plt.legend(handles=[yellow, black])
            plt.title('Clustered labels and final centroids')

            plt.show()

    else:
        quit()



if __name__ == '__main__':
    main(sys.argv)