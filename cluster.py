from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

    
def silhouette_elbow_method(X, list_k):
    """"
    Silhouette and Elbow method to find the best K for the KMeans algorithm
    """
    errors = []
    sil = []
    for k in list_k:
        print("Iteration {} / {}".format(k, len(list_k)))
        kmeans = KMeans(k)
        kmeans.fit(X)
        errors.append(kmeans.inertia_)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, errors, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.plot()
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sil, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel("Silhouette's mesure")

list_k = list(range(2,21))

df_train_normalized = np.array(list(df_train["SrcEmbed"])) / np.linalg.norm(np.array(list(df_train["SrcEmbed"])), 2, 1, keepdims = True)
silhouette_elbow_method(np.array(list(df_train["SrcEmbed"])), list_k)