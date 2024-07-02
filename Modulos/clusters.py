from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from collections import defaultdict
from time import time

evaluations = []
evaluations_std = []


def fit_and_evaluate(km, X, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        X_densa = X.toarray() if hasattr(X, 'toarray') else X #de dispersa a densa para la evaluacion davies y calinski
        scores["Silhouette Coefficient"].append(silhouette_score(X, km.labels_, sample_size=2000))
        #error con la X, necesita una matriz densa
        scores["Davies-Bouldin Index"].append(davies_bouldin_score(X_densa, km.labels_))
        scores["Calinski-Harabasz Index"].append(calinski_harabasz_score(X_densa, km.labels_))
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)


def optimal_clusters(X_tfidf):
    # Encontrar el número óptimo de clusters basado en el método de la silueta
    silhouette_scores = []
    for n_clusters in range(2, 15):
        kmeans = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(X_tfidf)
        silhouette_avg = silhouette_score(X_tfidf, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    #numero optimo de clusters
    return np.argmax(silhouette_scores) + 2 # +2 porque empezamos desde n_clusters=2


def elementos_pcluster(X_tfidf, n_clusters):
    for seed in range(5):
        kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=100,
        n_init=1,
        random_state=seed,
    ).fit(X_tfidf)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    return cluster_sizes #lementos para cada cluster

def decribir_cluster(kmeans, lsa, vectorizer, n_clusters):
    truncated_svd = lsa.named_steps['truncatedsvd']
    
    # Aplicar la transformada de truncated_svd a kmeans.cluster_centers_
    cluster_centers_lsa = truncated_svd.transform(kmeans.cluster_centers_)
    
    # Ahora, podemos aplicar la transformada inversa
    original_space_centroids = truncated_svd.inverse_transform(cluster_centers_lsa)
    
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_desc = []
    for i in range(n_clusters):
        terms_list = [terms[ind] for ind in order_centroids[i, :10]]
        cluster_desc.append(terms_list)

    return cluster_desc


def clustering_kmeans(X_tfidf, n_clusters):
    # Crear y aplicar LSA
    truncated_svd = TruncatedSVD(n_components=100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(truncated_svd, normalizer)
    
    X_lsa = lsa.fit_transform(X_tfidf)
    
    print(f"Dimensionalidad de X_lsa: {X_lsa.shape}")
    
    explained_variance = truncated_svd.explained_variance_ratio_.sum()
    print(f"Explained variance of the SVD step: {explained_variance:.2%}")

    # Aplicar K-means a X_lsa
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_lsa)
    
    print(f"Forma de cluster_centers_: {kmeans.cluster_centers_.shape}")

    return kmeans, lsa, X_lsa

def clustering_minibatch(X_tfidf, n_clusters):
  lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
  X_lsa = lsa.fit_transform(X_tfidf)
  kmeans = MiniBatchKMeans(n_clusters = n_clusters, random_state=42)
  kmeans.fit(X_lsa)
  

  return kmeans, lsa