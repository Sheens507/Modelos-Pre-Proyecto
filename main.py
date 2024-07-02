"""LLamado a todas las funciones del proyecto"""
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from Modulos.procesamiento import cargar_data, quitar_columnas, elegir_columna, limpiar_tokenizar,eliminar_vacios, vectorize_data, join_tokens
from Modulos.clusters import fit_and_evaluate, optimal_clusters, decribir_cluster,clustering_kmeans, elementos_pcluster
from Modulos.analisisSentimiento import  analizar_sentimiento, promediar_sentimiento
from Modulos.verData import data_info, ver_head
from sklearn.cluster import KMeans
import time

# funcion que escribe un mensaje en el archivo logs.txt
def logs(mensaje):
    # comprueba si el archivo logs.txt existe, si no existe lo crea
    try:
        with open("logs.txt", "r") as file:
            pass
    except:
        with open("logs.txt", "w") as file:
            pass

    # escribe el mensaje en el archivo logs.txt
    fecha = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("logs.txt", "a") as file:
        file.write(fecha + mensaje + "\n")

#Cargar data
ruta = "C:/Users/marie/OneDrive/Documentos/7mo Semestre-LAPTOP-KM78BBT8/SIC/Protecto Final/Opiniones.csv"
data = cargar_data(ruta)
data_raw = cargar_data(ruta)
logs(" - Cargar data")
data.dropna(inplace=True)
ver_head(data)

data = quitar_columnas(data)
logs(" - Quitar columnas")
data_info(data)
column_index = 3  #indice de la columna a analizar
if column_index < len(data.columns):
    selected_column = elegir_columna(data, column_index)
    logs(" - Fijar columna a trabajar")
else:
    print(f"Índice {column_index} está fuera del rango de columnas.")
    selected_column = None
    
if selected_column is not None:
    # Convertir a cadena y manejar valores nulos
    selected_column = selected_column.astype(str).fillna('')
    
    ##limpieza y tokenizacion de la data
    data['TEXTO_TOKEN'] = selected_column.apply(limpiar_tokenizar)
    logs(" - Limpiar tokenizar")
    ver_head(data)
    # Eliminar filas sin tokens
    
    data = eliminar_vacios(data)
    logs(" - Eliminar registros con tokens vacios")
    data['TEXTO_STRING'] = data['TEXTO_TOKEN'].apply(join_tokens)
    # Vectorizar los datos
    X_tfidf, vectorizer = vectorize_data(data['TEXTO_STRING']) #vectorizar la data
    print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")
    data.head(10)
    logs(" - vectorizacion")
    
    n_clusters = optimal_clusters(X_tfidf) #número óptimo de clusters
    print(f"Número óptimo de clusters: {n_clusters}")
    cluster_size =  elementos_pcluster(X_tfidf, n_clusters)
    print(f"Number of elements asigned to each cluster: {cluster_size}")
    logs(" - # Elementos por cluster")
    kmeans, lsa, X_lsa = clustering_kmeans(X_tfidf, n_clusters)
    logs(" - Clustering")
    # Evaluar el modelo de clustering
    fit_and_evaluate(kmeans, X_tfidf)
    logs(" - Evaluaciones")
    cluster_terms = decribir_cluster(kmeans, lsa, vectorizer, n_clusters)
    logs(" - Descripcion de cluster")
    for i, terms in enumerate(cluster_terms):
        print(f"Cluster {i}: {' '.join(terms)}")
    
    minibatch_kmeans = MiniBatchKMeans(
            n_clusters= n_clusters,
            n_init=1,
            init_size=1000,
            batch_size=1000,
            random_state=42  # Para reproducibilidad
        )

    minibatch_kmeans.fit(X_tfidf)
    data['Cluster'] = minibatch_kmeans.predict(X_tfidf)
    logs(" - Agrupar observaciones por cluster")
    # Analizar sentimientos
    analizar_sentimiento(data)
    logs(" - Analisis de sentimiento")
    ver_head(data)
    sentimientos_cluster = data.groupby('Cluster').apply(promediar_sentimiento)
    logs(" - Promedio por cluster de analisis de sentimiento")
    print(sentimientos_cluster)   
else:
    print("No se seleccionó ninguna columna válida para analizar.")

     
logs(" - Creacion y funcionamiento del main")
print(time.strftime("%Y-%m-%d %H:%M:%S") + " - Inicio del programa")