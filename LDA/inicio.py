import pandas as pd
import time
import os
from procesamiento import cargar_data, quitar_columnas, elegir_columna, limpiar_tokenizar, eliminar_vacios
from LDA import lemmatize_text, creacion_LDA, mostrar_temas, predict_topic

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
    with open("logs.txt", "a") as file:
        file.write(time +mensaje + "\n")

# aqui legara si el cliente quiero solo negativo o positivos o ambos
def inicio(data, columna, cluster=False):
    if (data == None or data == "" or columna == None or columna == ""):
        
        return "No se ha ingresado ningun dato"
    
    datos = cargar_data(data)
    if isinstance(datos, str):
        print(datos)
        return
    
    # Continuar con el procesamiento del DataFrame `data`
    print(datos.head())  # Ejemplo de procesamiento del DataFrame
    
    if (type(cluster) != int or type(columna) != int):
        try:
            cluster = int(cluster)
            columna = int(columna)
        except:
            return "El dato en cluster no es un numero"
        
    # si cluster es un numero menor a 1 devolver mensaje de error
    if (cluster < 3):
        return "El cluster debe ser mayor a 2"
    
    # si cluster es mayor a la cantidad de datos devolver mensaje de error
    # df = pd.read_csv(data)
    if (cluster > len(datos)):
        return "El cluster debe ser menor a la cantidad de datos"
    
    # eliminamos las rows con valores nulos
    datos.dropna(inplace=True)

    # verificamos si la columna a trabajar existe
    if len(datos.columns) < columna:
        return "La columna no existe"
    
    # elegimos la columna a trabajar
    columna_trabajar = elegir_columna(datos, columna)

    # quitamos columnas con pocos subniveles
    datos = quitar_columnas(datos)
    print(datos.head())  # Ejemplo de procesamiento del DataFrame
    # print(type(columna_trabajar))

    # Convertir la Serie a un DataFrame
    columna_trabajar = columna_trabajar.to_frame()
    print(columna_trabajar.head())
    columna_trabajar['TEXTO_TOKEN'] = columna_trabajar.iloc[:, 0].fillna('').astype(str).apply(limpiar_tokenizar)
    print(columna_trabajar.head())
    
    # quito las rows con menos de 3 tokens
    columna_trabajar = eliminar_vacios(columna_trabajar)
    # print(columna_trabajar.head())

    # lematizo los tokens
    columna_trabajar['LEMMA'] = columna_trabajar['TEXTO_TOKEN'].apply(lemmatize_text)
    print(columna_trabajar.head())

    # creacion del modelo LDA
    diccionario, lda_model = creacion_LDA(columna_trabajar, cluster)

    # obtengo los temas con mas caloracion de cada cluster
    temas = mostrar_temas(lda_model)
    print(temas)

    # unimos la columnas TEXTO_TOKEN y LEMMA al DataFrame original
    datos = pd.concat([datos, columna_trabajar], axis=1)

    # dropea la columna TEXTO_TOKEN que sean null
    datos = datos.dropna(subset=['TEXTO_TOKEN'])

    # cluster asignado a cada fila
    datos_final = predict_topic(datos, diccionario, lda_model, temas)
    # guarda el DataFrame en un archivo CSV
    datos_final.to_csv("datos_procesados.csv", index=False, encoding="utf-8")
    print("Datos procesados guardados en 'datos_procesados.csv'")

if __name__ == "__main__":
    T_inicio = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " - Inicio del programa")
    cosa = inicio("Opiniones.csv", 4, 8)
    print(cosa)
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " - Fin del programa")
    print("Tiempo de ejecución:", time.time() - T_inicio)