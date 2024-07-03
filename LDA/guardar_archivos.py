import os
import json
import csv
import pandas as pd

def pasar_json(nombre_archivo_csv):
        carpeta_csv = "archivos-csv"
        ruta_csv = os.path.join(carpeta_csv, nombre_archivo_csv)

        archivo_csv, extension = os.path.splitext(nombre_archivo_csv)
        nombre_archivo_json = f"{archivo_csv}.json"

        carpeta_json = "archivos-json"
        ruta_json = os.path.join(carpeta_json, nombre_archivo_json)

        if not os.path.exists(carpeta_json):    # Si no existe la carpeta que la cree
            os.makedirs(carpeta_json)

        leer_csv = pd.read_csv(ruta_csv, sep=",", header = 0)
        with open(ruta_json, "w", encoding="utf-8") as archivo_json: # Crear archivo json
            print("Datos procesados guardados en 'archivos-json'")
            json.dump(leer_csv.to_dict(orient='records'), archivo_json, indent=1, ensure_ascii=False) # Cada row separada por un espacio

def guardar_csv(df, nombre_archivo):
        carpeta_csv = "archivos-csv"
        ruta_csv = os.path.join(carpeta_csv, nombre_archivo)
        
        if not os.path.exists(carpeta_csv): # Si no existe la carpeta que la cree
            os.makedirs(carpeta_csv)

        print("Datos procesados guardados en 'archivos-csv'")

        return df.to_csv(ruta_csv, index=False)