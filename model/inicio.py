import pandas as pd
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
    with open("logs.txt", "a") as file:
        file.write(time +mensaje + "\n")

# aqui legara si el cliente quiero solo negativo o positivos o ambos
def inicio(data, cluster=False):
    if (data == None):
        
        return "No se ha ingresado ningun dato"
    
    # si datos es un archivo distinto a csv devolver menaje de error
    if (data.split(".")[1] != "csv"):
        return "El archivo no es un csv"
    if (type(cluster) != int):
        try:
            cluster = int(cluster)
        except:
            return "El cluster no es un numero"
        
    # si cluster es un numero menor a 1 devolver mensaje de error
    if (cluster < 2):
        return "El cluster debe ser mayor a 2"
    
    # si cluster es mayor a la cantidad de datos devolver mensaje de error
    df = pd.read_csv(data)
    if (cluster > len(df)):
        return "El cluster debe ser menor a la cantidad de datos"
    
    # obtener los headers del archivo
    headers = df.columns

print(time.strftime("%Y-%m-%d %H:%M:%S") + " - Inicio del programa")