#aqui va cargar la data, quitar la columna anio y la vectorizacion
import pandas as pd
import re

# Preprocesado y modelado
from nltk.corpus import stopwords
import gensim

# cambiador de formatos para factorizar un dataset
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

spanish_stopwords = list(stopwords.words('spanish'))
spanish_stopwords.extend(("xq", "oye", "dale", "dele", "ba", "abc", "nan", "na"))
# print(spanish_stopwords)

def cargar_data(archivo):
    tipo_archivo = archivo.split(".")[-1]
    archivo = f".\{archivo}"
    if tipo_archivo not in ["csv", "xlsx"]:
        return f"El archivo no es un CSV o un XLSX. Extensiones válidas: CSV, XLSX"
    elif tipo_archivo == "csv":
        data = pd.read_csv(archivo, sep=",")
        return data
    elif tipo_archivo == "xlsx":
        data = pd.read_excel(archivo, sep=",")
        return data

def quitar_columnas(data):
    #if 'Año' in data.columns:
    #    data.rename(columns={'Año': 'YEAR'}, inplace=True)
    #elif 'AÑO' in data.columns:
    #    data.rename(columns={'AÑO': 'YEAR'}, inplace=True)
    #data.dropna(inplace=True)

    pocos_subniveles = []
    for columna in data.columns:
        if columna == 'TEXTO_TOKEN':
            data[columna] = data[columna].apply(lambda x: ' '.join(x))
        print(f'Columna {columna}: {data[columna].nunique()} subniveles')
        if data[columna].nunique() < 2: #verificar pocos subniveles
            pocos_subniveles.append(columna)

    data.drop(columns = pocos_subniveles, inplace=True) #eliminar columnas con pocos subniveles
    print(f"Columnas eliminadas, contienen pocos subniveles: {pocos_subniveles}")

    return data

#contando  desde cero
def elegir_columna(data, indice_col):
    columna_trabajar = data.iloc[:, indice_col]
    return columna_trabajar

#contando desde  1
#def elegir_columna(data, indice_col):
#    indice_col = indice_col - 1
#    columna_trabajar = data.iloc[:, indice_col]
#    return columna_trabajar



# def eliminar_stopwords(data):
#     spanish_stopwords = list(stopwords.words('spanish'))
#     spanish_stopwords.extend(("xq", "oye", "dale", "dele", "ba", "abc", "nan", "na"))

def limpiar_tokenizar(texto):
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', texto)
    # Se convierte todo el texto a minúsculas
    nuevo_texto = nuevo_texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\¡!\\"”\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\
    \\\.\\/\\:\\;\\<\\=\\>\\¿?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~—]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # cambio de - por nada para casos de palabras 
    nuevo_texto = re.sub('-', '', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 2]
    # Eliminación de stopwords
    nuevo_texto = [token for token in nuevo_texto if token not in spanish_stopwords]
    # quitar los token que sean solamente de una letra repetidas 2 o mas veces
    nuevo_texto = [token for token in nuevo_texto if not re.fullmatch(r'(.)\1*', token)]
    
    return(nuevo_texto)

def eliminar_vacios(data, token_column='TEXTO_TOKEN'):
    print(f'Tamaño del set antes de eliminar las filas menor a 3 tokens: {data.shape}')
    data = data[data[token_column].apply(len) > 3]
    print(f'Tamaño del set después de eliminar las filas menor a 3 tokens: {data.shape}')
    return data #ver darta despues de invocar esto

def join_tokens(tokens):
    if isinstance(tokens, list):
        return ' '.join(tokens)
    else:
        return ''

