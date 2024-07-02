from sentiment_analysis_spanish import sentiment_analysis
import pandas as pd

def clasificar_sentimiento(sentimiento):
    if 0 <= sentimiento < 0.5:
        return 'Negativo'
    elif 0.5 <= sentimiento <= 0.7:
        return 'Neutral'
    elif 0.7 < sentimiento <= 1:
        return 'Positivo'
    else:
        return 'No definido'

def analizar_sentimiento(data):
    sas = sentiment_analysis.SentimentAnalysisSpanish()
    data['Sentimiento'] = data['TEXTO_STRING'].apply(lambda x: clasificar_sentimiento(sas.sentiment(x)))
    return data

def promediar_sentimiento(data):
    positivos = round(100 * (data['Sentimiento'] == 'Positivo').mean(), 2)
    neutrales = round(100 * (data['Sentimiento'] == 'Neutral').mean(), 2)
    negativos = round(100 * (data['Sentimiento'] == 'Negativo').mean(), 2)
    return pd.Series({'Positivos %': positivos, 'Neutral %': neutrales, 'Negativos %': negativos})