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
    return data.apply(lambda x: clasificar_sentimiento(sas.sentiment(' '.join(x))))

def promediar_sentimiento(data):
    positivos = round(100 * (data['SENTIMIENTO'] == 'Positivo').mean(), 2)
    neutrales = round(100 * (data['SENTIMIENTO'] == 'Neutral').mean(), 2)
    negativos = round(100 * (data['SENTIMIENTO'] == 'Negativo').mean(), 2)
    return pd.Series({'Positivos %': positivos, 'Neutral %': neutrales, 'Negativos %': negativos})