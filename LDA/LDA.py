import spacy
import pandas as pd
import gensim

# Cargar el modelo en español
nlp = spacy.load("es_core_news_sm")

# Función para lematizar una palabra o frase
def lemmatize_text(text):
    
    if isinstance(text, list):
        text = ' '.join(text)
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]

def creacion_LDA(data, n_topics=8):
    processed_docs = []
    for doc in data['LEMMA']:
        processed_docs.append(doc)
    print(processed_docs[:2])
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    return dictionary, entrenamiento_LDA(bow_corpus, dictionary, n_topics)
    # print(dictionary)

def entrenamiento_LDA(bow_corpus, dictionary, n_topics):
    return gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = n_topics, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)

# Iterar sobre los temas en el modelo LDA
def mostrar_temas(lda_model):
    topic_to_top_word = {}
    for idx, topic in lda_model.print_topics(-1):
        # Dividir el tema en palabras y pesos
        words_weights = topic.split(" + ")
        # Extraer las palabras y sus pesos
        words_weights = [(float(weight.split('*')[0]), weight.split('*')[1].strip('"')) for weight in words_weights]
        # Filtrar las palabras con un peso mayor a 0.1
        words_weights = [ww for ww in words_weights if ww[0] > 0.09]
        if words_weights:
            # Seleccionar la palabra con el mayor peso
            top_word = max(words_weights, key=lambda ww: ww[0])
            # Asignar la palabra con el mayor peso al tema en el diccionario
            topic_to_top_word[idx] = top_word[1]
    return topic_to_top_word

def predict_topic(data, dictionary, lda_model, topic_to_top_word):
    data['SCORE'] = ""
    data['TOPIC'] = ""
    data['name_topic'] = ""
    
    for i, row in data.iterrows():
        unseen_document = row["LEMMA"]
        if isinstance(unseen_document, list):
            unseen_document = [str(token) for token in unseen_document]
        else:
            unseen_document = [str(unseen_document)]
        
        bow_vector = dictionary.doc2bow(unseen_document)
        lda_scores = sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1])
        primary_topic = lda_scores[0][0] if lda_scores else None
        
        if i < 10:  # Mostrar solo para los primeros 10 documentos
            print(f"Documento {i}")
            print(f"Texto: {unseen_document}")
            print(f"Puntuación y tema asignado: {lda_scores}")
            print(f"Tema más representativo: {primary_topic}")
            print("\n")
        
        data.loc[i, "TOPIC"] = primary_topic
        
        if primary_topic in topic_to_top_word:
            data.loc[i, "name_topic"] = topic_to_top_word[primary_topic]
        
        # Guardar el score con 3 decimales en formato de cadena
        if lda_scores:
            score_formatted = f"{lda_scores[0][1] * 100:.2f}%"
        else:
            score_formatted = ""
        
        data.loc[i, "SCORE"] = score_formatted
    
    return data
