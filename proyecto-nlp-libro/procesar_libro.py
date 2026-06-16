import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Cargar modelo
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

def preparar_y_vectorizar(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        texto = f.read()
    
    doc = nlp(texto)
    
    # 1. Crear el corpus (lista de oraciones procesadas)
    corpus = []
    for oracion in doc.sents:
        lemas = [token.lemma_.lower() for token in oracion 
                 if not token.is_stop and not token.is_punct and not token.is_space]
        if len(lemas) > 3: # Filtro para oraciones con contenido relevante
            corpus.append(" ".join(lemas))
    
    # 2. Vectorización Bag of Words
    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(corpus)
    
    # 3. Vectorización TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)
    
    return corpus, X_bow, X_tfidf, tfidf_vectorizer

# Ejecución
corpus, X_bow, X_tfidf, tfidf = preparar_y_vectorizar("ready-player-one.txt")

# Mostrar resultados para verificar
print(f"Oraciones procesadas: {len(corpus)}")
print(f"Dimensiones de la matriz TF-IDF: {X_tfidf.shape}")
print(f"Primeras 5 palabras del vocabulario TF-IDF: {tfidf.get_feature_names_out()[:5]}")