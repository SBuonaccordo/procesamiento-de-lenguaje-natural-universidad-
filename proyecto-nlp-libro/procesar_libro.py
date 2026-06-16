import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# Carga de datos y limpieza
nlp = spacy.load("es_core_news_sm")
with open("ready-player-one.txt", "r", encoding="utf-8") as f:
    texto = f.read()

doc = nlp(texto)
sentences = [[token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct and token.text.strip()] 
             for sent in doc.sents if len(sent) > 3]

# 1. Vectorizaciones Clásicas
corpus = [" ".join(s) for s in sentences]
bow_vec = CountVectorizer().fit_transform(corpus)
tfidf_vec = TfidfVectorizer().fit_transform(corpus)

# 2. Vectorización Semántica (Word2Vec)
model = Word2Vec(sentences, vector_size=50, window=5, min_count=2, workers=4)

# 3. Generación de Gráficos para el Repo
def guardar_grafico_pca(modelo, nombre_archivo):
    vocab = list(model.wv.index_to_key)[:50] # Top 50 palabras
    vecs = model.wv[vocab]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
    plt.title("Espacio Semántico PCA (Top 50)")
    plt.savefig(nombre_archivo)

guardar_grafico_pca(model, "espacio_semantico.png")
print("Procesamiento completado y gráficos guardados.")