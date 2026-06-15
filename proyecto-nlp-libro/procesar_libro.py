import spacy
import pandas as pd

# Cargar modelo
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

def limpiar_y_lematizar(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        texto = f.read()
    
    doc = nlp(texto)
    
    datos = []
    for token in doc:
        # Filtrado: sin stop words, sin puntuación y sin espacios
        if not token.is_stop and not token.is_punct and not token.is_space:
            datos.append({
                "original": token.text,
                "lema": token.lemma_.lower()
            })
    
    return pd.DataFrame(datos)

# Ejecución
df_resultado = limpiar_y_lematizar("ready-player-one.txt")
print(df_resultado.head(20))
df_resultado.to_csv("libro_procesado.csv", index=False)