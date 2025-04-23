from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords

# ========== Configuración inicial ==========
nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))
nlp = spacy.load("es_core_news_sm")

# ========== Cargar el modelo ==========
try:
    with open("modelo_entrenado.pkl", "rb") as f:
        data = pickle.load(f)
    clf = data["model"]
    scaler = data["scaler"]
    features = data["features"]
    vectorizer = data["vectorizer"]
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    raise e

# ========== FastAPI app ==========
app = FastAPI()

# ========== Esquema de entrada ==========
class TextoEntrada(BaseModel):
    texto: str

# ========== Función de limpieza ==========
def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    doc = nlp(texto)
    lemas = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]
    return " ".join(lemas)

# ========== Endpoint principal ==========
@app.post("/predecir/")
def predecir_texto(entrada: TextoEntrada):
    try:
        limpio = limpiar_texto(entrada.texto)
        X_vec = vectorizer.transform([limpio])
        df = pd.DataFrame(X_vec.toarray(), columns=vectorizer.get_feature_names_out())

        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]

        X_scaled = scaler.transform(df)
        pred = clf.predict(X_scaled)
        return {"severidad_predicha": pred[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
