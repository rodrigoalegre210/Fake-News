import re
import nltk
import os
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Configuración manual de la ruta de NLTK.
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Descargamos recursos de NLTK.
nltk.download('stopwords', download_dir = nltk_data_path)
nltk.download('punkt', download_dir = nltk_data_path)
nltk.download('wordnet', download_dir = nltk_data_path)

# Configuramos stopwords en inglés.
stop_words = set(stopwords.words('english'))

# Inicializamos lematizador.
lemmatizer = WordNetLemmatizer()

# Función para limpiar y procesar un texto.
def limpiar_texto(texto):

    if not isinstance(texto, str) or texto.strip() == '':
        return '' # Evitamos errores con valores nulos o no string.
    
    texto = texto.lower() # Convertimos a minúsculas.
    texto = re.sub(r'\d+', '', texto) # Eliminamos números.
    texto = re.sub(r'[^\w\s]', '', texto) # Eliminamos puntuación.
    
    # Tokenización de oraciones primero para evitar problemas con punkt.
    oraciones = sent_tokenize(texto)
    palabras = [word_tokenize(oracion) for oracion in oraciones]
    palabras = [palabra for sublist in palabras for palabra in sublist] # Aplanamos lista.

    # Filtramos stopwords y aplicamos lematización.
    palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]

    return ' '.join(palabras)

# Aplicamos limpieza al dataset.
def procesar_dataframe(df):

    # Limpiamos las columnas de texto.
    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).apply(limpiar_texto)
    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).apply(limpiar_texto)

    # Normalizamos la columna 'subject'
    if 'text' in df.columns:
        df['subject'] = df['subject'].str.lower().str.strip()

    # Convertimos 'date' en formato de fecha.
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

    # Convertimos 'label' a valores numéricos.
    if 'label' in df.columns:
        df['label'] = df['label'].map({'real': 1, 'fake': 0})

    return df