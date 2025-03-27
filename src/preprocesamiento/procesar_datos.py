import re
import nltk
import os
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configuración manual de la ruta de NLTK.
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok = True)
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

    if isinstance(texto, str): # Nos aseguramos que el input es un string.
        texto = texto.lower() # Convertimos a minúsculas.
        texto = re.sub(r'\d+', '', texto) # Eliminamos números.
        texto = re.sub(r'[^\w\s]', '', texto) # Eliminamos puntuaciones.
        palabras = word_tokenize(texto) # Tokenizamos el texto.
        palabras = [palabra for palabra in palabras if palabra not in stop_words] # Eliminamos stopwords.
        palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras] # Aplicamos lematización.

        return ' '.join(palabras) # Reconstruimos texto limpio.
    
    return ''

# Aplicamos limpieza al dataset.
def procesar_dataframe(df):

    # Limpiamos las columnas de texto.
    df['title'] = df['title'].astype(str).apply(limpiar_texto)
    df['text'] = df['text'].astype(str).apply(limpiar_texto)

    # Normalizamos la columna 'subject'
    df['subject'] = df['subject'].str.lower().str.strip()

    # Convertimos 'date' en formato de fecha.
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Convertimos 'label' a valores numéricos.
    df['label'] = df['label'].map({'real': 1, 'fake': 0})

    return df