import re
import os
import sys
import nltk
import pandas as pd

# Configuración de rutas de NLTK
def configurar_rutas_nltk():
    # Directorios potenciales para recursos NLTK
    posibles_rutas = [
        os.path.join(os.path.expanduser('~'), 'nltk_data'),
        os.path.join(sys.prefix, 'nltk_data'),
        os.path.join(os.path.dirname(sys.executable), 'nltk_data'),
        os.path.join(os.getcwd(), 'nltk_data')
    ]

    # Buscar o crear una ruta válida
    ruta_nltk = None
    for ruta in posibles_rutas:
        try:
            os.makedirs(ruta, exist_ok=True)
            ruta_nltk = ruta
            break
        except Exception as e:
            print(f"No se pudo usar la ruta {ruta}: {e}")

    if not ruta_nltk:
        raise ValueError("No se pudo encontrar o crear una ruta para NLTK data")

    # Añadir la ruta al path de NLTK
    if ruta_nltk not in nltk.data.path:
        nltk.data.path.append(ruta_nltk)

    return ruta_nltk

# Descargar recursos de NLTK de manera exhaustiva
def descargar_recursos_nltk(ruta_nltk):
    # Recursos necesarios
    recursos = [
        'punkt',
        'stopwords', 
        'wordnet'
    ]

    # Descargar cada recurso
    for recurso in recursos:
        try:
            nltk.data.find(f'tokenizers/{recurso}' if recurso == 'punkt' else f'corpora/{recurso}')
            print(f"{recurso} ya está instalado.")
        except LookupError:
            print(f"Descargando {recurso}...")
            try:
                nltk.download(recurso, download_dir=ruta_nltk)
            except Exception as e:
                print(f"Error descargando {recurso}: {e}")

# Configurar rutas y descargar recursos
ruta_nltk = configurar_rutas_nltk()
descargar_recursos_nltk(ruta_nltk)

# Importaciones después de configurar NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Configuramos stopwords en inglés.
stop_words = set(stopwords.words('english'))

# Inicializamos lematizador.
lemmatizer = WordNetLemmatizer()

# Función para limpiar y procesar un texto.
def limpiar_texto(texto):
    try:
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
        palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra.lower() not in stop_words]

        return ' '.join(palabras)
    except Exception as e:
        print(f"Error procesando texto: {e}")
        return ''

# Aplicamos limpieza al dataset.
def procesar_dataframe(df):
    # Crear una copia del dataframe para evitar modificaciones
    df_procesado = df.copy()

    # Limpiamos las columnas de texto.
    if 'title' in df_procesado.columns:
        df_procesado['title'] = df_procesado['title'].astype(str).apply(limpiar_texto)
    if 'text' in df_procesado.columns:
        df_procesado['text'] = df_procesado['text'].astype(str).apply(limpiar_texto)

    # Normalizamos la columna 'subject'
    if 'subject' in df_procesado.columns:
        df_procesado['subject'] = df_procesado['subject'].str.lower().str.strip()

    # Convertimos 'date' en formato de fecha.
    if 'date' in df_procesado.columns:
        df_procesado['date'] = pd.to_datetime(df_procesado['date'], errors='coerce')
        df_procesado['year'] = df_procesado['date'].dt.year
        df_procesado['month'] = df_procesado['date'].dt.month
        df_procesado['day'] = df_procesado['date'].dt.day

    # Convertimos 'label' a valores numéricos.
    if 'label' in df_procesado.columns:
        df_procesado['label'] = df_procesado['label'].map({'real': 1, 'fake': 0})

    return df_procesado