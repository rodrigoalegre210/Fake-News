import os
import nltk
import sys

# Definimos múltiples rutas posibles para los datos de NLTK
def setup_nltk_recursos():

    posibles_paths = [
        os.path.join(os.path.expanduser('~'), 'nltk_data'),
        os.path.join(os.path.dirname(sys.executable), 'nltk_data'),
        os.path.join(os.getcwd(), 'nltk_data'),
        'D:\\nltk_data',
        'E:\\nltk_data',
        'C:\\Users\\PrancherC\\nltk_data'
    ]

    # Intentamos encontrar una ruta existente o creamos una.
    nltk_data_path = None

    for path in posibles_paths:
        try:
            os.makedirs(path, exist_ok = True)
            nltk_data_path = path
            break

        except Exception as e:
            print(f'No se pudo usar la ruta {path}: {e}')

    if not nltk_data_path:
        raise ValueError('No se pudo encontrar o crear una ruta válida para NLTK data')
    
    # Añadimos la ruta al path de NLKT
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    # Recursos.
    recursos = ['punkt', 'stopwords', 'wordnet']

    for recurso in recursos:
        try:
            nltk.data.find(f'tokenizers/{recurso}' if recurso == 'punkt' else f'corpora/{recurso}')
            print(f'{recurso} ya está instalado.')

        except LookupError:
            print(f'Descargando {recurso}...')
            nltk.download(recurso, download_dir = nltk_data_path)

    return nltk_data_path

# Llamamos a la función.
nltk_path = setup_nltk_recursos()
print(f'NLTK data path establecido en: {nltk_path}')