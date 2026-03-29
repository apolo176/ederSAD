# -*- coding: utf-8 -*-
import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import pickle
import time
import json
import csv
import os
from colorama import Fore

# Sklearn
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone

from imblearn.pipeline import Pipeline as ImbPipeline

# Nltk
import nltk

# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
import re

# Descargas necesarias de NLTK
nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('stopwords')
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Si es una matriz dispersa de scipy, la convertimos a array de numpy
        if hasattr(X, "toarray"):
            return X.toarray()
        return X
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, language='spanish'):
        self.language = language
        self.stop_words = set(nltk.corpus.stopwords.words(self.language))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Si X es una Serie de Pandas, la procesamos
        return X.apply(self._clean)

    def _clean(self, text):
        if not isinstance(text, str):
            return ""
        # 1. Minúsculas y quitar caracteres raros
        text = re.sub(r'[^\w\s]', '', text.lower())
        # 2. Tokenización simple y quitar stopwords
        tokens = nltk.word_tokenize(text)
        filtered = [w for w in tokens if w not in self.stop_words]
        return " ".join(filtered)

def signal_handler(sig, frame):
    print("\nSaliendo del programa...")
    sys.exit(0)


def parse_args():
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo (kNN, decision_tree, random_forest o naive_bayes(gaussianNB o multinomialNB))", required=False,
                       default="kNN")
    parse.add_argument("-p", "--prediction", help="Columna a predecir", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar", required=False, default="f1_macro")
    parse.add_argument("-c", "--cpu", help="Número de CPUs [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas", required=False, default=False,
                       action="store_true")
    parse.add_argument("--debug", help="Modo debug", required=False, default=False, action="store_true")

    args = parse.parse_args()

    with open('clasificador.json') as json_file:
        config = json.load(json_file)

    for key, value in config.items():
        setattr(args, key, value)
    return args


def load_data(file):
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        sys.exit(1)


# =======================================================================================================
# -------------------------------------------- Metricas -------------------------------------------------
# =======================================================================================================
def calculate_fscore(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='macro')

# Reporte de clasificacion
def calculate_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)

# Matriz de confusion
def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


# =======================================================================================================
# ------------------------------------------ PREPROCESADO -----------------------------------------------
# =======================================================================================================

# Aqui dividimos train y dev. Primero separamos la columna objetivo y las demas y luego borramos las columnas que aparezcan en drop_features
def preparar_y_dividir(data):
    print("- Dividiendo datos (Train/Dev)...")
    X = data.drop(columns=[args.prediction])
    y = data[args.prediction]

    # Borrado de columnas inutiles
    drop_cols = args.preprocessing.get("drop_features", [])
    drop_cols = [c for c in drop_cols if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Codificamos la variable objetivo 'y' si es categórica (texto)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=y.name)
        # Opcional: No tengo ni idea de para que sirbe esto pero yo lo guardo por si acaso
        with open('output/label_encoder_y.pkl', 'wb') as f:
            pickle.dump(le, f)
        print(Fore.CYAN + "Mapeo de etiquetas guardado." + Fore.RESET)

    # División estratificada. Podemos darle mas instancias a test cambiando el 0.2
    x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=args.preprocessing.get("test_size", 0.2), stratify=y, random_state=42)

    return x_train, x_dev, y_train, y_dev

# Flujo para los datos Pasito a Pasito suave suavesito
def crear_pipeline(algoritmo_nombre, x_train):
    # 1. Identificar columnas dinámicamente
    # Sacamos las de texto primero (vienen del JSON de config)
    text_cols = args.preprocessing.get("text_features", [])

    # Columnas numéricas: Todo lo que sea float o int y NO esté en text_cols
    num_cols = [c for c in x_train.select_dtypes(include=['int64', 'float64']).columns
                if c not in text_cols]

    # Columnas categóricas: Todo lo que sea object/category y NO esté en text_cols
    cat_cols = [c for c in x_train.select_dtypes(include=['object', 'category']).columns
                if c not in text_cols]

    transformers = []

    # 2. Añadir rama Numérica (si existen)
    if num_cols:
        scale_method = args.preprocessing.get("scaling", "standard").lower()
        scaler = StandardScaler()  # Default
        if scale_method == "minmax":
            scaler = MinMaxScaler()
        elif scale_method == "maxabs":
            scaler = MaxAbsScaler()

        num_transformer = ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy=args.preprocessing.get("impute_strategy", "mean"))),
            ('scaler', scaler)
        ])
        transformers.append(('num', num_transformer, num_cols))

    # 3. Añadir rama Categórica (si existen)
    if cat_cols:
        cat_transformer = ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, cat_cols))

    # 4. Añadir ramas de Texto (una por cada columna de texto)
    for col in text_cols:
        if col in x_train.columns:
            text_pipeline = ImbPipeline(steps=[
                ('limpiador', TextCleaner(language=args.preprocessing.get("language", "spanish"))),
                ('tfidf', TfidfVectorizer())
            ])
            transformers.append((f'text_{col}', text_pipeline, col))

    # Si por casualidad el JSON viene vacío de columnas útiles
    if not transformers:
        raise ValueError("No se detectaron columnas para procesar. Revisa el JSON y el CSV.")

    preprocessor = ColumnTransformer(transformers=transformers)
    # 5. Seleccionar el modelo
    if algoritmo_nombre == "knn":
        modelo = KNeighborsClassifier()
    elif algoritmo_nombre == "decision_tree":
        modelo = DecisionTreeClassifier(random_state=42)
    elif algoritmo_nombre == "random_forest":
        modelo = RandomForestClassifier(random_state=42)
    elif algoritmo_nombre == "naive_bayes":
        modelo = GaussianNB() # Solo para números (necesita DenseTransformer si hay texto)
    elif algoritmo_nombre == "multinomial_nb":
        modelo = MultinomialNB() # El rey del texto
    else:
        raise ValueError("Algoritmo no soportado")

    # 6. Seleccionar el balanceo (none para no balancear)
    sampling = args.preprocessing.get("sampling", "none")
    sampler = None
    if sampling == "undersampling":
        sampler = RandomUnderSampler(random_state=42)
    elif sampling == "oversampling":
        sampler = RandomOverSampler(random_state=42)

    # 7. Construir el Pipeline final (Preproceso -> Balanceo -> Modelo)
    pasos = [('preprocesador', preprocessor)]
    if sampler is not None:
        pasos.append(('balanceo', sampler))
    if algoritmo_nombre == "naive_bayes":
        pasos.append(('to_dense', DenseTransformer()))

    pasos.append(('clasificador', modelo))

    pipeline_final = ImbPipeline(steps=pasos)

    return pipeline_final

# =======================================================================================================
# --------------------------------------- GUARDADO Y EVALUACION -----------------------------------------
# =======================================================================================================

def save_model(gs, y_dev, x_dev):
    try:
        # 1. Guardar el mejor modelo
        with open('output/bestModel.pkcl', 'wb') as file:
            pickle.dump(gs.best_estimator_, file)
            print(Fore.CYAN + "Mejor modelo guardado como bestModel.pkcl" + Fore.RESET)

        # 2. Guardar todos los resultados de la validación cruzada
        resultados = pd.DataFrame(gs.cv_results_)
        resultados.to_csv('output/resultadosEvaluacionDev.csv', index=False)
        print(Fore.CYAN + "Resultados de todos los modelos guardados en resultadosEvaluacionDev.csv" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al guardar los archivos" + Fore.RESET)
        print(e)

# =======================================================================================================
# ------------------------------------------ RESULTADOS -------------------------------------------------
# =======================================================================================================
def mostrar_resultados(gs, x_dev, y_dev):
    y_pred = gs.predict(x_dev)
    print(Fore.MAGENTA + "\n--- RESULTADOS DEL MEJOR MODELO EN DEV ---" + Fore.RESET)
    print(Fore.YELLOW + "> Mejores parametros: " + Fore.RESET, gs.best_params_)
    print(Fore.YELLOW + "> F1-score micro: " + Fore.RESET, calculate_fscore(y_dev, y_pred)[0])
    print(Fore.YELLOW + "> F1-score macro: " + Fore.RESET, calculate_fscore(y_dev, y_pred)[1])
    if args.verbose:
        print(Fore.YELLOW + "> Informe de clasificación:\n" + Fore.RESET,
              calculate_classification_report(y_dev, y_pred))
        print(Fore.YELLOW + "> Matriz de confusión:\n" + Fore.RESET, calculate_confusion_matrix(y_dev, y_pred))




def mostrar_resultados_tabla(gs, x_train, y_train,x_dev, y_dev, algoritmo_nombre):
    print(Fore.CYAN + "\n- Extrayendo TODAS las métricas para TODOS los modelos probados..." + Fore.RESET)

    lista_resultados = []

    # gs.cv_results_['params'] tiene la lista de todos los hiperparámetros probados
    for params in gs.cv_results_['params']:
        # 1. Clonamos la "cadena de montaje" original para no ensuciar nada
        modelo_actual = clone(gs.estimator)

        # 2. Le inyectamos los hiperparámetros de esta iteración
        modelo_actual.set_params(**params)

        # 3. Entrenamos y predecimos
        modelo_actual.fit(x_train, y_train)
        y_pred = modelo_actual.predict(x_dev)

        # 4. Calculamos TODAS las métricas solicitadas
        acc = accuracy_score(y_dev, y_pred)
        # Para precision y recall usamos macro como estándar, puedes cambiarlo si te exige otro
        prec = precision_score(y_dev, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_dev, y_pred, average='macro', zero_division=0)

        # Desglose total del F1-Score
        f1_macro = f1_score(y_dev, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_dev, y_pred, average='micro', zero_division=0)
        f1_avg = f1_score(y_dev, y_pred, average='weighted', zero_division=0)  # 'weighted' es 'avg' en sklearn
        f1_none = f1_score(y_dev, y_pred, average=None, zero_division=0)  # Esto devuelve un array

        # 5. Guardamos la fila para la tabla
        lista_resultados.append({
            "Modelo": algoritmo_nombre,
            "Hiperparametros": str(params),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 (macro)": round(f1_macro, 4),
            "F1 (micro)": round(f1_micro, 4),
            "F1 (avg)": round(f1_avg, 4),
            "F1 (None)": str(np.round(f1_none, 4))  # Lo pasamos a string para que no rompa el CSV
        })

    # 6. Convertimos a DataFrame para que se vea como una tabla perfecta
    df_todos = pd.DataFrame(lista_resultados)

    # Ordenamos la tabla por el mejor Accuracy o F1-macro para que sea fácil de leer
    df_todos = df_todos.sort_values(by="F1 (macro)", ascending=False)

    print(Fore.MAGENTA + f"\n--- TABLA COMPLETA DE MODELOS PROBADOS ({algoritmo_nombre}) ---" + Fore.RESET)
    print(df_todos.to_string(index=False))

    # 7. Guardamos en CSV para la profesora
    archivo_salida = f'output/metricas_completas_{algoritmo_nombre}.csv'
    df_todos.to_csv(archivo_salida, index=False)
    print(Fore.GREEN + f"\nTabla guardada con éxito en {archivo_salida}" + Fore.RESET)

# =======================================================================================================
# ------------------------------------------- ALGORITMOS ------------------------------------------------
# =======================================================================================================
#Helper para sacar el rango
def procesar_parametros(params_crudos):
    params_limpios = {}
    for clave, valor in params_crudos.items():
        # Si detecta tu truco del diccionario min/max/step...
        if isinstance(valor, dict) and "min" in valor and "max" in valor and "step" in valor:
            # Crea la lista automáticamente (ej: del 1 al 100 de 5 en 5)
            params_limpios[clave] = list(range(valor["min"], valor["max"] + 1, valor["step"]))
        else:
            # Si es una lista normal (como ["uniform", "distance"]), la deja igual
            params_limpios[clave] = valor
    return params_limpios

def ejecutar_modelo(algoritmo_nombre, x_train, x_dev, y_train, y_dev):
    print(f"\n- Construyendo pipeline y entrenando {algoritmo_nombre} con GridSearchCV...")
    start_time = time.time()

    # Construimos la cadena de montaje
    pipeline = crear_pipeline(algoritmo_nombre, x_train)

    # Obtenemos los parámetros del JSON según el algoritmo elegido
    if algoritmo_nombre == "knn":
        params = args.kNN
    elif algoritmo_nombre == "decision_tree":
        params = args.decision_tree
    elif algoritmo_nombre == "random_forest":
        params = args.random_forest
    elif algoritmo_nombre == "naive_bayes" :
        params = args.naive_bayes
    elif algoritmo_nombre == "multinomial_nb":
        params = args.multinomial_nb
    

    params = procesar_parametros(params)

    # Entrenamos buscando los mejores hiperparámetros
    gs = GridSearchCV(pipeline, params, cv=5, n_jobs=args.cpu, scoring=args.estimator)
    gs.fit(x_train, y_train)

    end_time = time.time()
    print("Tiempo de ejecución:" + Fore.MAGENTA, round(end_time - start_time, 2), Fore.RESET + "segundos")

    # muestra resultados
    mostrar_resultados_tabla(gs, x_train,y_train, x_dev, y_dev, algoritmo_nombre)
    save_model(gs, y_dev, x_dev)

# =======================================================================================================
# --------------------------------------------- TEST ----------------------------------------------------
# =======================================================================================================

def load_model():
    try:
        #Cambiar para testear otro modelo
        with open('output/bestModel.pkcl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN + "Modelo cargado con éxito" + Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET)
        sys.exit(1)


def predict(model, data):
    print("- Realizando predicciones sobre datos nuevos...")

    if args.prediction in data.columns:
        X_test = data.drop(columns=[args.prediction])
    else:
        X_test = data

    #Quita las columnas inutiles
    drop_cols = args.preprocessing.get("drop_features", [])
    drop_cols = [c for c in drop_cols if c in X_test.columns]
    if drop_cols:
        X_test = X_test.drop(columns=drop_cols)

    # Predecimos
    prediction_numbers = model.predict(X_test)

    # AAA Para esto sirve el label encoder que habia guardado, para devolver a su estado las clases predichas
    try:
        with open('output/label_encoder_y.pkl', 'rb') as f:
            le = pickle.load(f)
        prediction_labels = le.inverse_transform(prediction_numbers)
    except FileNotFoundError:
        # Si no existe el encoder (porque y ya era numérica), usamos los números
        prediction_labels = prediction_numbers

    # 3. Guardamos los resultados con los NOMBRES
    data['Prediccion_Final'] = prediction_labels
    data.to_csv('output/data-prediction.csv', index=False)
    print(Fore.GREEN + "Predicción guardada con nombres reales en 'output/data-prediction.csv'" + Fore.RESET)

if __name__ == "__main__":
    np.random.seed(42)
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()

    os.makedirs('output', exist_ok=True)

    print("\n- Cargando datos...")
    data = load_data(args.file)

    if args.mode == "train":
        x_train, x_dev, y_train, y_dev = preparar_y_dividir(data)
        ejecutar_modelo(args.algorithm, x_train, x_dev, y_train, y_dev)

    elif args.mode == "test":
        model = load_model()
        # Asumiendo que `data` ya viene sin la columna a predecir y preprocesada,
        # o que usas una pipeline.
        predict(model, data)
    else:
        print(Fore.RED + "Modo no soportado" + Fore.RESET)
        sys.exit(1)


"""
==================================================================================================
CHULETA DE SUPERVIVENCIA - EXAMEN S.A.D. (Sistemas de Ayuda a la Decisión)
==================================================================================================

1. MÉTRICAS DE EVALUACIÓN (¿Cuál elijo y por qué?)
--------------------------------------------------
- Accuracy (Exactitud): % de aciertos totales. MIENTE si los datos están desbalanceados (ej. 99% sanos, 1% enfermos. Si digo siempre "sano", tengo 99% accuracy pero el modelo es inútil).
- Precision: De todos los que el modelo dijo que eran de la Clase X, ¿cuántos lo eran realmente? (Útil si los Falsos Positivos son muy caros/peligrosos).
- Recall (Exhaustividad): De todos los que REALMENTE eran de la Clase X, ¿cuántos encontró el modelo? (Útil en medicina: no quieres dejar escapar a un enfermo, Falsos Negativos).
- F1-Score: La media armónica entre Precision y Recall. Es la métrica REINA para clases desbalanceadas.
    * F1 'macro': Calcula el F1 para cada clase y hace la media aritmética. Trata a todas las clases por igual. USA ESTA SI LA PROFESORA TE PIDE QUE LA CLASE MINORITARIA IMPORTE TANTO COMO LA MAYORITARIA.
    * F1 'micro': Suma todos los Verdaderos Positivos, Falsos Positivos, etc., globalmente. (En multiclase, suele ser igual al Accuracy).
    * F1 'weighted': Hace la media pero dándole más peso a las clases que tienen más muestras.
    * F1 'None': Devuelve un array con la nota exacta de cada clase por separado. Ideal para ver si el modelo ignora alguna clase.

2. BALANCEO DE DATOS (Undersampling / Oversampling)
---------------------------------------------------
- ¿Para qué sirve?: Para que el modelo no se vuelva "vago" y prediga siempre la clase mayoritaria.
- ¿Qué pasa si la profesora me dice "Quita el balanceo y compara"?: 
  Ve al JSON, pon "sampling": "none", ejecuta y mira el 'F1 (None)'. Verás que la clase minoritaria (la que tiene menos datos) pasa a tener un F1 muy bajo o 0. El modelo la ha ignorado porque le sale más rentable predecir la mayoritaria.
- REGLA DE ORO: El balanceo SOLO se aplica al conjunto de entrenamiento (Train). Nunca al Dev ni al Test (por eso usamos el Pipeline de imblearn).

3. PREPROCESAMIENTO: TF-IDF vs ONE-HOT ENCODER
----------------------------------------------
- One-Hot Encoder: Crea columnas de 0s y 1s. Útil para categorías cerradas y cortas (ej. Color: Rojo, Verde, Azul).
- TF-IDF (Term Frequency - Inverse Document Frequency): Útil SOLO para TEXTO LIBRE (ej. descripciones, tweets). 
  * ¿Cómo funciona?: Cuenta cuántas veces aparece una palabra en una fila, pero la PENALIZA si esa palabra aparece en todas las filas (ej. "el", "la", "de"). Así se queda solo con las palabras clave que diferencian a las clases.

4. HIPERPARÁMETROS CLAVE (Búsqueda GridSearchCV)
------------------------------------------------
- kNN: 
  * K (n_neighbors): Busca siempre impares (para evitar empates). La mejor K suele rondar la raíz cuadrada del nº de filas.
  * weights: 'distance' da más peso a los puntos cercanos (útil para desempatar zonas densas).
- Árboles (Decision Tree / Random Forest): 
  * max_depth: Frena el crecimiento. Evita que el árbol se aprenda los datos de memoria (Overfitting).
  * min_samples_split / leaf: Exige un mínimo de datos para crear una rama/hoja. Suaviza el modelo.
- Rango de hiperparámetros: Si usamos un diccionario {"min": 1, "max": 10, "step": 2}, hacemos un barrido grueso a fino de forma automatizada.

5. ARQUITECTURA DEL CÓDIGO (Data Leakage y Pipelines)
-----------------------------------------------------
- ¿Por qué el escalador (Z-score) y el imputador van DENTRO de un Pipeline?
  Para evitar la "Fuga de Datos" (Data Leakage). El escalador debe calcular la media y desviación estándar SOLO con x_train. Luego, usa esos mismos valores guardados para transformar x_dev y Test. Si escaláramos todo el CSV antes de dividirlo, estaríamos haciendo trampa porque el modelo "vería" información del futuro.
==================================================================================================
"""