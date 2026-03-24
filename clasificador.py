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
from sklearn.naive_bayes import GaussianNB
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


def signal_handler(sig, frame):
    print("\nSaliendo del programa...")
    sys.exit(0)


def parse_args():
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo (kNN, decision_tree o random_forest)", required=False,
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
# Esto antes hacia el preprocesado podriamos meterlo si hay un desastre en el examen pero se inventa
# columnas si las necesita en test y hace cosas chungas

#def preparar_y_dividir(data):
#    """
#    1. Divide estratificadamente.
#    2. Imputa, escala y codifica (entrenando en train, aplicando en train y dev).
#    3. Balancea SOLO el train.
#    """
#    print("- Dividiendo datos (Train/Dev)...")
#    X = data.drop(columns=[args.prediction])
#    y = data[args.prediction]
#
#    # 1. División estratificada
#    x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#
#    # 2. Borrado de columnas innecesarias
#    drop_cols = args.preprocessing.get("drop_features", [])
#    drop_cols = [c for c in drop_cols if c in x_train.columns]
#    if drop_cols:
#        x_train = x_train.drop(columns=drop_cols)
#        x_dev = x_dev.drop(columns=drop_cols)
#
#    # 3. Tratamiento de valores nulos (Imputación)
#    if args.preprocessing["missing_values"] == "impute":
#        strategy = args.preprocessing.get("impute_strategy", "mean")
#        num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
#
#        if len(num_cols) > 0:
#            imputer = SimpleImputer(strategy=strategy)
#            x_train[num_cols] = imputer.fit_transform(x_train[num_cols])
#            x_dev[num_cols] = imputer.transform(x_dev[num_cols])
#
#        # Imputar categóricas con moda (most frequent)
#        cat_cols = x_train.select_dtypes(include=['object']).columns
#        if len(cat_cols) > 0:
#            cat_imputer = SimpleImputer(strategy='most_frequent')
#            x_train[cat_cols] = cat_imputer.fit_transform(x_train[cat_cols])
#            x_dev[cat_cols] = cat_imputer.transform(x_dev[cat_cols])
#
#    # 4. Codificación Categóricas a Numéricas (Label Encoding)
#    cat_cols = x_train.select_dtypes(include=['object']).columns
#    for col in cat_cols:
#        le = LabelEncoder()
#        # Entrenamos con todos los valores posibles uniendo train y dev para evitar errores de clases nuevas
#        le.fit(pd.concat([x_train[col], x_dev[col]]))
#        x_train[col] = le.transform(x_train[col])
#        x_dev[col] = le.transform(x_dev[col])
#
#    # 5. Escalado
#    scale_method = args.preprocessing.get("scaling", "standard")
#    scaler = None
#    if scale_method == "standard":
#        scaler = StandardScaler()
#    elif scale_method == "minmax":
#        scaler = MinMaxScaler()
#    elif scale_method == "maxabs":
#        scaler = MaxAbsScaler()
#
#    if scaler:
#        x_train[x_train.columns] = scaler.fit_transform(x_train)
#        x_dev[x_dev.columns] = scaler.transform(x_dev)
#
#    # 6. Balanceo de clases (SOLO EN TRAIN)
#    sampling = args.preprocessing.get("sampling", "none")
#    if sampling == "undersampling":
#        print("- Aplicando Undersampling al Train...")
#        sampler = RandomUnderSampler(random_state=42)
#        x_train, y_train = sampler.fit_resample(x_train, y_train)
#    elif sampling == "oversampling":
#        print("- Aplicando Oversampling al Train...")
#        sampler = RandomOverSampler(random_state=42)
#        x_train, y_train = sampler.fit_resample(x_train, y_train)
#
#    # Guardamos los CSV post-procesados como te pidió la profesora
#    train_post = pd.concat([x_train, y_train], axis=1)
#    dev_post = pd.concat([x_dev, y_dev], axis=1)
#    train_post.to_csv('output/trainPostProcesado.csv', index=False)
#    dev_post.to_csv('output/DevPostProcesado.csv', index=False)
#    print(Fore.GREEN + "Datos preprocesados y guardados en 'output/'" + Fore.RESET)
#
#    return x_train, x_dev, y_train, y_dev

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
    # Paso 1. Identificar columnas por tipo (Texto, numero, o categorias
    num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    text_cols = args.preprocessing.get("text_features", [])
    cat_cols = x_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Pipeline para números: Imputar -> Escalar
    scale_method = args.preprocessing.get("scaling", "standard").lower()

    if scale_method == "minmax":
        scaler = MinMaxScaler()
    elif scale_method == "maxabs":
        scaler = MaxAbsScaler()
    elif scale_method in ["standard", "z-score", "zscore"]:
        scaler = StandardScaler()
    else:
        scaler = StandardScaler()

    #Estrategias: mean(Media Solo numericos), median (Mediana), most_frequent (Moda), constant ("constant", valorConst)
    num_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy=args.preprocessing.get("impute_strategy", "mean"))),
        ('scaler', scaler)
    ])

    #Paso 3. Pipeline para categorías: Imputar moda -> OneHotEncoder (ignora categorías nuevas en test)
    cat_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    #Paso 4. Texto (TF-IDF)
    #Esto necesita ser un pipeline por que patata
    def get_text_pipeline():
        return ImbPipeline(steps=[
            ('tfidf', TfidfVectorizer())
        ])
    transformers = [
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]

    # Solo añadimos la rama de texto si hay columnas definidas
    for col in text_cols:
        if col in x_train.columns:
            # TF-IDF suele trabajar columna por columna, le pasamos el nombre
            transformers.append((f'text_{col}', get_text_pipeline(), col))

    #Paso 4: Unir preprocesamiento con ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers)

    # 5. Seleccionar el modelo
    if algoritmo_nombre == "knn":
        modelo = KNeighborsClassifier()
    elif algoritmo_nombre == "decision_tree":
        modelo = DecisionTreeClassifier(random_state=42)
    elif algoritmo_nombre == "random_forest":
        modelo = RandomForestClassifier(random_state=42)
    elif algoritmo_nombre == "naive_bayes":
        modelo = GaussianNB()
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




def mostrar_resultados_tabla(gs, x_dev, y_dev, algoritmo_nombre):
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
    elif algoritmo_nombre == "naive_bayes":
        params = args.naive_bayes

    # Entrenamos buscando los mejores hiperparámetros
    gs = GridSearchCV(pipeline, params, cv=5, n_jobs=args.cpu, scoring=args.estimator)
    gs.fit(x_train, y_train)

    end_time = time.time()
    print("Tiempo de ejecución:" + Fore.MAGENTA, round(end_time - start_time, 2), Fore.RESET + "segundos")

    # muestra resultados
    mostrar_resultados_tabla(gs, x_dev, y_dev, algoritmo_nombre)
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