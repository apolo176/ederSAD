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
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

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


# --- Métricas ---
def calculate_fscore(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='macro')


def calculate_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)


def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


# --- WORKFLOW DE PREPROCESADO ---

def preprocesar_y_dividir(data):
    """
    1. Divide estratificadamente.
    2. Imputa, escala y codifica (entrenando en train, aplicando en train y dev).
    3. Balancea SOLO el train.
    """
    print("- Dividiendo datos (Train/Dev)...")
    X = data.drop(columns=[args.prediction])
    y = data[args.prediction]

    # 1. División estratificada
    x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 2. Borrado de columnas innecesarias
    drop_cols = args.preprocessing.get("drop_features", [])
    drop_cols = [c for c in drop_cols if c in x_train.columns]
    if drop_cols:
        x_train = x_train.drop(columns=drop_cols)
        x_dev = x_dev.drop(columns=drop_cols)

    # 3. Tratamiento de valores nulos (Imputación)
    if args.preprocessing["missing_values"] == "impute":
        strategy = args.preprocessing.get("impute_strategy", "mean")
        num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns

        if len(num_cols) > 0:
            imputer = SimpleImputer(strategy=strategy)
            x_train[num_cols] = imputer.fit_transform(x_train[num_cols])
            x_dev[num_cols] = imputer.transform(x_dev[num_cols])

        # Imputar categóricas con moda (most frequent)
        cat_cols = x_train.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            x_train[cat_cols] = cat_imputer.fit_transform(x_train[cat_cols])
            x_dev[cat_cols] = cat_imputer.transform(x_dev[cat_cols])

    # 4. Codificación Categóricas a Numéricas (Label Encoding)
    cat_cols = x_train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        # Entrenamos con todos los valores posibles uniendo train y dev para evitar errores de clases nuevas
        le.fit(pd.concat([x_train[col], x_dev[col]]))
        x_train[col] = le.transform(x_train[col])
        x_dev[col] = le.transform(x_dev[col])

    # 5. Escalado
    scale_method = args.preprocessing.get("scaling", "standard")
    scaler = None
    if scale_method == "standard":
        scaler = StandardScaler()
    elif scale_method == "minmax":
        scaler = MinMaxScaler()
    elif scale_method == "maxabs":
        scaler = MaxAbsScaler()

    if scaler:
        x_train[x_train.columns] = scaler.fit_transform(x_train)
        x_dev[x_dev.columns] = scaler.transform(x_dev)

    # 6. Balanceo de clases (SOLO EN TRAIN)
    sampling = args.preprocessing.get("sampling", "none")
    if sampling == "undersampling":
        print("- Aplicando Undersampling al Train...")
        sampler = RandomUnderSampler(random_state=42)
        x_train, y_train = sampler.fit_resample(x_train, y_train)
    elif sampling == "oversampling":
        print("- Aplicando Oversampling al Train...")
        sampler = RandomOverSampler(random_state=42)
        x_train, y_train = sampler.fit_resample(x_train, y_train)

    # Guardamos los CSV post-procesados como te pidió la profesora
    train_post = pd.concat([x_train, y_train], axis=1)
    dev_post = pd.concat([x_dev, y_dev], axis=1)
    train_post.to_csv('output/trainPostProcesado.csv', index=False)
    dev_post.to_csv('output/DevPostProcesado.csv', index=False)
    print(Fore.GREEN + "Datos preprocesados y guardados en 'output/'" + Fore.RESET)

    return x_train, x_dev, y_train, y_dev


# --- FUNCIONES DE GUARDADO Y EVALUACIÓN ---

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


# --- ALGORITMOS ---

def ejecutar_modelo(algoritmo_nombre, x_train, x_dev, y_train, y_dev):
    if algoritmo_nombre == "kNN":
        modelo = KNeighborsClassifier()
        params = args.kNN
    elif algoritmo_nombre == "decision_tree":
        modelo = DecisionTreeClassifier(random_state=42)
        params = args.decision_tree
    elif algoritmo_nombre == "random_forest":
        modelo = RandomForestClassifier(random_state=42)
        params = args.random_forest
    elif algoritmo_nombre == "naive_bayes":
        modelo = GaussianNB()
        params = args.naive_bayes  # Leerá los hiperparámetros del JSON
    else:
        raise ValueError("Algoritmo no soportado")

    print(f"\n- Entrenando {algoritmo_nombre} con GridSearchCV...")
    start_time = time.time()

    gs = GridSearchCV(modelo, params, cv=5, n_jobs=args.cpu, scoring=args.estimator)
    gs.fit(x_train, y_train)

    end_time = time.time()
    print("Tiempo de ejecución:" + Fore.MAGENTA, round(end_time - start_time, 2), Fore.RESET + "segundos")

    mostrar_resultados(gs, x_dev, y_dev)
    save_model(gs, y_dev, x_dev)


# --- TEST ---
def load_model():
    try:
        with open('output/bestModel.pkcl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN + "Modelo cargado con éxito" + Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET)
        sys.exit(1)


def predict(model, data):
    # Aquí deberías aplicar el mismo preprocesamiento que en train (scaler, etc.)
    # Para simplificar en producción, lo ideal es guardar un "Pipeline" de sklearn.
    prediction = model.predict(data)
    data = pd.concat([data, pd.DataFrame(prediction, columns=['Prediccion'])], axis=1)
    data.to_csv('output/data-prediction.csv', index=False)
    print(Fore.GREEN + "Predicción guardada con éxito" + Fore.RESET)


if __name__ == "__main__":
    np.random.seed(42)
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()

    os.makedirs('output', exist_ok=True)

    print("\n- Cargando datos...")
    data = load_data(args.file)

    if args.mode == "train":
        x_train, x_dev, y_train, y_dev = preprocesar_y_dividir(data)
        ejecutar_modelo(args.algorithm, x_train, x_dev, y_train, y_dev)

    elif args.mode == "test":
        model = load_model()
        # Asumiendo que `data` ya viene sin la columna a predecir y preprocesada,
        # o que usas una pipeline.
        predict(model, data)
    else:
        print(Fore.RED + "Modo no soportado" + Fore.RESET)
        sys.exit(1)