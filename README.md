# Clasificador Automático S.A.D.

Este proyecto implementa un flujo completo de Machine Learning automatizado para tareas de clasificación. Está diseñado para ser robusto ante fugas de datos (Data Leakage) utilizando `Pipeline` de Sklearn/Imblearn y configurable a través de un archivo JSON.

## 🛠️ Requisitos e Instalación


1. Crea y activa un entorno virtual conda / venv : Python 3.12.3
```bash
conda create -n eder python=3.12.3
conda activate eder
conda install pip
```
2. Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```
## ⚙️ Configuración (clasificador.json):

Toda la lógica de preprocesamiento, selección de algoritmos y barrido de hiperparámetros se controla desde clasificador.json.

El pipeline detectará automáticamente qué columnas son numéricas o categóricas. Tenemos que indicar en el json las de texto libre (text_features). Tambien podemos borrar columnas innecesarias indicandolo en drop_features.

## 🚀 Modos de Ejecución

El script cuenta con dos modos estrictamente separados para respetar el ciclo de vida del dato:

### 1. Modo Entrenamiento (-m train)

En este modo, el sistema carga el CSV, separa un 20% para validación (dev), construye un Pipeline completo (imputación, escalado, balanceo y modelo) y busca la mejor combinación de hiperparámetros mediante validación cruzada (GridSearchCV).

Ejemplo de uso:

```Bash
python3 clasificador.py -m train -f datos.csv -a knn -p columna_objetivo
```
Outputs generados:

#### output/bestModel.pkcl: 
El pipeline entrenado y listo para producción.

#### output/metricas_completas_knn.csv: 
Tabla exhaustiva con todas las combinaciones probadas y sus métricas calculadas sobre el conjunto Dev.

#### output/resultadosEvaluacionDev
Es el historico de validación. Nos da información sobre cada iteración.
### 2. Modo Predicción/Test (-m test)

Este modo no entrena nada. Carga el pipeline exportado en (output/bestModel.pkcl) y lo aplica sobre un conjunto de datos nuevos y ciegos. El pipeline se encarga de aplicar los mismos escalados y transformaciones exactas que aprendió durante el entrenamiento.

Ejemplo de uso:

```Bash
python3 clasificador.py -m test -f nuevos_datos_ciegos.csv -p columna_objetivo
```
Outputs generados:

#### output/data-prediction.csv 
El mismo archivo de entrada con una nueva columna que contiene las predicciones (traduciendo los números de vuelta a sus etiquetas originales si aplica).
