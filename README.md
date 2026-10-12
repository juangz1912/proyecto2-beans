# Proyecto 2: Clasificación de enfermedades en hojas de frijol

Proyecto de visión artificial para la materia Procesamiento de Datos. La idea es, a partir de una
foto de una hoja de frijol tomada con el celular, decir si está sana, si tiene *Angular Leaf Spot*
o si tiene *Bean Rust*.

Todo el trabajo está en un solo notebook: carga del dataset desde TensorFlow Datasets, análisis
exploratorio con imágenes de las tres clases, una CNN pequeña desde cero (para mover el número de
filtros) y fine-tuning de dos modelos fundacionales preentrenados en ImageNet, **VGG16** y
**ResNet50**. Al final se elige el mejor por F1 macro en validation y se evalúa una sola vez en test.

## Dataset

**Fuente:** [beans | TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/beans),
publicado por el Makerere AI Lab con el NaCRRI de Uganda
([ibean](https://github.com/AI-Lab-Makerere/ibean/)).

Se carga en una celda con `tfds.load("beans")`. No subimos las imágenes al repositorio. Si el
bucket oficial de Google responde 403, el notebook baja los mismos ZIP de Makerere desde el
mirror de Hugging Face y los abre con Keras, que sigue siendo TensorFlow.

| Split | Imágenes | Uso |
|---|---|---|
| train | 1.034 | Entrenamiento y malla |
| validation | 133 | Elegir hiperparámetros |
| test | 128 | Evaluación final, una sola vez |

Imágenes RGB de 500 × 500, redimensionadas a 224 × 224 para VGG16 y ResNet50.

### Variable objetivo

`label`, tres clases (problema **multiclase supervisado**):

| Clase | Nombre | Qué es |
|---|---|---|
| 0 | `angular_leaf_spot` | Mancha angular (*Pseudocercospora griseola*) |
| 1 | `bean_rust` | Roya del frijol (*Uromyces appendiculatus*) |
| 2 | `healthy` | Hoja sana |

Las clases están casi balanceadas (432 / 436 / 427 en el total que usamos; el catálogo cita 1.296
y nosotros tenemos 1.295 porque un archivo de `healthy` no era imagen). Accuracy sirve; igual
reportamos F1 macro.

## Metodología

1. EDA: conteos por split, tres fotos de cada clase, forma de las imágenes y brillo medio.
2. Preprocesamiento: resize a 224, `preprocess_input` propio de cada red (no un `/255` genérico
   para VGG/ResNet), aumento suave solo en train (flip horizontal y rotación chica).
3. **CNN desde cero** con malla de filtros `{16, 32}` y batch `{16, 32}`. Es el piso, y es donde
   sí se mueve el número de filtros.
4. **VGG16** y **ResNet50** con ImageNet, base congelada y cabeza nueva (GAP + dropout + densa +
   softmax de 3). Malla de batch `{16, 32}` y learning rate `{1e-3, 1e-4}`.
5. Fine-tuning del ganador: se descongela el último bloque con learning rate `1e-5`.
6. Test una sola vez.

No hay 10 folds: cada epoch de VGG16 en CPU se va a más de un minuto y el dataset ya trae un
split de validation hecho por los autores.

## Resultados

Métricas de **validation** (ahí se eligió el modelo):

| Modelo | batch | lr | F1 macro | Accuracy | Tiempo |
|---|---|---|---|---|---|
| CNN filtros=16 | 16 | 0.001 | 0.6441 | 0.6541 | 43 s |
| CNN filtros=16 | 32 | 0.001 | 0.6589 | 0.6617 | 46 s |
| CNN filtros=32 | 16 | 0.001 | 0.7075 | 0.7143 | 88 s |
| CNN filtros=32 | 32 | 0.001 | 0.6489 | 0.6541 | 91 s |
| VGG16 | 32 | 0.001 | 0.9005 | 0.9023 | 552 s |
| VGG16 | 16 | 0.0001 | 0.7865 | 0.7970 | 534 s |
| **ResNet50** | **32** | **0.001** | **0.9326** | **0.9323** | **166 s** |
| ResNet50 | 16 | 0.0001 | 0.9176 | 0.9173 | 182 s |

Después se descongelaron las últimas 20 capas del ResNet ganador (lr 1e-5). Validation subió a
F1 **0.9775**. Esa es la versión que se mandó a test.

**Test** (128 fotos, nunca usadas para entrenar ni para elegir):

| Métrica | Valor |
|---|---|
| Accuracy | 0.9297 |
| F1 macro | 0.9302 |
| Precision macro | 0.9324 |
| Recall macro | 0.9300 |

`healthy` casi no se falla (precision 1.00, recall 0.98). Las dos enfermedades se confunden entre
sí (8 casos). **Ninguna hoja enferma se clasificó como sana**, que es el error caro en el cultivo.

**Modelo ganador:** ResNet50 preentrenada en ImageNet, cabeza nueva y fine-tuning del último bloque.

## Estructura del repositorio

```
proyecto2-beans/
├── README.md
└── Proyecto2_Beans_Vision_Artificial.ipynb
```

## Cómo reproducirlo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juangz1912/proyecto2-beans/blob/main/Proyecto2_Beans_Vision_Artificial.ipynb)

**Opción 1:** abrir el notebook en Colab (con GPU, si se puede) y *Entorno de ejecución > Ejecutar todas*.
La celda de carga baja `beans` sola.

**Opción 2 (local):**

```bash
git clone https://github.com/juangz1912/proyecto2-beans.git
cd proyecto2-beans
pip install tensorflow tensorflow-datasets numpy matplotlib pillow scikit-learn
jupyter notebook Proyecto2_Beans_Vision_Artificial.ipynb
```

`random_state` / semillas en 42. En Colab con GPU el entrenamiento debería quedar en unos 15-25
minutos. En CPU se va más cerca de una hora.

## Integrantes

- Juan Jose Giraldo
- Alejandro Hernandez
- Alejandro Marin

**Curso:** Procesamiento de Datos
**Programa:** Ingeniería de Sistemas, Facultad de Ingenierías
**Universidad de Medellín**
