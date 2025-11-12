# Red Neuronal desde Cero - Conversor Celsius a Fahrenheit

Una implementación desde cero de una red neuronal sin usar librerías de Machine Learning, diseñada para convertir temperaturas de Celsius a Fahrenheit.
El objetivo era aprender por mi cuenta a hacer una red neuronal, pero no entendía bien las librerías como TensorFlow entonces para asentar conocimientos decidí desarrollar este repositorio.

## Características

- **Implementación manual** de redes neuronales sin librerías externas (solo NumPy para operaciones básicas)
- **Programación orientada a objetos**: Diseño con clases `Neurona` y `CapaSalida` para una arquitectura modular
- **Backpropagation** implementado manualmente (aunque con ayuda de Internet para las fórmulas)
- **Función de activación Leaky ReLU**
- **Normalización de datos** y división train/test
- **Métricas de evaluación**: accuracy y pérdida

## Tecnologías

- Python 3.x
- NumPy (operaciones matemáticas básicas)

## Estructura
├── scratch.py # Clases neuronas y capas, funciones para normalización y división de datos de entreno o prueba

├── prueba.ipynb # Jupyter Notebook con la red neuronal entrenada y lista para probar

├── datos.csv # Dataset

└── README.md

## Prueba

Si desea probar la conversión, entra a prueba.ipynb y baje a la sección "Predicción", el código se debería de ver así:
```
resultado = red.predict(neuronas, capa, X, cel, far)
print(f"{resultado}ºF")
```

Donde X será el valor en º Celsius a predecir.
