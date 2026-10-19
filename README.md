# Clasificador de conejos y aves con IA, Python y TensorFlow

Réplica del tutorial **"Crea un clasificador de perros y gatos con IA, Python y Tensorflow – Proyecto completo"** del canal [Ringa Tech](https://www.youtube.com/watch?v=DbwKbsCWPSg), adaptado para clasificar **conejos** y **aves**.

**Autor:** Santiago Acosta López — Universidad de Caldas
**Docente:** Albeiro Montes Gil

## Contenido

```
├── clasificador_conejos_aves.ipynb   # Notebook de Google Colab: datos, entrenamiento, evaluación y exportación
├── web/
│   ├── index.html                    # Página que clasifica con la cámara o con una foto (TensorFlow.js)
│   └── modelo/                       # Modelo exportado (model.json + group1-shard1of1.bin)
└── README.md
```

## Qué hace el notebook

Sigue los mismos pasos del video:

1. **Datos.** Aves de CIFAR‑10 (clase `bird`) y conejos de CIFAR‑100 (clase `rabbit`), descargados desde Hugging Face (`uoft-cs/cifar10` y `uoft-cs/cifar100`) con la librería `datasets`. Como CIFAR‑100 solo tiene 600 conejos, se toma una muestra aleatoria de 600 aves para que las clases queden balanceadas (1 200 imágenes en total). También permite usar fotos propias desde `dataset/ave/` y `dataset/conejo/`.
2. **Preprocesamiento.** Escala de grises, 100×100 píxeles y normalización a 0‑1, como en el video.
3. **División.** 70 % entrenamiento, 15 % validación y 15 % prueba, estratificado. (El video usa 85/15; se agregó un conjunto de prueba para medir el resultado final con datos que el modelo nunca vio).
4. **Modelos.** Los tres del video: denso (150‑150‑1), CNN (32‑64‑128 filtros + densa de 100) y CNN con Dropout 0.5 (+ densa de 250). Salida sigmoide: 0 = ave, 1 = conejo.
5. **Entrenamiento sin aumento de datos**, donde se observa el sobreajuste.
6. **Aumento de datos** con `ImageDataGenerator` (rotación, desplazamiento, inclinación, zoom y volteo horizontal) y reentrenamiento de los tres modelos.
7. **Comparación** con TensorBoard y gráficas de precisión, tabla de resultados y **matriz de confusión** del mejor modelo sobre el conjunto de prueba.
8. **Exportación a TensorFlow.js** (`model.json` + pesos `.bin`) para la página web.
9. **Mejora con transfer learning (sección 14).** MobileNetV2 preentrenada en ImageNet, a color y 96×96, para que la página funcione con fotos reales. Ver [Corrección: fotos reales](#corrección-el-modelo-decía-ave-con-fotos-reales).

## Cómo ejecutarlo

### 1. Entrenar (Google Colab)

El repositorio ya incluye el modelo entrenado en `web/modelo/`, así que este paso solo es necesario para volver a entrenar.

1. Abrir [Google Colab](https://colab.research.google.com) → *Archivo → Subir notebook* → `clasificador_conejos_aves.ipynb`.
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.
3. *Entorno de ejecución → Ejecutar todo*. Al final se descarga `resultado_entrenamiento.zip`.
4. Copiar el contenido de la carpeta `modelo_tfjs` del zip dentro de `web/modelo/`.

### 2. Usar la página web

**En línea (GitHub Pages):** https://santiago270903.github.io/clasificador-conejos-aves/web/

**Localmente:** TensorFlow.js necesita que la página se abra por `http://`, no como archivo:

```bash
cd web
python -m http.server 8000
```

Abrir `http://localhost:8000`, presionar **Usar cámara** y apuntar a un conejo o un ave (sirve una foto en otra pantalla), o usar **Subir foto**. Para abrirla desde el celular hace falta HTTPS (por ejemplo `ngrok http 8000`, como en el video).

## Diferencias con el tutorial original

| Tutorial (perros y gatos) | Este proyecto (conejos y aves) | Motivo |
|---|---|---|
| `cats_vs_dogs` de TensorFlow Datasets (~23 000 fotos) | CIFAR‑10 `bird` + CIFAR‑100 `rabbit` (1 200 imágenes balanceadas) | No hay un dataset conejos/aves en TFDS; CIFAR es público y no requiere cuentas |
| 85 % entrenamiento / 15 % validación | 70 / 15 / 15 con conjunto de prueba | Evaluar el modelo final con datos no vistos |
| Conversión con `tensorflowjs_converter` | Función `exportar_tfjs` en el notebook (mismo formato *layers-model*) | El paquete `tensorflowjs` actual instala `tensorflow-decision-forests` y rompe el entorno de Colab |
| Gris con promedio (R+G+B)/3 en la web | Gris con pesos 0.299/0.587/0.114 en entrenamiento **y** en la web | Que el modelo reciba lo mismo en el navegador que al entrenar |
| La web reduce la foto directamente al tamaño del modelo | La web reduce la foto a 32×32 y luego la amplía | Las fotos de CIFAR miden 32×32; así la web imita lo que el modelo vio al entrenar |
| — | Modelo adicional con *transfer learning* (MobileNetV2) | El modelo entrenado desde cero no generaliza a fotos reales |
| — | Matriz de confusión y reporte de precisión/recall/F1 | Evaluación más completa |

## Resultados

Entrenamiento ejecutado en Google Colab con GPU T4 (TensorFlow 2.21, `tf_keras` 2.21). Conjuntos: 840 imágenes de entrenamiento, 180 de validación y 180 de prueba, todos balanceados (mitad aves, mitad conejos).

| Modelo | Precisión entrenamiento | Precisión validación | Brecha | Precisión prueba |
|---|---|---|---|---|
| CNN + Dropout, con aumento | 0.729 | 0.728 | 0.001 | **0.761** |
| CNN + Dropout, sin aumento | 1.000 | 0.728 | 0.272 | 0.683 |
| CNN, con aumento | 0.679 | 0.694 | −0.016 | 0.706 |
| CNN, sin aumento | 1.000 | 0.683 | 0.317 | 0.644 |
| Denso, con aumento | 0.617 | 0.611 | 0.006 | 0.639 |
| Denso, sin aumento | 0.710 | 0.561 | 0.148 | 0.617 |

**Modelo elegido:** CNN + Dropout con aumento de datos (el de mayor precisión en validación; empata con la versión sin aumento, pero esta tiene una brecha mucho mayor). Reporte sobre el conjunto de prueba:

| Clase | Precision | Recall | F1 | Imágenes |
|---|---|---|---|---|
| Ave | 0.783 | 0.722 | 0.751 | 90 |
| Conejo | 0.742 | 0.800 | 0.770 | 90 |
| **Exactitud global** | | | **0.761** | 180 |

A partir del recall, la matriz de confusión queda así: de 90 aves, 65 se clasificaron bien y 25 como conejo; de 90 conejos, 72 se clasificaron bien y 18 como ave.

### Análisis

- **Sobreajuste sin aumento de datos.** Las dos CNN llegaron al 100 % en entrenamiento pero se quedaron en 68–73 % en validación (brechas de 27 y 32 puntos). Es el mismo comportamiento que se ve en el video: con solo 840 imágenes de entrenamiento, el modelo memoriza las fotos en lugar de aprender rasgos generales.
- **El aumento de datos casi elimina la brecha.** Con `ImageDataGenerator` la precisión de entrenamiento y la de validación quedan prácticamente iguales en los tres modelos, y la precisión en prueba sube en todos (CNN + Dropout: de 0.683 a 0.761; CNN: de 0.644 a 0.706; denso: de 0.617 a 0.639).
- **Las convoluciones ayudan.** El modelo denso fue el peor en ambos escenarios, porque al aplanar la imagen pierde la relación espacial entre píxeles, que es justo lo que capturan las capas convolucionales.
- **Dropout + aumento es la mejor combinación.** Las dos técnicas de regularización juntas dieron el mejor resultado.
- **Errores equilibrados.** El modelo confunde un poco más aves con conejos (25) que al revés (18), pero no tiene un sesgo fuerte hacia una sola clase.
- **Límite del resultado (~76 %).** Las imágenes de CIFAR son de 32×32 píxeles y se convirtieron a escala de grises. A esa resolución, y sin color, muchos detalles que distinguen a un ave de un conejo se pierden. Más imágenes y de mayor resolución (por ejemplo, con la opción `USAR_CARPETA_PROPIA`) o *transfer learning* deberían mejorar el resultado.

Las gráficas de precisión por época, la matriz de confusión y ejemplos de predicciones están en el notebook (secciones 10 y 11).

## Corrección: el modelo decía "Ave" con fotos reales

**Síntoma.** En la página web, casi cualquier foto real (incluso un conejo evidente) se clasificaba como "Ave". Con una foto de un conejo, la página marcaba *Ave 75 %*.

**Causa 1: el navegador no preparaba las fotos igual que el entrenamiento.** El modelo entrenó con fotos de CIFAR de 32×32 ampliadas a 100×100, que se ven borrosas. La página reducía cada foto real directamente a 100×100, así que el modelo recibía imágenes mucho más nítidas que las de entrenamiento. Con esa diferencia, el modelo tiende a asociar el detalle fino (textura) con "ave".
*Corrección en `web/index.html`:* la página recorta el centro, reduce la foto a 32×32 por pasos (para evitar *aliasing*), la amplía con interpolación bilineal (`tf.image.resizeBilinear` con *half pixel centers*, igual que `tf.image.resize`) y convierte a gris con los mismos pesos de `tf.image.rgb_to_grayscale`. Con la misma foto del conejo, la probabilidad de "Conejo" pasó de 25 % a 45 %, igual a la que da Python con el mismo preprocesamiento.

**Causa 2: el modelo entrenado desde cero aprendió poco.** Aun con el preprocesamiento corregido, la foto seguía quedando del lado de "Ave". 840 imágenes de 32×32 en gris no alcanzan para que una red entrenada desde cero generalice a fotos reales.
*Corrección en el notebook (sección 14):* un cuarto modelo con **transfer learning**. MobileNetV2 preentrenada con ImageNet se usa congelada como extractor de características, y solo se entrena una capa de salida nueva. Usa imágenes a color de 96×96 y la misma división de datos. Los resultados de este modelo se ven en la sección 14 del notebook después de ejecutarlo; la sección 14.1 permite probar los dos modelos con fotos reales.

**Cómo activar el modelo nuevo en la página.** La página detecta sola el tamaño de entrada del modelo (100×100 gris o 96×96 a color), así que no hay que cambiar el código:

1. Ejecutar el notebook en Colab. Al final de la sección 14 se descarga `modelo_tl_tfjs.zip`.
2. Reemplazar `web/modelo/model.json` y `web/modelo/group1-shard1of1.bin` por los dos archivos de ese zip.

## Limitaciones

Las imágenes de CIFAR son de 32×32 píxeles, así que todos los modelos aprenden de fotos de muy baja resolución, y la página reduce las fotos reales a ese mismo tamaño. Esto limita lo que el modelo puede distinguir, sobre todo cuando el animal ocupa una parte pequeña de la foto: conviene que el conejo o el ave quede centrado y ocupe la mayor parte de la imagen. Con más fotos propias y de mayor resolución (opción `USAR_CARPETA_PROPIA`) los modelos deberían generalizar mejor.

## Créditos

Basado en el tutorial y el repositorio [clasificador-perros-gatos](https://github.com/ringa-tech/clasificador-perros-gatos) de Ringa Tech. Datasets: CIFAR‑10 y CIFAR‑100 (Krizhevsky, 2009).
