Modelo entrenado exportado a TensorFlow.js (formato layers-model):

- model.json: arquitectura y lista de pesos
- group1-shard1of1.bin: pesos en float32

Modelo actual: MobileNetV2 con transfer learning (entrada 96x96 a color), sección 14 del notebook.
El modelo anterior (CNN + Dropout con aumento de datos, entrada 100x100 en gris) se puede volver a generar con la sección 13.
