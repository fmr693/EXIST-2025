# 🎯 EXIST 2025 — Detección multimodal de sexismo en memes

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-XLM--RoBERTa-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema **multimodal (texto + imagen)** que detecta sexismo en memes de redes sociales, construido sobre la *shared task* internacional [EXIST 2025](https://nlp.uned.es/exist2025/) (CLEF 2025, Madrid — 244 equipos registrados de 38 países):

- **Subtarea 2.1** — ¿Es sexista el meme? → `YES / NO`
- **Subtarea 2.2** — ¿Con qué intención? → `DIRECT` (sexista en sí mismo) / `JUDGEMENTAL` (condena el sexismo)

> **Contexto honesto:** este trabajo se realizó como proyecto académico **después del cierre de la campaña oficial**, partiendo del ejercicio base publicado y siguiendo las *guidelines* oficiales (incluidas en el repo). Las predicciones se generan en el formato oficial de evaluación (PyEvALL) y los resultados se contrastaron con los publicados por los equipos participantes, quedando **por encima de la media de la tarea** y superando el baseline en **+4,8 puntos de F1**.

## 🧠 Arquitectura

```mermaid
flowchart TB
    subgraph entrada["📄 Entrada"]
        J["Texto JSON<br/>(OCR de los organizadores)"]
        I["🖼️ Imagen del meme"]
    end

    I --> OCR["🔍 OCR propio (EasyOCR)<br/>2 variantes de preprocesado ·<br/>orden por posición · deduplicación"]
    OCR --> M["Fusión de texto<br/>JSON + OCR"]
    J --> M

    M --> T["XLM-RoBERTa Twitter<br/>(198M tweets, 100 idiomas)<br/>CLS + Mean Pooling → 1536d"]
    I --> R["ResNet50 (ImageNet)<br/>transfer learning → 2048d"]

    T --> F["🔗 Fusión multimodal<br/>concat + interacción texto⊙imagen"]
    R --> F
    F --> C["BiGRU bidireccional<br/>+ atención sobre pasos de tiempo"]
    C --> E["🗳️ Ensemble 5-fold CV<br/>soft voting ponderado por F1<br/>+ CrossModalGRU (fusión temprana)"]
    E --> S["Salida jerárquica T2.1 → T2.2<br/>formato oficial EXIST (hard + soft)"]
```

### Decisiones técnicas y por qué

| Técnica | Por qué |
|---------|---------|
| **OCR propio** (EasyOCR, doble variante de contraste) | El texto oficial del dataset pierde tipografías decorativas y texto lateral. El OCR propio recupera texto omitido, lo ordena por posición (bbox) y se fusiona con el oficial solo si aporta ≥6 palabras nuevas |
| **Backbone de Twitter** (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) | Embeddings entrenados en 198M de tweets: mismo dominio (redes sociales, ES+EN) y mismo coste que el XLM-R genérico |
| **CLS + Mean Pooling concatenados** (1536d) | El CLS captura el significado global; el mean pooling, las palabras clave distribuidas. El clasificador decide cuál pesa más en cada caso |
| **Soft labels de los 6 anotadores** | Los memes con empate 3-3 no se descartan: se entrena con *soft cross-entropy* sobre la distribución real de votos, penalizando menos los errores donde ni los humanos se pusieron de acuerdo |
| **Ensemble 5-fold + voto ponderado por F1** | Diversidad real por subconjunto de datos (no solo por semilla) y más peso a los folds que mejor validan |
| **Fusión con interacción** (texto ⊙ imagen) | El producto elemento a elemento captura correlaciones directas texto-imagen que la concatenación no ve — clave en memes donde el sexismo nace del contraste entre ambos |
| **Focal Loss (γ=4) + oversampling + calibración de umbral** (T2.2) | La clase `JUDGEMENTAL` es minoritaria y ambigua: la Focal Loss concentra el aprendizaje en los ejemplos difíciles y el umbral se calibra en validación |
| **Salida jerárquica** | P(DIRECT) y P(JUDGEMENTAL) se condicionan a P(YES) de la subtarea 2.1, garantizando distribuciones coherentes que suman 1 |

## ⚡ Ejecución

```bash
pip install -r requirements.txt
python Proyecto.py
```

**Dataset:** el EXIST 2025 Memes Dataset se solicita a los [organizadores](https://nlp.uned.es/exist2025/) (no se redistribuye). Colocar las carpetas `EXIST 2025 Memes Dataset.../` y `Resultados Training.../` junto a `Proyecto.py`.

Notas de ejecución:

- Primera ejecución: descarga los modelos (~1 GB) y extrae características + OCR (15-90 min en CPU). Todo queda **cacheado** en `cache/` — las siguientes ejecuciones tardan <1 min.
- `LOAD_PRETRAINED = True` carga modelos ya entrenados desde `predictions/`; ponlo a `False` para reentrenar desde cero.
- Las predicciones se guardan en `predictions/` en el formato oficial (hard + soft JSON).

## 📊 Evaluación

- **Métricas internas:** F1 macro en validación por fold (se imprime el `classification_report` completo de ambas subtareas).
- **Contraste externo:** las predicciones sobre el test oficial se compararon con los resultados publicados de la campaña (métricas ICM-Hard/ICM-Soft y F1). Resultado por encima de la media de los equipos participantes; **+4,8 puntos de F1 sobre el baseline oficial**.

## 📂 Estructura

```
EXIST-2025/
├── Proyecto.py                        # Pipeline completo (~1.700 líneas comentadas)
├── requirements.txt
├── EXIST_2025_Lab_Guidelines.V0.4.pdf # Guidelines oficiales de la tarea
├── cache/                             # Features + OCR cacheados (generado)
└── predictions/                       # Modelos y predicciones (generado)
```
