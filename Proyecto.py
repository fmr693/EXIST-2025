# =============================================================================
# EXIST 2025 - Sistema Multimodal de Detección de Sexismo en Memes
# =============================================================================
#
# SUBTAREA 2.1: ¿Es sexista el meme? → YES / NO
# SUBTAREA 2.2: ¿Cuál es la intención? → DIRECT / JUDGEMENTAL
#
# PIPELINE DE TRABAJO:
#   1. Carga del dataset JSON (texto extraído + metadatos)
#   2. Detección de idioma (campo 'lang': 'en' / 'es')
#   3. Extracción de características de TEXTO  → XLM-RoBERTa (multilingual)
#   4. Extracción de características de IMAGEN → ResNet50 (pretrained ImageNet)
#   5. Fusión multimodal → texto + imagen concatenados
#   6. Clasificador con Red Neuronal Recurrente (BiGRU)
#   7. Evaluación (F1, Precisión, Recall)
#   8. Generación de archivos en formato EXIST 2025
#
# DEPENDENCIAS - Ejecutar UNA VEZ en Anaconda Prompt:
#   pip install transformers scikit-learn langdetect tqdm sentencepiece
#
# NOTA: La primera ejecución descarga los modelos (~1 GB) y extrae
#       características (puede tardar 15-30 min sin GPU). Las siguientes
#       ejecuciones cargan todo desde caché (<1 min).
# =============================================================================

# =============================================================================
# SECCIÓN 1: IMPORTACIONES
# =============================================================================
# Dependencias: pip install -r requirements.txt
import os
import sys
# Fix para codificación UTF-8 en consola Windows (evita UnicodeEncodeError con
# caracteres como →, ✓, etc. cuando la terminal usa cp1252).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
# Fix para conflicto OpenMP entre PyTorch y EasyOCR en Windows/macOS.
# Debe ir ANTES de cualquier import de torch o easyocr.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =============================================================================

# ---- Rutas del proyecto ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_JSON = os.path.join(
    BASE_DIR,
    "EXIST 2025 Memes Dataset-20260421T070023Z-3-001",
    "EXIST 2025 Memes Dataset", "training", "EXIST2025_training.json"
)
TRAIN_IMG_DIR = os.path.join(
    BASE_DIR,
    "EXIST 2025 Memes Dataset-20260421T070023Z-3-001",
    "EXIST 2025 Memes Dataset", "training", "memes"
)
TEST_JSON = os.path.join(
    BASE_DIR,
    "EXIST 2025 Memes Dataset-20260421T070023Z-3-001",
    "EXIST 2025 Memes Dataset", "test", "EXIST2025_test_clean.json"
)
TEST_IMG_DIR = os.path.join(
    BASE_DIR,
    "EXIST 2025 Memes Dataset-20260421T070023Z-3-001",
    "EXIST 2025 Memes Dataset", "test", "memes"
)
GOLD_21_HARD = os.path.join(
    BASE_DIR,
    "Resultados Training-20260421T072644Z-3-001",
    "Resultados Training", "EXIST2025_training_task2_1_gold_hard.json"
)
GOLD_21_SOFT = os.path.join(
    BASE_DIR,
    "Resultados Training-20260421T072644Z-3-001",
    "Resultados Training", "EXIST2025_training_task2_1_gold_soft.json"
)
GOLD_22_HARD = os.path.join(
    BASE_DIR,
    "Resultados Training-20260421T072644Z-3-001",
    "Resultados Training", "EXIST2025_training_task2_2_gold_hard.json"
)

CACHE_DIR  = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Dispositivo de cómputo ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Modelo de texto preentrenado ----
# cardiffnlp/twitter-xlm-roberta-base-sentiment:
#   Entrenado en 198 millones de tweets en 100 idiomas con fine-tuning
#   de análisis de sentimientos en redes sociales. Produce embeddings
#   mucho más relevantes para nuestro dominio (memes/redes sociales)
#   que xlm-roberta-base entrenado en texto web genérico.
#   Mismo tamaño (125M params) → misma velocidad de extracción.
TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# ---- Hiperparámetros ----
MAX_LEN        = 128    # Longitud máxima de texto en tokens
IMAGE_SIZE     = 224    # Tamaño de entrada a ResNet50 (224×224 px)
BATCH_SIZE     = 32     # Muestras por lote de entrenamiento
EPOCHS_T21     = 30     # Épocas máximas para Tarea 2.1 (aumentado para mejor convergencia)
EPOCHS_T22     = 40     # Épocas máximas para Tarea 2.2 (aumentado)
LR             = 2e-4   # Tasa de aprendizaje (reducido para estabilizar las curvas de val)
HIDDEN_SIZE    = 128    # Neuronas en capa oculta del GRU
NUM_GRU_LAYERS = 2      # Capas del GRU (profundidad de la RNN)
DROPOUT        = 0.35   # Tasa de dropout (reducido ligeramente)
# TEXT_DIM = 1536 porque usamos CLS + mean pooling concatenados:
#   - CLS token [768-dim]: representación global de la frase
#   - Mean pooling [768-dim]: promedio de todos los tokens
#   Combinarlos da una descripción más rica que cualquiera por separado.
TEXT_DIM       = 1536   # CLS(768) + MeanPool(768) concatenados
IMAGE_DIM      = 2048   # Dimensión de salida de ResNet50
PROJ_DIM       = 256    # Dimensión de proyección común
GRU_SEQ_LEN         = 8       # Pasos de tiempo para el GRU (aumentado)
SEED                = 42
N_FOLDS_T21         = 5    # K-fold cross-validation para ensemble T2.1
                           # Reemplaza los 5 seeds: cada fold entrena en datos distintos
                           # → diversidad real por subconjunto, no solo por init aleatoria
# ---- Parámetros de OCR ----
OCR_MAX_IMG_WIDTH   = 1024    # Ancho máximo para redimensionar antes del OCR (velocidad)
OCR_CONFIDENCE_THR  = 0.30    # Confianza mínima EasyOCR por palabra
OCR_BOOST_CONTRAST  = 2.0     # Factor de mejora de contraste (variante 2 de preprocesado)
# Sufijo de versión de caché: v3_ocr_twitter = twitter backbone + OCR propio
_CACHE_VERSION = "v3_ocr_twitter"

# ---- Modo de ejecución ----
# False → entrena todos los modelos desde cero (minutos con GPU si las
#         features ya están cacheadas; sin GPU puede llevar horas)
# True  → carga los modelos ya entrenados desde OUTPUT_DIR (~1 min)
LOAD_PRETRAINED = False

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 62)
print("  EXIST 2025 — Detección de Sexismo en Memes")
print(f"  Dispositivo: {DEVICE}")
print("=" * 62)

# =============================================================================
# SECCIÓN 3: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def load_training_data():
    """
    Lee el JSON de entrenamiento y los gold labels (etiquetas de referencia).

    Incluye los 624 memes con empate 3-3 usando soft labels desde GOLD_21_SOFT.
    La columna 'soft_21' contiene [P(NO), P(YES)] derivada del ratio de votos
    de los anotadores. Para muestras con mayoría clara, coincide con one-hot.

    Retorna un DataFrame con:
      id_EXIST  → identificador único del meme
      text      → texto extraído automáticamente del meme
      lang      → idioma detectado ('en' o 'es')
      img_path  → ruta completa a la imagen
      label_21  → 1=YES / 0=NO (hard label; para muestras ambiguas, argmax del soft)
      soft_21   → [P(NO), P(YES)] distribución de votos (6 anotadores)
      label_22  → 1=DIRECT / 0=JUDGEMENTAL / -1=no aplica
    """
    print("\n[PASO 1] Cargando datos de entrenamiento...")

    with open(TRAIN_JSON, encoding='utf-8') as f:
        train_raw = json.load(f)

    with open(GOLD_21_HARD, encoding='utf-8') as f:
        gold_21_hard = {item['id']: item['value'] for item in json.load(f)}

    # Soft labels: distribución de votos de los 6 anotadores para TODAS las muestras
    with open(GOLD_21_SOFT, encoding='utf-8') as f:
        raw_soft = json.load(f)
        # Formato: [{"id": "...", "value": {"YES": 0.67, "NO": 0.33}}, ...]
        gold_21_soft = {item['id']: item['value'] for item in raw_soft}

    with open(GOLD_22_HARD, encoding='utf-8') as f:
        gold_22 = {item['id']: item['value'] for item in json.load(f)}

    records = []
    skipped_no_image = 0
    n_ambiguous = 0

    for id_exist, entry in train_raw.items():
        # Incluimos TODAS las muestras que tienen soft label (incluye los 624 empates)
        if id_exist not in gold_21_soft:
            continue

        img_path = os.path.join(TRAIN_IMG_DIR, entry['meme'])
        if not os.path.exists(img_path):
            skipped_no_image += 1
            continue

        soft_vals = gold_21_soft[id_exist]  # {"YES": float, "NO": float}
        p_yes = float(soft_vals.get('YES', 0.5))
        p_no  = float(soft_vals.get('NO',  0.5))

        # Hard label: argmax del soft (para muestras con mayoría usa el gold_hard)
        if id_exist in gold_21_hard:
            label_21 = 1 if gold_21_hard[id_exist] == 'YES' else 0
        else:
            # Muestras ambiguas (empate): hard label = argmax del soft
            label_21 = 1 if p_yes >= p_no else 0
            n_ambiguous += 1

        label_22 = -1
        if id_exist in gold_22:
            val = gold_22[id_exist]
            label_22 = 1 if val == 'DIRECT' else (0 if val == 'JUDGEMENTAL' else -1)

        records.append({
            'id_EXIST': id_exist,
            'text':     entry.get('text', '') or '',
            'lang':     entry.get('lang', 'en'),
            'img_path': img_path,
            'label_21': label_21,
            'soft_21':  [p_no, p_yes],   # [P(NO), P(YES)] para KL/soft-CE
            'label_22': label_22,
        })

    df = pd.DataFrame(records)
    n_yes = df['label_21'].sum()
    n_no  = (df['label_21'] == 0).sum()
    df_22 = df[df['label_22'].isin([0, 1])]

    print(f"  Total muestras (incl. ambiguas): {len(df)}")
    print(f"  Ambiguas con soft label incluidas: {n_ambiguous} | (imagen no encontrada): {skipped_no_image}")
    print(f"  T2.1 → YES (sexistas): {n_yes} | NO: {n_no}")
    print(f"  T2.2 → DIRECT: {(df_22['label_22']==1).sum()} | JUDGEMENTAL: {(df_22['label_22']==0).sum()}")
    return df


def load_test_data():
    """Carga el dataset de test (sin etiquetas, hay que predecirlas)."""
    with open(TEST_JSON, encoding='utf-8') as f:
        test_raw = json.load(f)

    records = []
    for id_exist, entry in test_raw.items():
        records.append({
            'id_EXIST': id_exist,
            'text':     entry.get('text', '') or '',
            'lang':     entry.get('lang', 'en'),
            'img_path': os.path.join(TEST_IMG_DIR, entry['meme']),
        })

    df = pd.DataFrame(records)
    print(f"  Muestras de test: {len(df)}")
    return df


# =============================================================================
# SECCIÓN 3b: OCR PROPIO — EXTRACCIÓN DE TEXTO DESDE IMÁGENES
# =============================================================================
#
# ¿Por qué OCR propio si el dataset ya incluye texto?
# ────────────────────────────────────────────────────
# El campo 'text' del JSON fue generado por el OCR de los organizadores.
# Tener nuestro propio OCR nos permite:
#   a) Detectar texto omitido o mal extraído (tipografías decorativas, zonas
#      no estándar como texto lateral o rotado)
#   b) Ordenar el texto de forma natural (arriba→abajo) usando coordenadas bbox
#   c) Hacer el pipeline independiente del JSON  → útil en producción real
#
# HERRAMIENTA: EasyOCR
#   - Soporta inglés y español simultáneamente (un solo Reader)
#   - Maneja tipografías decorativas de memes (Impact, Bebas Neue, etc.)
#   - Devuelve bounding boxes → podemos ordenar texto por posición Y
#   - Sin binarios externos (puro Python + PyTorch)
#   - Usa GPU automáticamente si está disponible
#   - Primera ejecución: descarga ~100 MB de modelos de lenguaje
#
# ESTRATEGIA DE 2 VARIANTES POR IMAGEN:
#   Los memes tienen dos estilos de texto predominantes:
#     Variante 1 — Original:          texto Impact blanco con borde negro
#     Variante 2 — Contraste alto ×2: texto de bajo contraste / fondos complejos
#   Las detecciones de ambas variantes se fusionan eliminando duplicados.
#   Así si una variante pierde texto (ej. blanco sobre blanco), la otra lo recupera.
#
# FUSIÓN JSON + OCR:
#   Si el OCR aporta ≥ 6 palabras nuevas respecto al JSON:
#       resultado = "{json_text} | {ocr_text}"
#   El separador ' | ' es nativo de Twitter/redes sociales → familiar para
#   XLM-RoBERTa (entrenado en 198M tweets).
# =============================================================================

def _preprocess_for_meme_ocr(pil_img):
    """
    Preprocesa una imagen de meme para maximizar la precisión de EasyOCR.

    Pasos:
      1. Upscale si la imagen es pequeña (OCR falla con texto < 20px de alto)
      2. Reducir si es muy grande (velocidad en CPU, sin pérdida de calidad OCR)
      3. Generar 2 variantes: original + contraste aumentado

    Retorna lista de arrays numpy listos para EasyOCR.
    """
    # ── 1. Tamaño mínimo: garantiza legibilidad en imágenes pequeñas ─────
    w, h = pil_img.size
    if min(w, h) < 400:
        scale   = 400 / min(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ── 2. Tamaño máximo: limita coste de cómputo en CPU ─────────────────
    w, h = pil_img.size
    if w > OCR_MAX_IMG_WIDTH:
        scale   = OCR_MAX_IMG_WIDTH / w
        pil_img = pil_img.resize((OCR_MAX_IMG_WIDTH, int(h * scale)), Image.LANCZOS)

    img_rgb  = pil_img.convert('RGB')
    variants = []

    # Variante 1: imagen original (óptima para texto Impact blanco / negro)
    variants.append(np.array(img_rgb))

    # Variante 2: contraste aumentado (rescata texto tenue o poco contrastado)
    enhanced = ImageEnhance.Contrast(img_rgb).enhance(OCR_BOOST_CONTRAST)
    variants.append(np.array(enhanced))

    return variants


def extract_ocr_text(df, split_name):
    """
    Extrae texto de las imágenes de memes con EasyOCR.

    Para cada imagen:
      1. Genera 2 variantes preprocesadas
      2. Ejecuta EasyOCR → lista de (bbox, texto, confianza)
      3. Filtra detecciones con confianza < OCR_CONFIDENCE_THR
      4. Ordena de arriba a abajo usando la coordenada Y del bbox
      5. Elimina duplicados entre variantes (comparación en minúsculas)
      6. Concatena en un único string de texto

    Resultado cacheado en: cache/ocr_text_{split_name}.json
    """
    cache_path = os.path.join(CACHE_DIR, f"ocr_text_{split_name}.json")
    if os.path.exists(cache_path):
        print(f"  [OCR] Caché encontrada: {cache_path}")
        with open(cache_path, encoding='utf-8') as f:
            return json.load(f)

    print(f"  [OCR] Iniciando EasyOCR en {len(df)} imágenes...")
    print(f"  [OCR] Primera vez: descarga ~100 MB de modelos de lenguaje")
    import easyocr
    reader = easyocr.Reader(
        ['en', 'es'],
        gpu=torch.cuda.is_available(),
        verbose=False,
    )

    img_paths = list(df['img_path'])
    ids       = list(df['id_EXIST'])
    results   = {}
    n         = len(ids)

    for i, (id_e, path) in enumerate(zip(ids, img_paths)):
        try:
            pil_img  = Image.open(path).convert('RGB')
            variants = _preprocess_for_meme_ocr(pil_img)

            # Función local: filtra palabra a palabra.
            # Criterios de descarte POR PALABRA:
            #   a) Confianza < OCR_CONFIDENCE_THR
            #   b) Primera letra es símbolo de artefacto OCR ([ @ ~ _ #)
            #   c) ≥ 14 caracteres SIN puntuación interna = palabras fusionadas
            def _read_clean(variant_arr):
                raw   = reader.readtext(variant_arr, detail=1, paragraph=False)
                clean = []
                for (bbox, word, conf) in raw:
                    w = word.strip()
                    if not w or conf < OCR_CONFIDENCE_THR:
                        continue
                    if w[0] in {'[', '@', '~', '_', '#'}:
                        continue
                    if len(w) >= 14 and not any(c in w for c in '.,-/:()'):
                        continue
                    clean.append((bbox[0][1], w, conf))
                return clean

            # Variante 1 (imagen original) — fuente de verdad principal
            all_detections = _read_clean(variants[0])

            # Variante 2 (contraste ×2) — refuerzo para imágenes oscuras.
            # Solo entra si la original detecta < 3 palabras.
            if len(all_detections) < 3:
                dets_boost = _read_clean(variants[1])
                if len(dets_boost) > len(all_detections):
                    all_detections = dets_boost

            if not all_detections:
                results[id_e] = ''
                continue

            # Ordenar de arriba a abajo (orden de lectura natural)
            all_detections.sort(key=lambda x: x[0])

            # Deduplicar: guardar solo tokens nuevos (comparación insensible a mayúsculas)
            seen_tokens = set()
            text_parts  = []
            for (_, word, _) in all_detections:
                token = word.lower()
                if len(token) > 1 and token not in seen_tokens:
                    seen_tokens.add(token)
                    text_parts.append(word)

            # Mínimo 2 tokens únicos: menos suele ser artefacto de fuente
            if len(text_parts) < 2:
                results[id_e] = ''
                continue

            ocr_result = ' '.join(text_parts)
            results[id_e] = ocr_result
            # Mostrar resultado por consola para verificar el OCR en tiempo real
            img_name = os.path.basename(path)
            preview  = ocr_result[:90] + ('...' if len(ocr_result) > 90 else '')
            print(f"    [OCR ✓] {img_name}: {preview}")

        except Exception as exc:
            results[id_e] = ''
            print(f"    [OCR !] {os.path.basename(path)}: error — {exc}")

        if (i + 1) % 100 == 0 or (i + 1) == n:
            found = sum(1 for v in results.values() if v.strip())
            print(f"    [{i+1:>4}/{n}] texto extraído en {found} imágenes hasta ahora")

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_found = sum(1 for v in results.values() if v.strip())
    print(f"  [OCR] Completado: {n_found}/{n} imágenes con texto → {cache_path}")
    return results


def merge_text_sources(json_text, ocr_text):
    """
    Fusiona el texto del JSON (organizers) con el de nuestro OCR propio.

    Lógica:
    ───────
    El JSON ya tiene texto razonablemente bueno. Nuestro OCR puede:
      a) Confirmar los mismos tokens (redundante → usar el más largo)
      b) Añadir tokens nuevos (mejorar cobertura → concatenar con ' | ')
      c) Añadir solo ruido (tokens de 1 carácter ya filtrados en OCR)

    Si el OCR aporta ≥ 6 palabras nuevas:
        "{json_text} | {ocr_text}"
    El separador ' | ' es habitual en Twitter → conocido por XLM-RoBERTa.
    """
    json_text = (json_text or '').strip()
    ocr_text  = (ocr_text  or '').strip()

    if not json_text and not ocr_text:
        return ''
    if not json_text:
        return ocr_text
    if not ocr_text:
        return json_text

    json_tokens = set(json_text.lower().split())
    ocr_tokens  = set(ocr_text.lower().split())
    new_tokens  = ocr_tokens - json_tokens

    if len(new_tokens) >= 6:
        # OCR aporta información adicional significativa (al menos 6 tokens nuevos)
        return f"{json_text} | {ocr_text}"
    # OCR es redundante: usar el texto más completo
    return json_text if len(json_text) >= len(ocr_text) else ocr_text


# =============================================================================
# SECCIÓN 4: EXTRACCIÓN DE CARACTERÍSTICAS DE TEXTO (XLM-RoBERTa)
# =============================================================================

def extract_text_features(df, split_name):
    """
    Vectoriza el texto de cada meme usando el modelo de texto configurado.

    MEJORA v2: CLS + Mean Pooling concatenados (1536-dim)
    ──────────────────────────────────────────────────────
    En lugar de usar solo el token [CLS], combinamos DOS representaciones:

    1. CLS token [768-dim]:
       Token especial al inicio de la frase cuyo estado final codifica
       el significado GLOBAL de toda la oración. Es la representación
       estándar usada en clasificación.

    2. Mean Pooling [768-dim]:
       Promedio de los estados ocultos de TODOS los tokens. Captura
       información distribuida por toda la frase, incluyendo palabras
       clave individuales que el CLS puede pasar por alto.

    Concatenarlos → 1536-dim: el modelo de clasificación ve AMBAS
    perspectivas y puede aprender cuál es más relevante para cada caso.

    BACKBONE: cardiffnlp/twitter-xlm-roberta-base-sentiment
       Fine-tuned en 198M tweets → embeddings específicos de redes sociales.
    """
    cache_feats = os.path.join(CACHE_DIR,
                               f"text_feats_{split_name}_{_CACHE_VERSION}.npy")
    cache_ids   = os.path.join(CACHE_DIR,
                               f"text_ids_{split_name}_{_CACHE_VERSION}.json")

    if os.path.exists(cache_feats):
        print(f"  [Texto] Caché encontrada: {cache_feats}")
        with open(cache_ids, encoding='utf-8') as f:
            return np.load(cache_feats), json.load(f)

    print(f"  [Texto] Descargando y ejecutando {TEXT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    xlmr      = AutoModel.from_pretrained(TEXT_MODEL)
    xlmr.eval().to(DEVICE)

    texts    = list(df['text'])
    id_list  = list(df['id_EXIST'])
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            batch = [t if t.strip() else "[vacío]" for t in batch]

            enc = tokenizer(
                batch,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = xlmr(**enc)

            hidden = out.last_hidden_state          # [B, seq_len, 768]

            # ── CLS token: posición 0 ────────────────────────────────────
            cls_vec  = hidden[:, 0, :]              # [B, 768]

            # ── Mean pooling: promedio de tokens NO-padding ──────────────
            # La máscara de atención (1=token real, 0=padding) se usa
            # para no incluir los tokens de relleno en el promedio.
            attn_mask = enc['attention_mask'].unsqueeze(-1).float()  # [B, seq, 1]
            sum_hidden = (hidden * attn_mask).sum(dim=1)             # [B, 768]
            n_tokens   = attn_mask.sum(dim=1).clamp(min=1e-9)       # [B, 1]
            mean_vec   = sum_hidden / n_tokens                       # [B, 768]

            # ── Concatenar: CLS + Mean → 1536-dim ───────────────────────
            combined = torch.cat([cls_vec, mean_vec], dim=1)         # [B, 1536]
            all_feats.append(combined.cpu().numpy())

            if i % 320 == 0:
                print(f"    Procesados {min(i+32, len(texts))}/{len(texts)} textos")

    feats = np.vstack(all_feats)   # [N, 1536]
    np.save(cache_feats, feats)
    with open(cache_ids, 'w') as f:
        json.dump(id_list, f)

    del xlmr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  [Texto] Características guardadas: {feats.shape}")
    return feats, id_list


# =============================================================================
# SECCIÓN 5: EXTRACCIÓN DE CARACTERÍSTICAS DE IMAGEN (ResNet50)
# =============================================================================

def extract_image_features(df, split_name):
    """
    Extrae características visuales de cada imagen de meme usando ResNet50.

    ¿Por qué ResNet50?
    ──────────────────
    ResNet50 es una red convolucional (CNN) preentrenada en ImageNet
    (1.2 millones de imágenes, 1000 categorías). Aunque fue entrenada para
    clasificar objetos comunes, sus capas internas aprenden a detectar
    texturas, formas, colores y estructuras visuales. Nosotros usamos
    esas representaciones intermedias (2048-dim) para describir visualmente
    cada meme, SIN las últimas capas de clasificación.

    Esto se llama "transfer learning": aprovechamos el conocimiento visual
    aprendido en ImageNet para nuestra tarea de detección de sexismo.
    """
    cache_path = os.path.join(CACHE_DIR, f"img_feats_{split_name}.npy")

    if os.path.exists(cache_path):
        print(f"  [Imagen] Caché encontrada: {cache_path}")
        return np.load(cache_path)

    print("  [Imagen] Extrayendo características con ResNet50...")

    # ResNet50 sin la capa de clasificación final (retiramos el último Linear)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Salida: [B, 2048, 1, 1]
    resnet.eval().to(DEVICE)

    # Transformaciones estándar requeridas por ResNet
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_paths = list(df['img_path'])
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(img_paths), 32):
            batch_paths = img_paths[i:i + 32]
            tensors = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    tensors.append(tf(img))
                except Exception:
                    # Imagen corrupta o no encontrada → vector de ceros
                    tensors.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))

            batch = torch.stack(tensors).to(DEVICE)
            feats = resnet(batch).squeeze(-1).squeeze(-1)  # [B, 2048]
            all_feats.append(feats.cpu().numpy())

            if i % 320 == 0:
                print(f"    Procesadas {min(i+32, len(img_paths))}/{len(img_paths)} imágenes")

    feats = np.vstack(all_feats)  # [N, 2048]
    np.save(cache_path, feats)

    del resnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  [Imagen] Características guardadas: {feats.shape}")
    return feats


# =============================================================================
# SECCIÓN 6: DATASET DE PYTORCH
# =============================================================================

class MemeDataset(Dataset):
    """
    Dataset de PyTorch que encapsula texto + imagen + etiqueta.

    Soporta dos modos de etiqueta:
      - Hard labels (LongTensor escalar): entrenamiento estándar con CrossEntropy
      - Soft labels (FloatTensor [2]):    distribución de votos, usa soft-CE
        que penaliza menos los errores en muestras ambiguas (p.ej. empates 3-3)
    """

    def __init__(self, text_feats, img_feats, labels=None, soft_labels=None):
        self.text        = torch.FloatTensor(text_feats)
        self.imgs        = torch.FloatTensor(img_feats)
        self.labels      = torch.LongTensor(labels)       if labels      is not None else None
        self.soft_labels = torch.FloatTensor(soft_labels) if soft_labels is not None else None

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if self.soft_labels is not None:
            return self.text[idx], self.imgs[idx], self.soft_labels[idx]
        if self.labels is not None:
            return self.text[idx], self.imgs[idx], self.labels[idx]
        return self.text[idx], self.imgs[idx]


# =============================================================================
# SECCIÓN 6b: FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — mejora para clases desbalanceadas.

    ¿Por qué Focal Loss en lugar de CrossEntropy ponderado?
    ────────────────────────────────────────────────────────
    CrossEntropy trata por igual todos los ejemplos mal clasificados.
    Focal Loss añade un factor (1 - p_t)^γ que:
      - Reduce el peso de ejemplos FÁCILES (bien clasificados, p_t alto)
      - Aumenta el peso de ejemplos DIFÍCILES (mal clasificados, p_t bajo)

    Para T2.2, los ejemplos JUDGEMENTAL son difíciles (pocos y ambiguos).
    Focal Loss hace que el modelo se concentre en aprenderlos en lugar
    de ignorarlos a favor de los DIRECT fáciles.

    Parámetros:
      gamma=0: equivale a CrossEntropy estándar
      gamma=2: recomendado para detección de objetos y texto desbalanceado
      alpha:   pesos adicionales por clase (igual que CrossEntropy weighted)
    """

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor [num_classes] o None

    def forward(self, logits, targets):
        # Probabilidades con softmax
        probs    = torch.softmax(logits, dim=1)                   # [B, C]
        # Probabilidad de la clase correcta para cada muestra
        p_t      = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        # CrossEntropy estándar por muestra
        ce_loss  = torch.nn.functional.cross_entropy(
            logits, targets,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            reduction='none'
        )                                                          # [B]
        # Factor focal: penaliza menos los ejemplos fáciles
        focal_w  = (1.0 - p_t) ** self.gamma                     # [B]
        loss     = (focal_w * ce_loss).mean()
        return loss


# =============================================================================
# SECCIÓN 7: ARQUITECTURA DEL MODELO MULTIMODAL CON BiGRU
# =============================================================================

class MultimodalBiGRUClassifier(nn.Module):
    """
    Clasificador multimodal de sexismo con Red Neuronal Recurrente Bidireccional (BiGRU).

    ARQUITECTURA COMPLETA:
    ══════════════════════

    ┌─────────────────────────────────────────────────────────────────┐
    │  RAMA DE TEXTO (XLM-RoBERTa features → BiGRU)                  │
    │                                                                  │
    │  Vector de texto [1536-dim]                                      │
    │         ↓ Linear(1536 → 512) + LayerNorm + ReLU                 │
    │  [512-dim]                                                       │
    │         ↓ reshape: [B, 8, 64]  (8 pasos × 64 features)         │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────┐       │
    │  │         RED NEURONAL RECURRENTE (BiGRU)              │       │
    │  │                                                       │       │
    │  │  paso1 → paso2 → paso3 → paso4 → paso5 → paso6→p7→p8│       │
    │  │    ↑←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←↑        │       │
    │  │  (dirección → + dirección ← = bidireccional)         │       │
    │  └──────────────────────────────────────────────────────┘       │
    │         ↓ estado oculto final [hidden_size × 2 = 256-dim]       │
    │  [256-dim representación de texto]                               │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  RAMA DE IMAGEN (ResNet50 features → MLP)                       │
    │                                                                  │
    │  Vector de imagen [2048-dim]                                     │
    │         ↓ Linear(2048 → 512) + ReLU + Linear(512 → 256)        │
    │  [256-dim representación de imagen]                              │
    └─────────────────────────────────────────────────────────────────┘

                    ↓ Concatenar (256 + 256 = 512-dim)

    ┌─────────────────────────────────────────────────────────────────┐
    │  CABEZA DE CLASIFICACIÓN                                         │
    │                                                                  │
    │  [512] → Linear(512→256) → ReLU → Dropout                      │
    │       → Linear(256→64)  → ReLU → Dropout                       │
    │       → Linear(64→num_classes)                                  │
    │                                                                  │
    │  Tarea 2.1: num_classes = 2  (YES / NO)                        │
    │  Tarea 2.2: num_classes = 2  (DIRECT / JUDGEMENTAL)            │
    └─────────────────────────────────────────────────────────────────┘

    ¿Por qué usar un GRU (Red Recurrente)?
    ───────────────────────────────────────
    Las redes recurrentes (RNN, LSTM, GRU) están diseñadas para procesar
    SECUENCIAS. El vector de texto de XLM-RoBERTa (768-dim) encapsula el
    significado global de la frase, pero no preserva la estructura temporal.

    Al dividir ese vector en 8 segmentos y procesarlos con un GRU, la red
    aprende qué PARTES de la representación semántica son más informativas
    para detectar sexismo. El GRU bidireccional lee la secuencia tanto de
    izquierda a derecha como de derecha a izquierda, capturando dependencias
    en ambas direcciones (útil cuando el contexto final modifica el inicio).

    GRU vs LSTM: El GRU (Gated Recurrent Unit) es más eficiente que el LSTM
    al tener menos parámetros, siendo preferible cuando el dataset es moderado.
    """

    def __init__(self, num_classes=2,
                 hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_GRU_LAYERS,
                 dropout=DROPOUT):
        super().__init__()

        # ── Proyección de texto ──────────────────────────────────────────
        # 1536-dim → 512-dim (GRU_SEQ_LEN=8 × 64-dim por paso)
        proj_text_dim = GRU_SEQ_LEN * 64  # = 512
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_DIM, proj_text_dim),
            nn.LayerNorm(proj_text_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gru_input_dim = 64        # Dimensión de cada paso del GRU
        self.gru_seq_len   = GRU_SEQ_LEN

        # ── Red Neuronal Recurrente Bidireccional (BiGRU) ────────────────
        # input_size  = 64 (features por paso de tiempo)
        # hidden_size = 128 (neuronas en cada dirección)
        # × 2 (bidireccional) = 256-dim al final
        self.bigru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Proyección de la salida del GRU: hidden*2 → PROJ_DIM
        self.text_out = nn.Sequential(
            nn.Linear(hidden_size * 2, PROJ_DIM),
            nn.ReLU(),
        )
        # ── Atención sobre salidas del GRU ───────────────────────────────
        # En lugar de tomar solo el último estado oculto, aprendemos a
        # PONDERAR todos los pasos de tiempo del GRU. La red aprende qué
        # segmentos del vector semántico son más relevantes para la tarea.
        self.gru_attention = nn.Linear(hidden_size * 2, 1)

        # ── Procesamiento de imagen ──────────────────────────────────────
        self.img_proj = nn.Sequential(
            nn.Linear(IMAGE_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, PROJ_DIM),
            nn.ReLU(),
        )

        # ── Cabeza de clasificación ──────────────────────────────────────
        # texto(256) + imagen(256) + interacción texto⊙imagen(256) = 768
        # El producto elemento a elemento captura correlaciones directas
        # entre texto e imagen que la simple concatenación no puede ver.
        fusion_dim = PROJ_DIM * 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

        # Inicialización de pesos (Xavier para capas lineales)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_feat, img_feat):
        """
        Paso hacia adelante (forward pass).

        Args:
            text_feat : [B, 1536] — vector CLS+MeanPool (Twitter XLM-R)
            img_feat  : [B, 2048] — vector de imagen (ResNet50)

        Returns:
            logits    : [B, num_classes] — puntuaciones crudas por clase
        """
        B = text_feat.size(0)

        # ── Rama de texto ────────────────────────────────────────────────
        t = self.text_proj(text_feat)                        # [B, 512]
        t_seq = t.view(B, self.gru_seq_len, self.gru_input_dim)  # [B, 8, 64]

        # Pasar la secuencia por el GRU bidireccional
        gru_out, _ = self.bigru(t_seq)  # [B, seq_len, hidden*2]

        # ── Atención sobre todos los pasos del GRU ───────────────────────
        # gru_attention aprende un peso escalar para cada paso de tiempo
        attn_w    = torch.softmax(self.gru_attention(gru_out), dim=1)  # [B, seq, 1]
        context   = (gru_out * attn_w).sum(dim=1)           # [B, hidden*2]
        text_repr = self.text_out(context)                  # [B, 256]

        # ── Rama de imagen ───────────────────────────────────────────────
        img_repr = self.img_proj(img_feat)                   # [B, 256]

        # ── Fusión multimodal con interacción ───────────────────────────
        # El producto elemento a elemento detecta correlaciones directas
        # texto↔imagen (p.ej. texto agresivo + imagen de mujer → señal fuerte)
        interaction = text_repr * img_repr                   # [B, 256]
        fused = torch.cat([text_repr, img_repr, interaction], dim=1)  # [B, 768]

        # ── Clasificación ────────────────────────────────────────────────
        return self.classifier(fused)                        # [B, num_classes]


# =============================================================================
# SECCIÓN 7b: CLASIFICADOR SOLO-TEXTO (TextOnlyBiGRUClassifier)
# =============================================================================

class TextOnlyBiGRUClassifier(nn.Module):
    """
    Clasificador basado únicamente en texto (sin rama de imagen).

    ¿Para qué sirve como base model del stacking?
    ──────────────────────────────────────────────
    Algunos memes son sexistas puramente por su texto, con imágenes neutras
    (una foto de fondo genérica). Incluir la imagen en esos casos añade ruido.
    Este modelo aprende esos casos mejor que los modelos multimodales.

    Al combinarlo en el meta-learner con modelos multimodales, el StackingHead
    puede aprender a ignorar la imagen cuando el texto ya es suficientemente
    discriminativo.
    """

    def __init__(self, num_classes=2,
                 hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_GRU_LAYERS,
                 dropout=DROPOUT):
        super().__init__()

        proj_text_dim = GRU_SEQ_LEN * 64
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_DIM, proj_text_dim),
            nn.LayerNorm(proj_text_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gru_input_dim = 64
        self.gru_seq_len   = GRU_SEQ_LEN

        self.bigru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_attention = nn.Linear(hidden_size * 2, 1)
        self.text_out = nn.Sequential(
            nn.Linear(hidden_size * 2, PROJ_DIM),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(PROJ_DIM, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_feat, img_feat=None):
        # img_feat se acepta pero se ignora (interfaz compatible con train_model)
        B = text_feat.size(0)
        t      = self.text_proj(text_feat)
        t_seq  = t.view(B, self.gru_seq_len, self.gru_input_dim)
        gru_out, _ = self.bigru(t_seq)
        attn_w     = torch.softmax(self.gru_attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        text_repr  = self.text_out(context)
        return self.classifier(text_repr)


# =============================================================================
# SECCIÓN 7d: CLASIFICADOR CROSSMODAL CON GRU (CrossModalGRUClassifier)
# =============================================================================

class CrossModalGRUClassifier(nn.Module):
    """
    Fusiona texto e imagen DENTRO del GRU (fusión temprana).

    texto[1536]  → seq[B, 8, 64] ──┐
    imagen[2048] → seq[B, 8, 64] ──┴→ [B, 16, 64] → BiGRU → atención → 2 logits

    El GRU aprende interacciones texto↔imagen paso a paso: útil en memes
    donde el sexismo nace del contraste entre texto e imagen.
    """

    def __init__(self, num_classes=2,
                 hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_GRU_LAYERS,
                 dropout=DROPOUT):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_DIM,  GRU_SEQ_LEN * 64),
            nn.LayerNorm(GRU_SEQ_LEN * 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(IMAGE_DIM, GRU_SEQ_LEN * 64),
            nn.LayerNorm(GRU_SEQ_LEN * 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.gru_input_dim = 64
        self.combined_seq  = GRU_SEQ_LEN * 2  # 16 pasos: 8 texto + 8 imagen

        self.bigru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_attention = nn.Linear(hidden_size * 2, 1)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, PROJ_DIM),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(PROJ_DIM, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_feat, img_feat):
        B = text_feat.size(0)
        t_seq = self.text_proj(text_feat).view(B, GRU_SEQ_LEN, self.gru_input_dim)
        i_seq = self.img_proj(img_feat).view(B, GRU_SEQ_LEN, self.gru_input_dim)
        combined   = torch.cat([t_seq, i_seq], dim=1)              # [B, 16, 64]
        gru_out, _ = self.bigru(combined)                          # [B, 16, 256]
        attn_w     = torch.softmax(self.gru_attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        fused      = self.out_proj(context)
        return self.classifier(fused)


# =============================================================================
# SECCIÓN 7e: UTILIDADES DE ENSEMBLE (soft_vote)
# =============================================================================

def collect_base_probs(models_list, loader, device):
    """
    Pasa un DataLoader por varios base models y devuelve sus probabilidades
    softmax concatenadas: [N, n_models * 2].
    """
    all_probs = []
    for m in models_list:
        m.eval()
        probs_m = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    text_f, img_f, _ = batch
                else:
                    text_f, img_f = batch
                logits = m(text_f.to(device), img_f.to(device))
                probs_m.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.vstack(probs_m))
    return np.concatenate(all_probs, axis=1)  # [N, n_models*2]


def soft_vote(models_list, loader, device, weights=None):
    """
    Ensambla varios modelos promediando sus probabilidades softmax.

    weights: lista de floats (p.ej. F1 de validaci\u00f3n de cada modelo).
      - None  \u2192 promedio aritmético simple (todos pesan igual)
      - lista \u2192 promedio ponderado normalizado por suma de pesos
        El modelo con mayor F1 de validaci\u00f3n contribuye m\u00e1s al ensemble.
    """
    stacked  = collect_base_probs(models_list, loader, device)  # [N, n_models*2]
    n_models = len(models_list)
    probs    = stacked.reshape(len(stacked), n_models, 2)       # [N, n_models, 2]

    if weights is not None:
        w = np.array(weights, dtype=np.float64)
        w = w / w.sum()                                          # normalizar a suma=1
        return (probs * w[np.newaxis, :, np.newaxis]).sum(axis=1)  # [N, 2]

    return probs.mean(axis=1)                                    # [N, 2]


# =============================================================================
# SECCIÓN 8: ENTRENAMIENTO
# =============================================================================

def _soft_cross_entropy(logits, soft_targets):
    """
    Cross-entropy con etiquetas suaves: -sum(p * log(q)).
    Generaliza la CE estándar: si soft_targets es one-hot, es equivalente.
    Para distribuciones suaves (p.ej. {YES:0.5, NO:0.5}), penaliza menos
    los errores en muestras que los propios anotadores no acordaron.
    """
    log_probs = torch.log_softmax(logits, dim=1)             # [B, C]
    return -(soft_targets * log_probs).sum(dim=1).mean()     # escalar


def _run_epoch(model, loader, optimizer, criterion, device, training=True, max_grad_norm=1.0):
    """Ejecuta una época completa de entrenamiento o evaluación."""
    model.train() if training else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            text_f, img_f, labels = batch
            text_f = text_f.to(device)
            img_f  = img_f.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(text_f, img_f)

            # Detectar si son soft labels [B, 2] o hard labels [B] (escalar)
            if labels.dim() == 2:
                # Soft labels: usar soft cross-entropy (ignora 'criterion')
                loss       = _soft_cross_entropy(logits, labels)
                hard_labels = labels.argmax(dim=1)   # para métricas
            elif criterion is not None:
                loss        = criterion(logits, labels)
                hard_labels = labels
            else:
                # Evaluación sin criterio (solo interesan las métricas)
                loss        = torch.zeros((), device=logits.device)
                hard_labels = labels

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(hard_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, epochs, lr, task_name,
                class_weights=None, patience=6, use_focal_loss=False, max_grad_norm=1.0):
    """
    Entrena el modelo con:
      - Optimizador AdamW (Adam con regularización de pesos)
      - Scheduler Cosine Annealing (reduce LR gradualmente)
      - Early stopping (detiene si no mejora en `patience` épocas)
      - Guarda el mejor modelo según F1 en validación
      - class_weights: [w_clase0, w_clase1] para penalizar la clase minoritaria
      - use_focal_loss: usa FocalLoss(gamma=2) en lugar de CrossEntropy
        (recomendado para clases muy desbalanceadas como T2.2)
    """
    if use_focal_loss:
        alpha = torch.FloatTensor(class_weights).to(DEVICE) if class_weights else None
        criterion = FocalLoss(gamma=4.0, alpha=alpha)
    else:
        w = torch.FloatTensor(class_weights).to(DEVICE) if class_weights else None
        criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1    = 0.0
    best_state = None
    no_improve = 0

    print(f"\n{'─' * 62}")
    print(f"  Entrenando: {task_name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros entrenables: {n_params:,}")
    print(f"{'─' * 62}")
    print(f"  {'Época':>5}  {'Train Loss':>10}  {'Train F1':>8}  {'Val Loss':>8}  {'Val F1':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = _run_epoch(model, train_loader, optimizer,
                                           criterion, DEVICE, training=True,
                                           max_grad_norm=max_grad_norm)
        vl_loss, vl_f1, _, _ = _run_epoch(model, val_loader, optimizer,
                                           criterion, DEVICE, training=False,
                                           max_grad_norm=max_grad_norm)
        scheduler.step()

        marker = " ←" if vl_f1 > best_f1 else ""
        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_f1:>8.4f}  "
              f"{vl_loss:>8.4f}  {vl_f1:>7.4f}{marker}")

        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  [Early stopping] Sin mejora en {patience} épocas.")
                break

    model.load_state_dict(best_state)
    print(f"\n  Mejor F1 en validación: {best_f1:.4f}")
    return model


# =============================================================================
# SECCIÓN 9: GENERACIÓN DE PREDICCIONES EN FORMATO EXIST 2025
# =============================================================================

def predict(model, loader):
    """Genera predicciones y probabilidades con softmax."""
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                text_f, img_f, _ = batch
            else:
                text_f, img_f = batch

            logits = model(text_f.to(DEVICE), img_f.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_probs)


def save_task21(ids, preds, probs, filename):
    """
    Guarda predicciones para T2.1 en formato EXIST 2025.

    Formato hard: {"test_case": "EXIST2025", "id": "...", "value": "YES"}
    Formato soft: {"test_case": "EXIST2025", "id": "...",
                   "value": {"YES": 0.82, "NO": 0.18}}
    """
    label_map = {1: 'YES', 0: 'NO'}
    hard_out, soft_out = [], []

    for i, id_e in enumerate(ids):
        hard_out.append({"test_case": "EXIST2025", "id": str(id_e),
                          "value": label_map[preds[i]]})
        soft_out.append({"test_case": "EXIST2025", "id": str(id_e),
                          "value": {"YES": round(float(probs[i][1]), 6),
                                    "NO":  round(float(probs[i][0]), 6)}})

    for suffix, data in [("_hard.json", hard_out), ("_soft.json", soft_out)]:
        path = os.path.join(OUTPUT_DIR, filename + suffix)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Guardado: {path}")

    return hard_out, soft_out


def save_task22(all_ids, preds_21, probs_21, probs_22_all, filename):
    """
    Guarda predicciones para T2.2 en formato EXIST 2025 (PyEvALL).

    El gold estándar de T2.2 incluye TODOS los memes con 3 clases:
      - NO:          meme no sexista
      - DIRECT:      meme sexista, intención directa
      - JUDGEMENTAL: meme sexista, intención crítica/condenatoria

    Para cumplir el formato, combinamos T2.1 y T2.2 jerárquicamente:
      P(NO)          = T2.1 P(NO)
      P(DIRECT)      = T2.1 P(YES) * T2.2 P(DIRECT)
      P(JUDGEMENTAL) = T2.1 P(YES) * T2.2 P(JUDGEMENTAL)
      -> suma = P(NO) + P(YES) * 1.0 = 1.0

    Args:
      all_ids   : lista con TODOS los IDs de test (len = N)
      preds_21  : array [N] con predicciones T2.1 (0=NO, 1=YES)
      probs_21  : array [N,2] con probabilidades T2.1 ([:, 0]=P(NO), [:, 1]=P(YES))
      probs_22_all : array [N,2] con probabilidades T2.2 para TODOS los memes
                     ([:, 0]=P(JUDGEMENTAL), [:, 1]=P(DIRECT))
    """
    hard_out, soft_out = [], []
    LABELS = ['NO', 'JUDGEMENTAL', 'DIRECT']

    for i, id_e in enumerate(all_ids):
        p_no   = float(probs_21[i][0])   # P(NO) de T2.1
        p_yes  = float(probs_21[i][1])   # P(YES) de T2.1
        p_judg = float(probs_22_all[i][0]) * p_yes   # P(JUDGEMENTAL) combinado
        p_dir  = float(probs_22_all[i][1]) * p_yes   # P(DIRECT) combinado

        # Hard: respetar jerarquía T2.1.
        # Si T2.1 predijo YES (preds_21[i]==1), el hard label de T2.2 NUNCA
        # puede ser NO — se elige entre DIRECT y JUDGEMENTAL.
        # Si T2.1 predijo NO, el hard label es siempre NO.
        if int(preds_21[i]) == 1:
            # Meme sexista: elegir entre DIRECT y JUDGEMENTAL
            hard_label = 'DIRECT' if p_dir >= p_judg else 'JUDGEMENTAL'
        else:
            hard_label = 'NO'

        hard_out.append({"test_case": "EXIST2025", "id": str(id_e),
                          "value": hard_label})
        soft_out.append({"test_case": "EXIST2025", "id": str(id_e),
                          "value": {"NO":          round(p_no,   6),
                                    "DIRECT":      round(p_dir,  6),
                                    "JUDGEMENTAL": round(p_judg, 6)}})

    for suffix, data in [("_hard.json", hard_out), ("_soft.json", soft_out)]:
        path = os.path.join(OUTPUT_DIR, filename + suffix)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Guardado: {path}")

    preds_22_combined = np.array([LABELS.index(e['value']) for e in hard_out])
    return hard_out, soft_out, preds_22_combined


# =============================================================================
# SECCIÓN 10: PIPELINE PRINCIPAL
# =============================================================================

if __name__ == '__main__':

    # ──────────────────────────────────────────────────────────────────────
    # PASO 1: Cargar datos
    # ──────────────────────────────────────────────────────────────────────
    df_train = load_training_data()
    print("\n[PASO 1b] Cargando datos de test...")
    df_test  = load_test_data()

    # ──────────────────────────────────────────────────────────────────────
    # PASO 1c: OCR propio — extraer texto directamente de las imágenes
    # ──────────────────────────────────────────────────────────────────────
    print("\n[PASO 1c] Extrayendo texto de imágenes con OCR propio (EasyOCR)...")
    print("  (Primera vez: ~30-90 min en CPU según hardware. Siguiente: <1 min)")
    ocr_tr = extract_ocr_text(df_train, 'train')
    ocr_te = extract_ocr_text(df_test,  'test')

    # Enriquecer df['text'] fusionando texto JSON + texto OCR
    df_train['text'] = df_train.apply(
        lambda r: merge_text_sources(r['text'], ocr_tr.get(r['id_EXIST'], '')),
        axis=1
    )
    df_test['text'] = df_test.apply(
        lambda r: merge_text_sources(r['text'], ocr_te.get(r['id_EXIST'], '')),
        axis=1
    )
    n_enr_tr = sum(1 for t in df_train['text'] if ' | ' in str(t))
    n_enr_te = sum(1 for t in df_test['text']  if ' | ' in str(t))
    print(f"  Textos enriquecidos (JSON + OCR) → train: {n_enr_tr} | test: {n_enr_te}")

    # ──────────────────────────────────────────────────────────────────────
    # PASO 2: Extraer características multimodales (con caché automático)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[PASO 2] Extrayendo características multimodales...")
    print("  (Primera vez: descarga modelos y tarda ~15-30 min. Siguiente: <1 min)")

    text_tr_raw, text_ids_tr = extract_text_features(df_train, 'train')
    text_te_raw, text_ids_te = extract_text_features(df_test,  'test')
    img_tr_raw = extract_image_features(df_train, 'train')
    img_te_raw = extract_image_features(df_test,  'test')

    # Reindexar features por ID para garantizar alineación con el DataFrame,
    # independientemente del orden en que se construyó el caché.
    def _align_feats(df, text_raw, text_ids, img_raw):
        id_to_text = {id_: text_raw[i] for i, id_ in enumerate(text_ids)}
        id_to_img  = {id_: img_raw[i]  for i, id_ in enumerate(df['id_EXIST'])}
        ids = list(df['id_EXIST'])
        text_aligned = np.array([id_to_text[i] for i in ids])
        img_aligned  = np.array([id_to_img[i]  for i in ids])
        return text_aligned, img_aligned

    text_tr, img_tr = _align_feats(df_train, text_tr_raw, text_ids_tr, img_tr_raw)
    text_te, img_te = _align_feats(df_test,  text_te_raw, text_ids_te, img_te_raw)
    print(f"  Features alineadas — train: {text_tr.shape} | test: {text_te.shape}")

    # ──────────────────────────────────────────────────────────────────────
    # PASO 3: Preparar features y soft labels para T2.1
    #   Evaluación honesta: ANTES del k-fold se aparta un HOLDOUT
    #   estratificado del 15% que ningún modelo ve durante el entrenamiento.
    #   Toda la métrica final de T2.1 se calcula sobre ese holdout.
    #   El split se persiste en disco para que LOAD_PRETRAINED=True evalúe
    #   siempre sobre las mismas muestras.
    # ──────────────────────────────────────────────────────────────────────
    print("\n[PASO 3] Preparando dataset para Tarea 2.1 (YES/NO) — holdout + K-fold + soft labels...")

    labels_21      = df_train['label_21'].values
    soft_labels_21 = np.array(list(df_train['soft_21']), dtype=np.float32)  # [N, 2]

    holdout_path = os.path.join(OUTPUT_DIR, "holdout_split_t21.json")
    if LOAD_PRETRAINED and os.path.exists(holdout_path):
        with open(holdout_path, encoding='utf-8') as f:
            _split = json.load(f)
        idx_dev, idx_hold = np.array(_split['dev']), np.array(_split['holdout'])
        print(f"  Split cargado de disco — dev: {len(idx_dev)} | holdout: {len(idx_hold)}")
    else:
        idx_dev, idx_hold = train_test_split(
            np.arange(len(labels_21)), test_size=0.15,
            stratify=labels_21, random_state=SEED
        )
        with open(holdout_path, 'w', encoding='utf-8') as f:
            json.dump({'dev': idx_dev.tolist(), 'holdout': idx_hold.tolist()}, f)
        print(f"  Split creado (estratificado, seed={SEED}) — dev: {len(idx_dev)} | holdout: {len(idx_hold)}")

    ds_te21 = MemeDataset(text_te, img_te)
    ld_te21 = DataLoader(ds_te21, batch_size=BATCH_SIZE, shuffle=False)

    # ──────────────────────────────────────────────────────────────────────
    # PASO 4: Base models Tarea 2.1 (K-fold o carga desde disco)
    # ──────────────────────────────────────────────────────────────────────
    models_21      = []
    model_weights_21 = []   # F1 de validación por modelo (para weighted voting)

    if LOAD_PRETRAINED:
        print("\n[PASO 4] Cargando base models T2.1 desde disco (LOAD_PRETRAINED=True)...")
        # Cargar pesos F1 guardados durante el entrenamiento
        weights_path = os.path.join(OUTPUT_DIR, "model_weights_t21.json")
        if os.path.exists(weights_path):
            with open(weights_path, encoding='utf-8') as f:
                model_weights_21 = json.load(f)
            print(f"  Pesos F1 cargados: {[f'{w:.4f}' for w in model_weights_21]}")
        else:
            print("  (Sin pesos guardados — usando promedio equitativo)")

        for fold_i in range(N_FOLDS_T21):
            m_i = MultimodalBiGRUClassifier(num_classes=2).to(DEVICE)
            path_i = os.path.join(OUTPUT_DIR, f"model_task21_fold{fold_i}.pt")
            m_i.load_state_dict(torch.load(path_i, map_location=DEVICE))
            m_i.eval()
            models_21.append(m_i)
            print(f"  Cargado: {os.path.basename(path_i)}")

        model_cross_21 = CrossModalGRUClassifier(num_classes=2).to(DEVICE)
        path_cross = os.path.join(OUTPUT_DIR, "model_task21_crossmodal.pt")
        model_cross_21.load_state_dict(torch.load(path_cross, map_location=DEVICE))
        model_cross_21.eval()
        models_21.append(model_cross_21)
        print(f"  Cargado: {os.path.basename(path_cross)}")

        if not model_weights_21:
            model_weights_21 = [1.0] * len(models_21)

    else:
        print(f"\n[PASO 4] Entrenando base models T2.1 con {N_FOLDS_T21}-fold cross-validation...")
        # Los soft_labels_21 se usan en training; para la métrica de validación
        # seguimos evaluando con hard labels (labels_21) para comparabilidad.

        skf = StratifiedKFold(n_splits=N_FOLDS_T21, shuffle=True, random_state=SEED)
        fold_val_loaders = []   # guardamos loaders de val para evaluar CrossModalGRU
        fold_splits      = []   # (tr_idx, vl_idx) absolutos de cada fold

        # K-fold SOLO sobre el conjunto de desarrollo (el holdout queda fuera)
        for fold_i, (tr_rel, vl_rel) in enumerate(skf.split(text_tr[idx_dev], labels_21[idx_dev])):
            tr_idx, vl_idx = idx_dev[tr_rel], idx_dev[vl_rel]
            fold_splits.append((tr_idx, vl_idx))
            print(f"\n  [Fold {fold_i+1}/{N_FOLDS_T21}] train:{len(tr_idx)} | val:{len(vl_idx)}")

            # Dataset de entrenamiento con SOFT LABELS
            ds_tr_fold = MemeDataset(
                text_tr[tr_idx], img_tr[tr_idx],
                soft_labels=soft_labels_21[tr_idx]
            )
            # Dataset de validación con HARD LABELS (para métricas comparables)
            ds_vl_fold = MemeDataset(
                text_tr[vl_idx], img_tr[vl_idx],
                labels=labels_21[vl_idx]
            )
            ld_tr_fold = DataLoader(ds_tr_fold, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
            ld_vl_fold = DataLoader(ds_vl_fold, batch_size=BATCH_SIZE, shuffle=False)
            fold_val_loaders.append(ld_vl_fold)

            # Pesos de clase sobre el fold de entrenamiento
            labs_tr = labels_21[tr_idx]
            n_tr = len(labs_tr)
            cw_fold = [
                n_tr / (2 * max((labs_tr == 0).sum(), 1)),
                n_tr / (2 * max((labs_tr == 1).sum(), 1)),
            ]

            torch.manual_seed(SEED + fold_i)
            np.random.seed(SEED + fold_i)
            m_i = MultimodalBiGRUClassifier(num_classes=2).to(DEVICE)
            m_i = train_model(
                m_i, ld_tr_fold, ld_vl_fold,
                epochs=EPOCHS_T21, lr=LR,
                task_name=f"MultimodalBiGRU [fold={fold_i+1}]",
                class_weights=cw_fold,
                patience=10
            )
            models_21.append(m_i)

            # F1 de validación de este fold → peso en el ensemble
            _, fold_f1, fold_preds, fold_labs = _run_epoch(
                m_i, ld_vl_fold, None, None, DEVICE, training=False
            )
            model_weights_21.append(fold_f1)
            print(f"  Fold {fold_i+1} val F1: {fold_f1:.4f} (peso en ensemble)")

        # CrossModalGRU: entrena con el TRAIN del fold 0 y valida en el val
        # del fold 0 (que nunca ve) — así su early-stopping y su peso en el
        # ensemble son comparables y honestos, igual que los del resto de folds
        tr0_idx, _ = fold_splits[0]
        labs_all = labels_21[tr0_idx]
        n_all = len(labs_all)
        cw_all = [
            n_all / (2 * max((labs_all == 0).sum(), 1)),
            n_all / (2 * max((labs_all == 1).sum(), 1)),
        ]
        ds_tr_all = MemeDataset(text_tr[tr0_idx], img_tr[tr0_idx], soft_labels=soft_labels_21[tr0_idx])
        ld_tr_all = DataLoader(ds_tr_all, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        print(f"\n  [CrossModalGRU] Entrenando con el split del fold 0 (train/val honestos)...")
        torch.manual_seed(SEED + N_FOLDS_T21)
        model_cross_21 = CrossModalGRUClassifier(num_classes=2).to(DEVICE)
        model_cross_21 = train_model(
            model_cross_21, ld_tr_all, fold_val_loaders[0],
            epochs=EPOCHS_T21, lr=LR,
            task_name="CrossModalGRU T2.1",
            class_weights=cw_all,
            patience=10
        )
        models_21.append(model_cross_21)

        _, cross_f1, _, _ = _run_epoch(
            model_cross_21, fold_val_loaders[0], None, None, DEVICE, training=False
        )
        model_weights_21.append(cross_f1)
        print(f"  CrossModalGRU fold-0 val F1: {cross_f1:.4f} (peso en ensemble)")

        # Guardar pesos F1 para reutilizar con LOAD_PRETRAINED=True
        with open(os.path.join(OUTPUT_DIR, "model_weights_t21.json"), 'w') as f:
            json.dump(model_weights_21, f)

    print(f"\n  Pesos ensemble T2.1 (F1 val por modelo): {[f'{w:.4f}' for w in model_weights_21]}")

    # ──────────────────────────────────────────────────────────────────────
    # PASO 4b: Ensemble T2.1 — Soft Voting PONDERADO por F1, evaluado en el
    # HOLDOUT del 15% que ningún modelo ha visto durante el entrenamiento
    # ──────────────────────────────────────────────────────────────────────
    ds_vl21_ref = MemeDataset(text_tr[idx_hold], img_tr[idx_hold], labels=labels_21[idx_hold])
    ld_vl21 = DataLoader(ds_vl21_ref, batch_size=BATCH_SIZE, shuffle=False)

    print("\n[PASO 4b] Ensemble T2.1 — Soft Voting ponderado por F1 (evaluación en holdout)...")
    probs_vl21  = soft_vote(models_21, ld_vl21, DEVICE, weights=model_weights_21)
    vl_preds_21 = probs_vl21.argmax(axis=1)
    vl_labs_21  = labels_21[idx_hold]
    print(f"  Holdout: {len(vl_labs_21)} muestras nunca vistas en entrenamiento")
    print("\n  [Evaluación HOLDOUT — T2.1 Soft Voting ponderado]")
    print(classification_report(vl_labs_21, vl_preds_21,
                                 target_names=['NO', 'YES'], zero_division=0))

    # ──────────────────────────────────────────────────────────────────────
    # PASO 5: Preparar y entrenar modelos Tarea 2.2 (DIRECT/JUDGEMENTAL)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[PASO 5] Preparando dataset para Tarea 2.2 (DIRECT/JUDGEMENTAL)...")

    mask22    = df_train['label_22'].isin([0, 1]).values
    t_feats22 = text_tr[mask22]
    i_feats22 = img_tr[mask22]
    labs22    = df_train['label_22'].values[mask22]

    # Split honesto 70/15/15: train / val (early-stopping + calibración del
    # umbral) / holdout (métrica final, nunca visto). Persistido en disco.
    n22 = len(labs22)
    holdout22_path = os.path.join(OUTPUT_DIR, "holdout_split_t22.json")
    if LOAD_PRETRAINED and os.path.exists(holdout22_path):
        with open(holdout22_path, encoding='utf-8') as f:
            _s22 = json.load(f)
        tr22 = np.array(_s22['train']); vl22 = np.array(_s22['val']); ho22 = np.array(_s22['holdout'])
        print("  Split T2.2 cargado de disco")
    else:
        tr22, _rest22 = train_test_split(
            np.arange(n22), test_size=0.30, stratify=labs22, random_state=SEED)
        vl22, ho22 = train_test_split(
            _rest22, test_size=0.50, stratify=labs22[_rest22], random_state=SEED)
        with open(holdout22_path, 'w', encoding='utf-8') as f:
            json.dump({'train': tr22.tolist(), 'val': vl22.tolist(),
                       'holdout': ho22.tolist()}, f)

    print(f"  Split T2.2 — train: {len(tr22)} | val: {len(vl22)} | holdout: {len(ho22)}")

    # Oversampling JUDGEMENTAL en el split de entrenamiento
    labs22_tr   = labs22[tr22]
    judg_idx_tr = np.where(labs22_tr == 0)[0]
    dir_idx_tr  = np.where(labs22_tr == 1)[0]
    oversample_ratio = max(1, (len(dir_idx_tr) * 4) // max(len(judg_idx_tr) * 3, 1))
    judg_oversampled  = np.tile(judg_idx_tr, oversample_ratio)
    balanced_tr_local = np.concatenate([dir_idx_tr, judg_oversampled])
    np.random.shuffle(balanced_tr_local)
    balanced_tr22 = tr22[balanced_tr_local]

    ds_tr22 = MemeDataset(t_feats22[balanced_tr22], i_feats22[balanced_tr22], labs22[balanced_tr22])
    ds_vl22 = MemeDataset(t_feats22[vl22],          i_feats22[vl22],          labs22[vl22])

    ld_tr22 = DataLoader(ds_tr22, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    ld_vl22 = DataLoader(ds_vl22, batch_size=BATCH_SIZE, shuffle=False)

    labs22_bal = labs22[balanced_tr22]
    print(f"  Muestras ORIGINALES  — DIRECT: {(labs22_tr==1).sum()} | JUDGEMENTAL: {(labs22_tr==0).sum()}")
    print(f"  Muestras BALANCEADAS — DIRECT: {(labs22_bal==1).sum()} | JUDGEMENTAL: {(labs22_bal==0).sum()}")

    n_bal = len(labs22_bal)
    cw_22 = [
        n_bal / (2 * max((labs22_bal == 0).sum(), 1)),
        n_bal / (2 * max((labs22_bal == 1).sum(), 1)),
    ]
    print(f"  Pesos de clase T2.2 → JUDGEMENTAL: {cw_22[0]:.3f} | DIRECT: {cw_22[1]:.3f}")

    # T2.2: un solo TextOnlyBiGRU seed=42 (FocalLoss gamma=4)
    # Los ensembles de múltiples seeds empeoran el resultado: las curvas de
    # validación son ruidosas y promediar distribuciones incoherentes diluye
    # la señal del mejor modelo (seed=42 → F1=0.6371 vs ensemble → 0.6073).
    seeds_22 = [42]
    models_22 = []
    if LOAD_PRETRAINED:
        print("\n[PASO 5] Cargando modelo T2.2 desde disco (LOAD_PRETRAINED=True)...")
        m_22 = TextOnlyBiGRUClassifier(num_classes=2,
                                       dropout=0.5,
                                       num_layers=1).to(DEVICE)
        path_22 = os.path.join(OUTPUT_DIR, f"model_task22_s{seeds_22[0]}.pt")
        m_22.load_state_dict(torch.load(path_22, map_location=DEVICE))
        m_22.eval()
        models_22.append(m_22)
        print(f"  Cargado: {os.path.basename(path_22)}")
    else:
        print(f"\n  [5a] TextOnlyBiGRU T2.2 [seed=42]")
        torch.manual_seed(42)
        m_22 = TextOnlyBiGRUClassifier(num_classes=2,
                                       dropout=0.5,
                                       num_layers=1).to(DEVICE)
        m_22 = train_model(
            m_22, ld_tr22, ld_vl22,
            epochs=EPOCHS_T22, lr=LR * 0.5,
            task_name="TextOnlyBiGRU T2.2 [seed=42]",
            class_weights=cw_22,
            patience=15,
            use_focal_loss=True,
            max_grad_norm=1.0
        )
        models_22.append(m_22)

    # Ensemble T2.2 — Soft Voting + calibración de umbral en validación
    print("\n[PASO 5b] Ensemble T2.2 — Soft Voting (promedio de probabilidades)...")
    probs_vl22  = soft_vote(models_22, ld_vl22, DEVICE)   # [N, 2]
    vl_labs_22  = labs22[vl22]

    # Buscar el umbral óptimo para JUDGEMENTAL (clase 0) en validación.
    # Con umbral=0.5 el modelo infraclasifica JUDGEMENTAL (recall 49%).
    # Bajarlo permite detectar más casos JUDGEMENTAL a costa de algo de precisión.
    best_thresh_22, best_f1_22 = 0.5, 0.0
    for t in np.arange(0.25, 0.60, 0.01):
        preds_t = np.where(probs_vl22[:, 0] >= t, 0, 1)  # 0=JUDGEMENTAL, 1=DIRECT
        f1_t = f1_score(vl_labs_22, preds_t, average='macro', zero_division=0)
        if f1_t > best_f1_22:
            best_f1_22    = f1_t
            best_thresh_22 = t

    print(f"  Umbral óptimo JUDGEMENTAL: {best_thresh_22:.2f} (F1 macro en val de calibración: {best_f1_22:.4f})")

    # Métrica final en el HOLDOUT de T2.2 (el umbral llega ya fijado en val)
    ds_ho22 = MemeDataset(t_feats22[ho22], i_feats22[ho22], labs22[ho22])
    ld_ho22 = DataLoader(ds_ho22, batch_size=BATCH_SIZE, shuffle=False)
    probs_ho22  = soft_vote(models_22, ld_ho22, DEVICE)
    ho_labs_22  = labs22[ho22]
    ho_preds_22 = np.where(probs_ho22[:, 0] >= best_thresh_22, 0, 1)
    print("\n  [Evaluación HOLDOUT — T2.2 Soft Voting + umbral calibrado en val]")
    print(classification_report(ho_labs_22, ho_preds_22,
                                 target_names=['JUDGEMENTAL', 'DIRECT'], zero_division=0))

    # ──────────────────────────────────────────────────────────────────────
    # PASO 6: Predecir sobre test y guardar en formato EXIST 2025
    # ──────────────────────────────────────────────────────────────────────
    print("\n[PASO 6] Generando predicciones sobre el conjunto de test...")

    test_ids = list(df_test['id_EXIST'])

    # T2.1: soft voting de todos los base models
    probs_21 = soft_vote(models_21, ld_te21, DEVICE)   # [N, 2]
    preds_21 = probs_21.argmax(axis=1)
    # Nombre con 5 segmentos separados por '_' para cumplir el validador:
    # task2_1_EXIST2025_UPM_hard.json → split('_') = ['task2','1','EXIST2025','UPM','hard.json']
    print("\n  [Guardando T2.1]")
    save_task21(test_ids, preds_21, probs_21, "task2_1_EXIST2025_UPM")

    # T2.2: correr en TODOS los memes con umbral calibrado
    ds_te22_all = MemeDataset(text_te, img_te)
    ld_te22_all = DataLoader(ds_te22_all, batch_size=BATCH_SIZE, shuffle=False)
    probs_22_all = soft_vote(models_22, ld_te22_all, DEVICE)   # [N, 2]

    # Aplicar umbral calibrado en validación para ajustar las probs antes de combinar.
    # Escalamos P(JUDGEMENTAL) por (0.5 / best_thresh_22) para que el argmax
    # refleje el umbral óptimo sin cambiar el formato de salida.
    probs_22_cal = probs_22_all.copy()
    if best_thresh_22 != 0.5:
        scale = 0.5 / best_thresh_22
        probs_22_cal[:, 0] = np.clip(probs_22_all[:, 0] * scale, 0, 1)
        probs_22_cal[:, 1] = np.clip(1 - probs_22_cal[:, 0], 0, 1)

    sexist_mask = preds_21 == 1
    print("\n  [Guardando T2.2]")
    _, _, preds_22_combined = save_task22(
        test_ids, preds_21, probs_21, probs_22_cal, "task2_2_EXIST2025_UPM"
    )
    preds_22 = preds_22_combined[sexist_mask]  # solo para métricas de sexistas
    print(f"\n  Memes sexistas predichos: {sexist_mask.sum()}/{len(test_ids)}")
    # Contar sobre las etiquetas hard finales (ya calibradas): DIRECT = índice 2
    n_direct = int((preds_22 == 2).sum())
    print(f"  DIRECT: {n_direct} | JUDGEMENTAL: {int(sexist_mask.sum()) - n_direct}")

    # ──────────────────────────────────────────────────────────────────────
    # PASO 7: Guardar modelos entrenados (solo si no se cargaron desde disco)
    # ──────────────────────────────────────────────────────────────────────
    if not LOAD_PRETRAINED:
        for i, m in enumerate(models_21[:N_FOLDS_T21]):
            torch.save(m.state_dict(),
                       os.path.join(OUTPUT_DIR, f"model_task21_fold{i}.pt"))
        torch.save(model_cross_21.state_dict(), os.path.join(OUTPUT_DIR, "model_task21_crossmodal.pt"))
        for i, m in enumerate(models_22):
            torch.save(m.state_dict(), os.path.join(OUTPUT_DIR, f"model_task22_s{seeds_22[i]}.pt"))
    else:
        print("  [PASO 7] Modelos cargados desde disco, no se sobreescriben.")

    f1_21 = f1_score(vl_labs_21, vl_preds_21, average='macro', zero_division=0)
    f1_22 = f1_score(ho_labs_22, ho_preds_22, average='macro', zero_division=0)

    print("\n" + "=" * 62)
    print("  EJECUCIÓN COMPLETADA")
    print("=" * 62)
    print(f"  T2.1 F1 macro (holdout 15%, soft voting):        {f1_21:.4f}")
    print(f"  T2.2 F1 macro (holdout 15%, umbral={best_thresh_22:.2f}):    {f1_22:.4f}")
    print(f"  Predicciones en:  {OUTPUT_DIR}")
    print(f"  Modelos en:       {OUTPUT_DIR}")
    print("=" * 62)

