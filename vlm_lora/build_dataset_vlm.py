# =============================================================================
# EXIST 2025 x Motor de LoRAs — generador de dataset VLM (Tarea 2.1)
# =============================================================================
# Convierte el dataset de memes EXIST 2025 al formato ChatML multimodal que
# consume el VLMTrainer del Motor de LoRAs, respetando EXACTAMENTE los mismos
# splits (dev/holdout persistidos + fold 0) que la evaluación del pipeline
# clásico, para que la comparación final sea manzanas con manzanas.
#
# Salidas (en exist_vlm/):
#   img_512/                  — memes redimensionados (lado mayor <= 512 px)
#   t21_train.jsonl           — fold-0 train (ChatML multimodal, para VLMTrainer)
#   t21_val.meta.jsonl        — fold-0 val   (id, image, prompt, label; calibración)
#   t21_holdout.meta.jsonl    — holdout 15%  (id, image, prompt, label; SOLO evaluación final)
# =============================================================================

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

BASE = Path(r"C:\Users\Felipe\Desktop\Proyecto")
OUT = BASE / "exist_vlm"
IMG_OUT = OUT / "img_512"
OUT.mkdir(exist_ok=True)
IMG_OUT.mkdir(exist_ok=True)

DS = BASE / "EXIST 2025 Memes Dataset-20260421T070023Z-3-001" / "EXIST 2025 Memes Dataset"
TRAIN_JSON = DS / "training" / "EXIST2025_training.json"
TRAIN_IMG = DS / "training" / "memes"
GOLD = BASE / "Resultados Training-20260421T072644Z-3-001" / "Resultados Training"
GOLD_HARD = GOLD / "EXIST2025_training_task2_1_gold_hard.json"
GOLD_SOFT = GOLD / "EXIST2025_training_task2_1_gold_soft.json"
OCR_CACHE = BASE / "cache" / "ocr_text_train.json"
HOLDOUT_SPLIT = BASE / "predictions" / "holdout_split_t21.json"

SEED = 42
MAX_SIDE = 512
MAX_TEXT_CHARS = 600

PROMPT = (
    "Texto extraído del meme: «{texto}»\n"
    "¿Es este meme sexista? Responde únicamente YES o NO."
)


def merge_text_sources(json_text, ocr_text):
    """Réplica exacta de la fusión JSON+OCR de Proyecto.py."""
    json_text = (json_text or "").strip()
    ocr_text = (ocr_text or "").strip()
    if not json_text and not ocr_text:
        return ""
    if not json_text:
        return ocr_text
    if not ocr_text:
        return json_text
    new_tokens = set(ocr_text.lower().split()) - set(json_text.lower().split())
    if len(new_tokens) >= 6:
        return f"{json_text} | {ocr_text}"
    return json_text if len(json_text) >= len(ocr_text) else ocr_text


def main():
    with open(TRAIN_JSON, encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(GOLD_HARD, encoding="utf-8") as f:
        gold_hard = {e["id"]: e["value"] for e in json.load(f)}
    with open(GOLD_SOFT, encoding="utf-8") as f:
        gold_soft = {e["id"]: e["value"] for e in json.load(f)}
    with open(OCR_CACHE, encoding="utf-8") as f:
        ocr = json.load(f)

    # --- Registros con la MISMA lógica de etiquetado que Proyecto.py ---
    records = []
    for id_e, entry in train_raw.items():
        if id_e not in gold_soft:
            continue
        img_path = TRAIN_IMG / entry["meme"]
        if not img_path.exists():
            continue
        soft = gold_soft[id_e]
        p_yes, p_no = float(soft.get("YES", 0.5)), float(soft.get("NO", 0.5))
        if id_e in gold_hard:
            label = 1 if gold_hard[id_e] == "YES" else 0
        else:
            label = 1 if p_yes >= p_no else 0
        text = merge_text_sources(entry.get("text", "") or "", ocr.get(id_e, ""))
        records.append({"id": id_e, "img": img_path, "label": label, "text": text})

    n = len(records)
    labels = np.array([r["label"] for r in records])
    print(f"Muestras: {n} | YES: {labels.sum()} | NO: {(labels == 0).sum()}")

    # --- Splits idénticos a la evaluación del pipeline clásico ---
    with open(HOLDOUT_SPLIT, encoding="utf-8") as f:
        split = json.load(f)
    idx_dev = np.array(split["dev"])
    idx_hold = np.array(split["holdout"])
    assert len(idx_dev) + len(idx_hold) == n, "El split persistido no cuadra con el dataset"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_rel, vl_rel = next(iter(skf.split(np.zeros(len(idx_dev)), labels[idx_dev])))
    idx_tr, idx_vl = idx_dev[tr_rel], idx_dev[vl_rel]
    print(f"fold-0 train: {len(idx_tr)} | fold-0 val: {len(idx_vl)} | holdout: {len(idx_hold)}")

    # --- Redimensionar imágenes (una vez) ---
    print("Redimensionando imágenes...")
    resized = {}
    for i, r in enumerate(records):
        dst = IMG_OUT / (Path(r["img"]).stem + ".jpg")
        if not dst.exists():
            im = Image.open(r["img"]).convert("RGB")
            w, h = im.size
            if max(w, h) > MAX_SIDE:
                s = MAX_SIDE / max(w, h)
                im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
            im.save(dst, "JPEG", quality=90)
        resized[r["id"]] = str(dst.resolve())
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")

    def make_prompt(r):
        t = r["text"][:MAX_TEXT_CHARS]
        return PROMPT.format(texto=t if t else "(sin texto detectado)")

    # --- t21_train.jsonl (ChatML multimodal para VLMTrainer) ---
    with open(OUT / "t21_train.jsonl", "w", encoding="utf-8") as f:
        for i in idx_tr:
            r = records[i]
            ex = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": resized[r["id"]]},
                            {"type": "text", "text": make_prompt(r)},
                        ],
                    },
                    {"role": "assistant", "content": "YES" if r["label"] == 1 else "NO"},
                ]
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # --- metadatos de val y holdout (para el script de evaluación) ---
    for name, idxs in [("t21_val", idx_vl), ("t21_holdout", idx_hold)]:
        with open(OUT / f"{name}.meta.jsonl", "w", encoding="utf-8") as f:
            for i in idxs:
                r = records[i]
                f.write(json.dumps({
                    "id": r["id"],
                    "image": resized[r["id"]],
                    "prompt": make_prompt(r),
                    "label": int(r["label"]),
                }, ensure_ascii=False) + "\n")

    print("Listo:")
    print(f"  {OUT / 't21_train.jsonl'}  ({len(idx_tr)} ejemplos)")
    print(f"  {OUT / 't21_val.meta.jsonl'}  ({len(idx_vl)})")
    print(f"  {OUT / 't21_holdout.meta.jsonl'}  ({len(idx_hold)})")


if __name__ == "__main__":
    main()
