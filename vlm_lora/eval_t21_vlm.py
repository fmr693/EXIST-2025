# =============================================================================
# EXIST 2025 x Motor de LoRAs — evaluación del VLM afinado (Tarea 2.1)
# =============================================================================
# Mide P(YES) vs P(NO) por log-probabilidades de la respuesta completa (robusto
# a la tokenización), igual en espíritu al softmax del pipeline clásico.
#
# Uso:
#   # calibración (val del fold 0):
#   python exist_vlm\eval_t21_vlm.py --meta exist_vlm\t21_val.meta.jsonl --adapter exist_vlm\adapter_t21_qwen2vl2b
#   # medición FINAL (holdout, UNA sola vez, con el umbral elegido en val):
#   python exist_vlm\eval_t21_vlm.py --meta exist_vlm\t21_holdout.meta.jsonl --adapter ... --threshold 0.5
#
# Sin --adapter evalúa el modelo base (zero-shot), útil como referencia.
# =============================================================================

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, f1_score

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR = r"C:\Users\Felipe\Desktop\Proyecto\cache"
CANDIDATES = ["YES", "NO"]


def build_messages(row):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": row["image"]},
            {"type": "text", "text": row["prompt"]},
        ],
    }]


@torch.no_grad()
def score_candidates(model, processor, row):
    """Devuelve log P(candidato | prompt+imagen) para YES y NO."""
    img = Image.open(row["image"]).convert("RGB")
    prompt_text = processor.apply_chat_template(
        build_messages(row), tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
    n_prompt = prompt_inputs["input_ids"].shape[1]

    logps = []
    for cand in CANDIDATES:
        inputs = processor(text=[prompt_text + cand], images=[img], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logits = model(**inputs).logits[0]              # [seq, vocab]
        ids = inputs["input_ids"][0]
        # logprob de los tokens del candidato (posiciones n_prompt..fin)
        lp = 0.0
        logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
        for pos in range(n_prompt, ids.shape[0]):
            lp += logprobs[pos - 1, ids[pos]].item()
        logps.append(lp)
    return logps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Umbral P(YES) fijo (para el holdout). Sin él, hace barrido (calibración).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.meta, encoding="utf-8")]
    print(f"Muestras: {len(rows)}  |  Split: {args.meta}")

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as AutoVLM
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVLM
    print(f"Cargando {args.base} (bf16)...")
    processor = AutoProcessor.from_pretrained(args.base, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoVLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda",
        cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel
        print(f"Aplicando adapter LoRA: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    t0 = time.time()
    p_yes_all, labels = [], []
    for i, row in enumerate(rows):
        lp_yes, lp_no = score_candidates(model, processor, row)
        # softmax sobre los dos candidatos
        m = max(lp_yes, lp_no)
        e_yes, e_no = np.exp(lp_yes - m), np.exp(lp_no - m)
        p_yes_all.append(float(e_yes / (e_yes + e_no)))
        labels.append(row["label"])
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(rows)}  ({rate:.1f} muestras/s, ETA {int((len(rows)-i-1)/rate)}s)")

    p_yes_all = np.array(p_yes_all)
    labels = np.array(labels)

    out_path = args.out or (str(Path(args.meta).with_suffix("")) +
                            (".adapter" if args.adapter else ".base") + ".probs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{"id": r["id"], "label": int(l), "p_yes": float(p)}
                   for r, l, p in zip(rows, labels, p_yes_all)], f)
    print(f"Probabilidades guardadas en {out_path}")

    if args.threshold is not None:
        preds = (p_yes_all >= args.threshold).astype(int)
        print(f"\n=== RESULTADO con umbral fijo {args.threshold:.2f} ===")
        print(classification_report(labels, preds, target_names=["NO", "YES"], zero_division=0))
        print(f"F1-YES:   {f1_score(labels, preds, pos_label=1, zero_division=0):.4f}")
        print(f"F1 macro: {f1_score(labels, preds, average='macro', zero_division=0):.4f}")
    else:
        print("\n=== Calibración (barrido de umbral en este split) ===")
        preds05 = (p_yes_all >= 0.5).astype(int)
        print(f"Umbral 0.50 → F1-YES {f1_score(labels, preds05, pos_label=1, zero_division=0):.4f} "
              f"| F1 macro {f1_score(labels, preds05, average='macro', zero_division=0):.4f}")
        best_y, best_m = (0.5, 0.0), (0.5, 0.0)
        for t in np.arange(0.05, 0.96, 0.01):
            preds = (p_yes_all >= t).astype(int)
            fy = f1_score(labels, preds, pos_label=1, zero_division=0)
            fm = f1_score(labels, preds, average="macro", zero_division=0)
            if fy > best_y[1]:
                best_y = (t, fy)
            if fm > best_m[1]:
                best_m = (t, fm)
        print(f"Mejor F1-YES:   {best_y[1]:.4f} con umbral {best_y[0]:.2f}")
        print(f"Mejor F1 macro: {best_m[1]:.4f} con umbral {best_m[0]:.2f}")


if __name__ == "__main__":
    main()
