# =============================================================================
# EXIST 2025 x Motor de LoRAs — Etapa A: entrenar Qwen2-VL-2B con el VLMTrainer
# =============================================================================
# Usa el VLMTrainer del Motor de LoRAs tal cual (sin modificarlo).
# bf16 puro (regla de oro nº 1 del Motor: 4-bit NF4 crashea en Windows+CUDA 12.4).
#
# Lanzar con:  $env:PYTHONUTF8="1"; python exist_vlm\train_t21_vlm.py
# =============================================================================

import sys
from pathlib import Path

MOTOR = r"C:\Users\Felipe\Desktop\Proyecto\motor-de-loras-custom"
sys.path.insert(0, MOTOR)

from motor.trainer_vlm import VLMTrainer  # noqa: E402

BASE = Path(r"C:\Users\Felipe\Desktop\Proyecto")

trainer = VLMTrainer(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    load_in_4bit=False,          # bf16 puro en Windows (regla de oro nº 1)
    max_seq_length=1024,
    cache_dir=str(BASE / "cache"),
)

metrics = trainer.fit(
    dataset_path=str(BASE / "exist_vlm" / "t21_train.jsonl"),
    output_dir=str(BASE / "exist_vlm" / "adapter_t21_qwen2vl2b"),
    epochs=3,
    batch_size=1,
    grad_accum=16,
    learning_rate=2e-4,
    eval_split=0.05,             # eval interna del trainer (dentro del fold-0 train)
)

print("\nMétricas del entrenamiento:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
