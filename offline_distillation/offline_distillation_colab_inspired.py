#!/usr/bin/env python3
"""
Offline Knowledge Distillation — Vision-Language

Phase 1 : Générer les logits du teacher 7B
Phase 2 : Entraîner le student 2B avec les logits + images re-processées

Adapté depuis le notebook Colab pour exécution locale.
Données BoundingDocs lues depuis /mounts/datasets/datasets/BoundingDocs
"""

# ============================================================
# 0. CONFIGURATION — modifie ici
# ============================================================

# Modèles
TEACHER_MODEL = "letxbe/qwen2-7b-BoundingDocs-rephrased"
STUDENT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# Dataset local
DATASET_PATH = "/mounts/datasets/datasets/BoundingDocs"
MAX_SAMPLES = None  # None pour tout le dataset, ou un int pour test rapide

# Chemins de sortie
LOGITS_DIR = "./cached_logits_vl"
OUTPUT_DIR = "./distilled_docexplainer"

# Teacher
LOAD_IN_4BIT = True  # True pour GPU <= 16 Go VRAM, False si A100/H100

# Student — hyperparamètres d'entraînement
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
TEMPERATURE = 2.0
ALPHA = 0.5  # 0.0 = fine-tuning pur, 1.0 = distillation pure
SAVE_EVERY = 500

# Constantes internes
TOP_K_LOGITS = 64
MAX_SEQ_LEN = 1024

# Phases à exécuter (permet de relancer seulement la phase 2 si logits déjà générés)
RUN_PHASE1 = True
RUN_PHASE2 = True

# ============================================================
# 1. IMPORTS
# ============================================================

import json
import os
import gc
import sys
import argparse
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from datasets import load_from_disk, load_dataset
from tqdm.auto import tqdm
from PIL import Image

from qwen_vl_utils import process_vision_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Dtype adapté au GPU
STUDENT_DTYPE = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16

print(f"Student dtype: {STUDENT_DTYPE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ============================================================
# 2. PROMPTS DU TEACHER
# ============================================================

SYSTEM_MESSAGE = (
    "You are a Vision Language Model specialized in extracting information "
    "from document images.\n"
    "Your task is to analyze the provided document image and extract relevant "
    "information accurately.\n"
    "Documents may contain text, tables, forms, and structured or unstructured data.\n"
    "Ensure responses are precise and concise, without additional explanations "
    "unless required for clarity."
)

TEMPLATE_PROMPT = (
    '<starttask>\n'
    'Answer the following question about the document:\n'
    'Question: "{QUESTION}"\n'
    'Answer completing the following format:\n'
    "'''json\n"
    '{{"value": ""}}\n'
    "'''\n"
    '<endtask>'
)

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================


def load_local_dataset(dataset_path: str, max_samples: Optional[int] = None):
    """
    Charge le dataset BoundingDocs depuis le disque local.

    Tente d'abord load_from_disk (format Arrow/HF sauvegardé),
    puis load_dataset sur un dossier (format Parquet, CSV, JSON, ImageFolder…).
    """
    dataset_path = Path(dataset_path)
    print(f"Chargement du dataset depuis : {dataset_path}")

    # Lister le contenu pour diagnostiquer le format
    if dataset_path.exists():
        contents = list(dataset_path.iterdir())
        print(f"  Contenu du répertoire ({len(contents)} éléments) :")
        for p in sorted(contents)[:20]:
            print(f"    {p.name}  ({'dir' if p.is_dir() else f'{p.stat().st_size / 1e6:.1f} MB'})")
        if len(contents) > 20:
            print(f"    ... et {len(contents) - 20} autres")
    else:
        print(f"  ERREUR : le chemin {dataset_path} n'existe pas !")
        sys.exit(1)

    # --- Tentative 1 : load_from_disk (format Arrow natif HF) ---
    try:
        ds = load_from_disk(str(dataset_path))
        # Si c'est un DatasetDict, prendre le split "train"
        if hasattr(ds, "keys"):
            if "train" in ds:
                ds = ds["train"]
            else:
                first_key = next(iter(ds.keys()))
                print(f"  Pas de split 'train', utilisation de '{first_key}'")
                ds = ds[first_key]
        print(f"  Dataset chargé via load_from_disk : {len(ds)} samples")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
            print(f"  Réduit à {len(ds)} samples (MAX_SAMPLES={max_samples})")
        return ds
    except Exception as e:
        print(f"  load_from_disk a échoué : {e}")

    # --- Tentative 2 : load_dataset sur le dossier ---
    try:
        ds = load_dataset(str(dataset_path), split="train")
        print(f"  Dataset chargé via load_dataset : {len(ds)} samples")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
            print(f"  Réduit à {len(ds)} samples (MAX_SAMPLES={max_samples})")
        return ds
    except Exception as e:
        print(f"  load_dataset a échoué : {e}")

    # --- Tentative 3 : load_dataset avec trust_remote_code ---
    try:
        ds = load_dataset(str(dataset_path), split="train", trust_remote_code=True)
        print(f"  Dataset chargé via load_dataset (trust_remote_code) : {len(ds)} samples")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds
    except Exception as e:
        print(f"  load_dataset (trust_remote_code) a échoué : {e}")

    print("ERREUR : Impossible de charger le dataset. Vérifiez le format.")
    sys.exit(1)


def format_boundingdocs_sample(sample: dict) -> Optional[dict]:
    """
    Transforme un sample brut du dataset BoundingDocs
    en format chat Qwen2-VL (messages + answer).
    Gère les deux formats de Q&A : liste ou dict.
    Retourne None si le sample est invalide.
    """
    try:
        qa_data = (
            json.loads(sample["Q&A"])
            if isinstance(sample["Q&A"], str)
            else sample["Q&A"]
        )
    except (json.JSONDecodeError, KeyError):
        return None

    if not qa_data:
        return None

    # Gérer les deux formats possibles
    if isinstance(qa_data, list):
        qa = qa_data[0]
    elif isinstance(qa_data, dict):
        first_key = next(iter(qa_data))
        qa = qa_data[first_key]
    else:
        return None

    question = qa.get("rephrased_question") or qa.get("question", "")
    if not question:
        return None

    answers = qa.get("answers", [])
    if not answers:
        return None

    answer_value = answers[0].get("value", "")
    if not answer_value:
        return None

    image = None
    if sample.get("doc_images") and len(sample["doc_images"]) > 0:
        image = sample["doc_images"][0]

    if image is None:
        return None

    prompt = TEMPLATE_PROMPT.format(QUESTION=question)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    return {
        "messages": messages,
        "answer": json.dumps({"value": answer_value}),
        "answer_raw": answer_value,
    }


# ============================================================
# 4. DATASET DE DISTILLATION (Phase 2)
# ============================================================

class VLDistillationDataset(Dataset):
    """
    Charge les logits du teacher + image JPEG depuis les .npz.
    Re-processe l'image avec le processor du student pour obtenir
    des input_ids et pixel_values cohérents.
    """

    def __init__(self, logits_dir, student_processor, max_seq_len=MAX_SEQ_LEN):
        self.logits_dir = Path(logits_dir)
        self.processor = student_processor
        self.max_seq_len = max_seq_len

        with open(self.logits_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.num_samples = self.metadata["num_samples"]
        self.top_k = self.metadata["top_k"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = np.load(
            self.logits_dir / f"sample_{idx:06d}.npz", allow_pickle=True
        )

        # Charger les logits du teacher
        top_k_values = torch.tensor(data["top_k_values"].astype(np.float32))
        top_k_indices = torch.tensor(data["top_k_indices"], dtype=torch.long)

        # Reconstruire l'image depuis les bytes JPEG
        jpeg_bytes = data["image_jpeg"].tobytes()
        image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")

        # Reconstruire les messages et re-processer avec le student
        question_text = str(data["question"][0])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Tronquer manuellement après le processing
        L = min(len(input_ids), self.max_seq_len)
        input_ids = input_ids[:L]
        attention_mask = attention_mask[:L]

        # Tronquer les logits du teacher à la même longueur
        T = min(top_k_values.shape[0], L)
        top_k_values = top_k_values[:T]
        top_k_indices = top_k_indices[:T]
        if T < L:
            pad_len = L - T
            top_k_values = F.pad(top_k_values, (0, 0, 0, pad_len), value=0.0)
            top_k_indices = F.pad(top_k_indices, (0, 0, 0, pad_len), value=0)

        # Labels : décalage causal
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "teacher_top_k_values": top_k_values,
            "teacher_top_k_indices": top_k_indices,
        }

        if "pixel_values" in inputs:
            item["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            item["image_grid_thw"] = inputs["image_grid_thw"]

        return item


# ============================================================
# 5. COLLATE & LOSS
# ============================================================

def vl_collate_fn(batch):
    """Collate : padding dynamique pour le texte, concaténation pour les pixels."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    out = {
        k: []
        for k in [
            "input_ids", "labels", "attention_mask",
            "teacher_top_k_values", "teacher_top_k_indices",
        ]
    }

    for item in batch:
        pad = max_len - item["input_ids"].shape[0]
        out["input_ids"].append(F.pad(item["input_ids"], (0, pad), value=0))
        out["labels"].append(F.pad(item["labels"], (0, pad), value=-100))
        out["attention_mask"].append(
            F.pad(item["attention_mask"], (0, pad), value=0)
        )
        out["teacher_top_k_values"].append(
            F.pad(item["teacher_top_k_values"], (0, 0, 0, pad), value=0.0)
        )
        out["teacher_top_k_indices"].append(
            F.pad(item["teacher_top_k_indices"], (0, 0, 0, pad), value=0)
        )

    result = {k: torch.stack(v) for k, v in out.items()}

    has_pixels = any("pixel_values" in item for item in batch)
    if has_pixels:
        all_pv = [item["pixel_values"] for item in batch if "pixel_values" in item]
        all_thw = [item["image_grid_thw"] for item in batch if "image_grid_thw" in item]
        if all_pv:
            result["pixel_values"] = torch.cat(all_pv, dim=0)
        if all_thw:
            result["image_grid_thw"] = torch.cat(all_thw, dim=0)

    return result


def distillation_loss(
    student_logits, t_vals, t_idx, labels, mask, vocab_size,
    temperature=2.0, alpha=0.5,
):
    """Loss combinée : KL divergence (soft labels) + Cross-entropy (hard labels)."""
    s_log = student_logits[:, :-1, :].contiguous()
    s_lab = labels[:, 1:].contiguous()
    s_mask = mask[:, 1:].contiguous().float()
    t_v = t_vals[:, :-1, :].contiguous()
    t_i = t_idx[:, :-1, :].contiguous()
    bs, seq, _ = s_log.shape

    # Cross-entropy (hard labels)
    ce = F.cross_entropy(
        s_log.view(-1, vocab_size), s_lab.view(-1),
        ignore_index=-100, reduction="none",
    ).view(bs, seq)
    ce_loss = (ce * s_mask).sum() / s_mask.sum().clamp(min=1)

    # KL divergence (soft labels du teacher)
    teacher_full = torch.full(
        (bs, seq, vocab_size), -1e4,
        device=s_log.device, dtype=s_log.dtype,
    )
    teacher_full.scatter_(2, t_i, t_v.to(s_log.dtype))

    student_lp = F.log_softmax(s_log / temperature, dim=-1)
    teacher_p = F.softmax(teacher_full / temperature, dim=-1)

    kl = F.kl_div(student_lp, teacher_p, reduction="none").sum(-1)
    kl_loss = (kl * s_mask).sum() / s_mask.sum().clamp(min=1) * (temperature ** 2)

    total = alpha * kl_loss + (1 - alpha) * ce_loss
    return total, kl_loss.item(), ce_loss.item()


# ============================================================
# 6. PHASE 1 — Générer les logits du teacher
# ============================================================

def run_phase1(dataset):
    """
    Le teacher fait de l'inférence sample par sample. On sauvegarde :
    - les top-K logits par position
    - les input_ids (tokens)
    - le dataset_index (pour retrouver l'image en Phase 2)
    """
    print("\n" + "=" * 60)
    print("PHASE 1 : Génération des logits du teacher")
    print("=" * 60)

    # Charger le processor
    print("Loading processor from Qwen/Qwen2-VL-7B-Instruct...")
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    teacher_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_fast=True,
        trust_remote_code=True,
    )
    print("Processor OK")

    # Charger le teacher
    print(f"Loading teacher: {TEACHER_MODEL}")
    if LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        teacher = AutoModelForImageTextToText.from_pretrained(
            TEACHER_MODEL,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        teacher = AutoModelForImageTextToText.from_pretrained(
            TEACHER_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    teacher.eval()
    n_params = sum(p.numel() for p in teacher.parameters()) / 1e9
    print(f"Teacher loaded: {n_params:.2f}B params")

    # Préparer le répertoire de sortie
    logits_dir = Path(LOGITS_DIR)
    logits_dir.mkdir(parents=True, exist_ok=True)
    teacher_processor.save_pretrained(logits_dir / "processor")

    metadata = {
        "teacher_model": TEACHER_MODEL,
        "dataset": DATASET_PATH,
        "top_k": TOP_K_LOGITS,
        "max_seq_len": MAX_SEQ_LEN,
        "num_samples": 0,
        "model_type": "vision_language",
        "system_message": SYSTEM_MESSAGE,
        "has_image_bytes": True,
    }

    # Vérification rapide
    print("Vérification sur les premiers samples...")
    found_valid = False
    for test_idx in range(min(10, len(dataset))):
        sample_test = dataset[test_idx]
        formatted_test = format_boundingdocs_sample(sample_test)
        if formatted_test:
            print(f"  Sample {test_idx} valide — réponse : {formatted_test['answer_raw'][:80]}")
            found_valid = True
            break
    if not found_valid:
        print("  ATTENTION : aucun sample valide dans les 10 premiers !")

    # Générer les logits
    sample_idx = 0
    skipped = 0

    for i in tqdm(range(len(dataset)), desc="Phase 1: generating logits"):
        sample = dataset[i]
        formatted = format_boundingdocs_sample(sample)
        if formatted is None:
            skipped += 1
            continue

        try:
            messages = formatted["messages"]

            text = teacher_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = teacher_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
                max_length=MAX_SEQ_LEN,
                truncation=True,
            )
            inputs = {k: v.to(teacher.device) for k, v in inputs.items()}

            if inputs["input_ids"].shape[1] < 2:
                skipped += 1
                continue

            with torch.no_grad():
                outputs = teacher(**inputs)
                logits = outputs.logits[0]

            top_k_vals, top_k_idx = torch.topk(logits, TOP_K_LOGITS, dim=-1)

            # Sauvegarder l'image en JPEG bytes
            img_buffer = BytesIO()
            image_inputs[0].save(img_buffer, format="JPEG", quality=95)
            image_bytes = np.frombuffer(img_buffer.getvalue(), dtype=np.uint8)

            np.savez_compressed(
                logits_dir / f"sample_{sample_idx:06d}.npz",
                top_k_values=top_k_vals.cpu().half().numpy(),
                top_k_indices=top_k_idx.cpu().numpy().astype(np.int32),
                attention_mask=inputs.get(
                    "attention_mask",
                    torch.ones_like(inputs["input_ids"]),
                )[0].cpu().numpy().astype(np.int8),
                image_jpeg=image_bytes,
                question=np.array([formatted["messages"][1]["content"][1]["text"]], dtype=object),
                answer=np.array([formatted["answer_raw"]], dtype=object),
            )
            sample_idx += 1
        except Exception as e:
            print(f"  Warning: skipping sample {i}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
            continue

        if i % 100 == 0:
            torch.cuda.empty_cache()

    metadata["num_samples"] = sample_idx
    metadata["skipped"] = skipped
    with open(logits_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPhase 1 terminée ! {sample_idx} samples sauvegardés, {skipped} skippés -> {logits_dir}")

    # Vérification
    if sample_idx > 0:
        data_check = np.load(logits_dir / "sample_000000.npz", allow_pickle=True)
        print("Contenu du premier .npz :")
        for key in data_check.files:
            arr = data_check[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    # Libérer la VRAM
    del teacher
    del teacher_processor
    torch.cuda.empty_cache()
    gc.collect()
    print("Teacher supprimé de la VRAM")


# ============================================================
# 7. PHASE 2 — Entraîner le student
# ============================================================

def run_phase2():
    """
    Le student charge les logits du teacher + re-processe les images.
    """
    print("\n" + "=" * 60)
    print("PHASE 2 : Entraînement du student par distillation")
    print("=" * 60)

    # Charger le student
    print(f"Loading student: {STUDENT_MODEL}")
    student_processor = AutoProcessor.from_pretrained(
        STUDENT_MODEL,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    student = AutoModelForImageTextToText.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=STUDENT_DTYPE,
        trust_remote_code=True,
    ).to("cuda")
    student.train()

    VOCAB_SIZE = student.config.text_config.vocab_size
    n_params = sum(p.numel() for p in student.parameters()) / 1e9
    print(f"Student: {n_params:.2f}B parameters, dtype={STUDENT_DTYPE}")
    print(f"Vocab size: {VOCAB_SIZE}")

    # Vérifier les métadonnées
    with open(Path(LOGITS_DIR) / "metadata.json") as f:
        meta = json.load(f)
    print(f"Logits metadata: {json.dumps(meta, indent=2)}")

    # Créer les DataLoaders
    torch.manual_seed(42)
    dataset = VLDistillationDataset(LOGITS_DIR, student_processor, MAX_SEQ_LEN)
    val_size = max(1, int(0.05 * len(dataset)))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size],
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=vl_collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=vl_collate_fn, num_workers=2, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Optimizer, scheduler, gradient checkpointing
    device = torch.device("cuda")
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LEARNING_RATE, weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.05 * total_steps), total_steps,
    )

    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        print("Gradient checkpointing activé")

    use_scaler = STUDENT_DTYPE == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total optimizer steps: {total_steps // GRADIENT_ACCUMULATION_STEPS}")
    print(f"Warmup steps: {int(0.05 * total_steps)}")

    # --------------------------------------------------------
    # BOUCLE D'ENTRAÎNEMENT
    # --------------------------------------------------------
    best_val_loss = float("inf")
    global_step = 0
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        student.train()
        ep_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            pixel_values = batch.pop("pixel_values", None)
            image_grid_thw = batch.pop("image_grid_thw", None)

            batch = {k: v.to(device) for k, v in batch.items()}
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            with torch.amp.autocast("cuda", dtype=STUDENT_DTYPE):
                forward_kwargs = dict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                if pixel_values is not None:
                    forward_kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    forward_kwargs["image_grid_thw"] = image_grid_thw

                out = student(**forward_kwargs)

                loss, kl, ce = distillation_loss(
                    out.logits,
                    batch["teacher_top_k_values"],
                    batch["teacher_top_k_indices"],
                    batch["labels"],
                    batch["attention_mask"],
                    VOCAB_SIZE,
                    TEMPERATURE,
                    ALPHA,
                )
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            ep_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            n_batch += 1
            pbar.set_postfix(
                loss=f"{ep_loss / n_batch:.4f}",
                kl=f"{kl:.4f}",
                ce=f"{ce:.4f}",
            )

            if global_step > 0 and global_step % SAVE_EVERY == 0:
                ckpt = output_dir / f"checkpoint-{global_step}"
                student.save_pretrained(ckpt)
                student_processor.save_pretrained(ckpt)
                print(f"  Checkpoint sauvegardé: {ckpt}")

        avg_train = ep_loss / max(n_batch, 1)
        train_losses.append(avg_train)

        # Validation
        student.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch.pop("pixel_values", None)
                image_grid_thw = batch.pop("image_grid_thw", None)

                batch = {k: v.to(device) for k, v in batch.items()}
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(device)

                with torch.amp.autocast("cuda", dtype=STUDENT_DTYPE):
                    forward_kwargs = dict(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    if pixel_values is not None:
                        forward_kwargs["pixel_values"] = pixel_values
                    if image_grid_thw is not None:
                        forward_kwargs["image_grid_thw"] = image_grid_thw

                    out = student(**forward_kwargs)

                    loss, _, _ = distillation_loss(
                        out.logits,
                        batch["teacher_top_k_values"],
                        batch["teacher_top_k_indices"],
                        batch["labels"],
                        batch["attention_mask"],
                        VOCAB_SIZE,
                        TEMPERATURE,
                        ALPHA,
                    )
                val_total += loss.item()
                val_n += 1

        avg_val = val_total / max(val_n, 1)
        val_losses.append(avg_val)
        print(f"\n  Epoch {epoch + 1}: train={avg_train:.4f} | val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            student.save_pretrained(output_dir / "best")
            student_processor.save_pretrained(output_dir / "best")
            print("  -> Nouveau meilleur modèle sauvegardé!")

    # Sauvegarder le modèle final
    student.save_pretrained(output_dir / "final")
    student_processor.save_pretrained(output_dir / "final")
    print(f"\nTerminé ! Meilleure val loss: {best_val_loss:.4f}")
    print(f"Modèles dans: {output_dir}")

    # Graphe des losses (sauvegardé en PNG)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(train_losses) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            epochs_range = range(1, len(train_losses) + 1)
            ax.plot(epochs_range, train_losses, "b-o", label="Train loss")
            ax.plot(epochs_range, val_losses, "r-o", label="Val loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Distillation Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = output_dir / "loss_curve.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Courbe de loss sauvegardée: {plot_path}")
    except ImportError:
        print("matplotlib non disponible, pas de graphe généré.")

    # Lister les fichiers sauvegardés
    print("\nFichiers dans output_dir:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.relative_to(output_dir)} ({size_mb:.1f} MB)")


# ============================================================
# 8. MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Knowledge Distillation — Vision-Language"
    )
    parser.add_argument(
        "--phase", type=str, default="both",
        choices=["1", "2", "both"],
        help="Phase à exécuter : '1' (logits), '2' (training), 'both' (défaut)",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=DATASET_PATH,
        help=f"Chemin vers le dataset BoundingDocs (défaut: {DATASET_PATH})",
    )
    parser.add_argument(
        "--logits-dir", type=str, default=LOGITS_DIR,
        help=f"Répertoire pour les logits cachés (défaut: {LOGITS_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Répertoire de sortie du modèle (défaut: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Nombre max de samples (None = tout le dataset)",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Nombre d'époques (défaut: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (défaut: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (défaut: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help=f"Alpha KL vs CE (défaut: {ALPHA})",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Désactiver la quantification 4-bit du teacher",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Appliquer les arguments CLI aux variables globales
    DATASET_PATH = args.dataset_path
    LOGITS_DIR = args.logits_dir
    OUTPUT_DIR = args.output_dir
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    ALPHA = args.alpha
    LOAD_IN_4BIT = not args.no_4bit

    if args.max_samples is not None:
        MAX_SAMPLES = args.max_samples

    run_phase1_flag = args.phase in ("1", "both")
    run_phase2_flag = args.phase in ("2", "both")

    print("=" * 60)
    print("Offline Knowledge Distillation — Vision-Language")
    print("=" * 60)
    print(f"  Dataset       : {DATASET_PATH}")
    print(f"  Logits dir    : {LOGITS_DIR}")
    print(f"  Output dir    : {OUTPUT_DIR}")
    print(f"  Teacher       : {TEACHER_MODEL}")
    print(f"  Student       : {STUDENT_MODEL}")
    print(f"  Max samples   : {MAX_SAMPLES}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  LR            : {LEARNING_RATE}")
    print(f"  Alpha         : {ALPHA}")
    print(f"  4-bit teacher : {LOAD_IN_4BIT}")
    print(f"  Phase(s)      : {args.phase}")
    print("=" * 60)

    if run_phase1_flag:
        dataset = load_local_dataset(DATASET_PATH, MAX_SAMPLES)
        run_phase1(dataset)
        del dataset
        gc.collect()

    if run_phase2_flag:
        run_phase2()

    print("\nTout est terminé !")