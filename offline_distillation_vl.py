"""
Offline Knowledge Distillation: Vision-Language variant 
============================================================
the student sees images by re-processing them from the dataset at
training time (no pixel data cached on disk).
"""

import argparse
import json
import os
import numpy as np
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
from datasets import load_dataset
from tqdm import tqdm

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Please install qwen_vl_utils: pip install qwen-vl-utils")
    process_vision_info = None


TOP_K_LOGITS = 64
MAX_SEQ_LEN = 1024

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


def format_boundingdocs_sample(sample: dict) -> Optional[dict]:
    try:
        qa_data = (
            json.loads(sample["Q&A"])
            if isinstance(sample["Q&A"], str)
            else sample["Q&A"]
        )
    except (json.JSONDecodeError, KeyError):
        return None

    if not qa_data or not isinstance(qa_data, list):
        return None

    qa = qa_data[0]
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

    answer_json = json.dumps({"value": answer_value})

    return {
        "messages": messages,
        "answer": answer_json,
        "answer_raw": answer_value,
    }


# ===========================================================================
# Phase 1: Generate teacher logits — saves dataset_index, no pixel data
# ===========================================================================

def generate_teacher_logits_vl(args):
    print("=" * 60)
    print("PHASE 1 (VL): Generating teacher logits")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading processor from Qwen/Qwen2-VL-7B-Instruct")
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_fast=True,
        trust_remote_code=True,
    )

    print(f"Loading teacher: {args.teacher_model}")
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        teacher = AutoModelForImageTextToText.from_pretrained(
            args.teacher_model,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        teacher = AutoModelForImageTextToText.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    teacher.eval()
    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B params")

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    logits_dir = Path(args.logits_dir)
    logits_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(logits_dir / "processor")

    metadata = {
        "teacher_model": args.teacher_model,
        "dataset": args.dataset,
        "top_k": TOP_K_LOGITS,
        "max_seq_len": MAX_SEQ_LEN,
        "num_samples": 0,
        "model_type": "vision_language",
        "system_message": SYSTEM_MESSAGE,
    }

    sample_idx = 0
    skipped = 0

    for i, sample in enumerate(tqdm(dataset, desc="Generating VL logits")):
        formatted = format_boundingdocs_sample(sample)
        if formatted is None:
            skipped += 1
            continue

        try:
            messages = formatted["messages"]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs, video_inputs = None, None

            inputs = processor(
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

            # >>> Save dataset_index so Phase 2 can retrieve the image
            np.savez_compressed(
                logits_dir / f"sample_{sample_idx:06d}.npz",
                input_ids=inputs["input_ids"][0].cpu().numpy().astype(np.int32),
                top_k_values=top_k_vals.cpu().half().numpy(),
                top_k_indices=top_k_idx.cpu().numpy().astype(np.int32),
                attention_mask=inputs.get(
                    "attention_mask",
                    torch.ones_like(inputs["input_ids"]),
                )[0].cpu().numpy().astype(np.int8),
                dataset_index=np.array([i], dtype=np.int64),  # <<< NEW
                answer=np.array([formatted["answer_raw"]], dtype=object),
            )
            sample_idx += 1

        except Exception as e:
            if i < 10:
                print(f"  Warning: skipping sample {i}: {e}")
            skipped += 1
            continue

        if i % 100 == 0:
            torch.cuda.empty_cache()

    metadata["num_samples"] = sample_idx
    metadata["skipped"] = skipped
    with open(logits_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! {sample_idx} samples saved, {skipped} skipped -> {logits_dir}")


# ===========================================================================
# Phase 2: Dataset that re-processes images from HF dataset
# ===========================================================================

class VLDistillationDataset(Dataset):
    """
    Loads cached teacher logits and re-processes images from the original
    HuggingFace dataset using the student's processor.
    """

    def __init__(
        self,
        logits_dir: str,
        hf_dataset,
        student_processor: AutoProcessor,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.logits_dir = Path(logits_dir)
        self.hf_dataset = hf_dataset
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

        input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
        top_k_values = torch.tensor(data["top_k_values"].astype(np.float32))
        top_k_indices = torch.tensor(data["top_k_indices"], dtype=torch.long)
        attention_mask = torch.tensor(data["attention_mask"], dtype=torch.long)

        L = min(len(input_ids), self.max_seq_len)
        input_ids = input_ids[:L]
        top_k_values = top_k_values[:L]
        top_k_indices = top_k_indices[:L]
        attention_mask = attention_mask[:L]

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        # ---------------------------------------------------------------
        # Re-process the image from the original dataset
        # ---------------------------------------------------------------
        dataset_index = int(data["dataset_index"][0])
        hf_sample = self.hf_dataset[dataset_index]
        formatted = format_boundingdocs_sample(hf_sample)

        pixel_values = None
        image_grid_thw = None

        if formatted is not None:
            messages = formatted["messages"]
            if process_vision_info is not None:
                image_inputs, _ = process_vision_info(messages)
            else:
                image_inputs = None

            if image_inputs is not None:
                # We only need the vision part from the processor.
                # Use a dummy text to avoid re-tokenizing (we already have
                # input_ids from the teacher).
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                vision_inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=None,
                    padding=False,
                    return_tensors="pt",
                )
                if "pixel_values" in vision_inputs:
                    # Remove batch dim: (1, num_patches, D) -> (num_patches, D)
                    pixel_values = vision_inputs["pixel_values"].squeeze(0)
                if "image_grid_thw" in vision_inputs:
                    image_grid_thw = vision_inputs["image_grid_thw"].squeeze(0)

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "teacher_top_k_values": top_k_values,
            "teacher_top_k_indices": top_k_indices,
        }
        if pixel_values is not None:
            item["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            item["image_grid_thw"] = image_grid_thw

        return item


def vl_collate_fn(batch):
    """Collate with dynamic padding + concatenation for vision tensors."""
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

    # Qwen2-VL expects pixel_values as (total_patches, D) concatenated
    # across the batch, and image_grid_thw as (total_images, 3).
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
    """Combined KL divergence + Cross-entropy distillation loss."""
    s_log = student_logits[:, :-1, :].contiguous()
    s_lab = labels[:, 1:].contiguous()
    s_mask = mask[:, 1:].contiguous().float()
    t_v = t_vals[:, :-1, :].contiguous()
    t_i = t_idx[:, :-1, :].contiguous()
    bs, seq, _ = s_log.shape

    ce = F.cross_entropy(
        s_log.view(-1, vocab_size), s_lab.view(-1),
        ignore_index=-100, reduction="none",
    ).view(bs, seq)
    ce_loss = (ce * s_mask).sum() / s_mask.sum().clamp(min=1)

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


# ===========================================================================
# Phase 2: Training
# ===========================================================================

def train_distillation_vl(args):
    print("=" * 60)
    print("PHASE 2 (VL): Distillation training (images from dataset)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load metadata to know which dataset to load ---
    with open(Path(args.logits_dir) / "metadata.json") as f:
        meta = json.load(f)

    # --- Load student ---
    print(f"\nLoading student: {args.student_model}")
    student_processor = AutoProcessor.from_pretrained(
        args.student_model, trust_remote_code=True,
    )
    student = AutoModelForImageTextToText.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    student.train()

    n_params = sum(p.numel() for p in student.parameters()) / 1e9
    print(f"Student: {n_params:.2f}B parameters")

    # --- Load the original HF dataset for image access ---
    ds_name = args.dataset or meta["dataset"]
    print(f"Loading HF dataset for images: {ds_name}")
    hf_dataset = load_dataset(ds_name, split="train")

    # --- Build distillation dataset ---
    dataset = VLDistillationDataset(
        args.logits_dir, hf_dataset, student_processor, MAX_SEQ_LEN,
    )
    val_size = max(1, int(0.05 * len(dataset)))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size],
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=vl_collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=vl_collate_fn, num_workers=2, pin_memory=True,
    )

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.05 * total_steps), total_steps,
    )

    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    scaler = torch.amp.GradScaler("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        student.train()
        ep_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            pixel_values = batch.pop("pixel_values", None)
            image_grid_thw = batch.pop("image_grid_thw", None)

            batch = {k: v.to(device) for k, v in batch.items()}
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            with torch.amp.autocast("cuda"):
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
                    student.config.vocab_size,
                    args.temperature,
                    args.alpha,
                )
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            ep_loss += loss.item() * args.gradient_accumulation_steps
            n_batch += 1
            pbar.set_postfix(
                loss=f"{ep_loss / n_batch:.4f}",
                kl=f"{kl:.4f}",
                ce=f"{ce:.4f}",
            )

            if global_step > 0 and global_step % args.save_every == 0:
                ckpt = output_dir / f"checkpoint-{global_step}"
                student.save_pretrained(ckpt)
                student_processor.save_pretrained(ckpt)

        # --- Validation ---
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

                with torch.amp.autocast("cuda"):
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
                        student.config.vocab_size,
                        args.temperature,
                        args.alpha,
                    )
                val_total += loss.item()
                val_n += 1

        avg_val = val_total / max(val_n, 1)
        print(f"\n  Epoch {epoch + 1}: train={ep_loss / n_batch:.4f} | val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            student.save_pretrained(output_dir / "best")
            student_processor.save_pretrained(output_dir / "best")
            print("  -> New best model!")

    student.save_pretrained(output_dir / "final")
    student_processor.save_pretrained(output_dir / "final")
    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Offline Distillation (VL)")
    parser.add_argument(
        "--phase", required=True, choices=["generate_logits", "distill"],
    )

    parser.add_argument(
        "--teacher_model", default="letxbe/qwen2-7b-BoundingDocs-rephrased",
    )
    parser.add_argument("--dataset", default="letxbe/BoundingDocs")
    parser.add_argument("--logits_dir", default="./cached_logits_vl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--load_in_4bit", action="store_true",
        help="Load teacher in 4-bit quantization (saves VRAM)",
    )

    parser.add_argument(
        "--student_model", default="Qwen/Qwen2-VL-2B-Instruct",
    )
    parser.add_argument("--output_dir", default="./distilled_docexplainer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--save_every", type=int, default=500)

    args = parser.parse_args()

    if args.phase == "generate_logits":
        generate_teacher_logits_vl(args)
    else:
        train_distillation_vl(args)


if __name__ == "__main__":
    main()

#Ligne rajoutée de test