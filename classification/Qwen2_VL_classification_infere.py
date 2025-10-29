import io
import os
import re
import json
import math
import logging
import functools
from argparse import ArgumentParser
from multiprocessing import Pool
import multiprocessing as mp

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.metrics import classification_report, confusion_matrix  # for compute_metrics

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.generation import GenerationConfig

from peft import AutoPeftModelForCausalLM

# Qwen-VL helper: make sure this module is available in your PYTHONPATH
from qwen_vl_utils import process_vision_info


# =========================
# ANSI color codes (optional pretty print)
# =========================
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# =========================
# Logging setup
# =========================
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Reproducibility
torch.manual_seed(1234)


# =========================
# Utility: visualize images quickly (debug only)
# =========================
def plot_images(image_paths):
    """
    Plot one or multiple images side-by-side. For debugging only.
    """
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        ax = axes if num_images == 1 else axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# =========================
# Model paths
# - model_path: RL-finetuned model
# - model_base: original base model for processor/tokenizer, etc.
# =========================
model_path = "./share_models/Qwen2-VL-2B-Instruct_RL/"   # after RL
model_base = "./share_models/Qwen2-VL-2B-Instruct/"      # original Qwen2-VL
ori_processor_path = model_base                           # use base processor


# =========================
# Core worker:
# - Loads model & processor on given rank device
# - Splits validation set by world_size
# - Runs generation and converts each sample to "yes"/"no"
# - Returns local stats and y_pred list
# =========================
def run(rank, world_size):
    # Select device: prefer CUDA per-rank; fallback to CPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Load model on CPU then move to target device to avoid over-allocation
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",  # load weights to CPU first
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(ori_processor_path)

    # ---- Load categories (FGVC Aircraft class names: index 0..N-1) ----
    with open('./val_data/fgvc_aircraft.txt', 'r') as file:
        categories = [line.strip() for line in file.readlines()]
    logger.info(f"[Rank {rank}] Num categories: {len(categories)}")

    # ---- Load validation set mapping: image_path -> label_index ----
    # Expecting a list of dicts like [{"/path/to/img1.jpg": {"label": idx}}, ...]
    pth_file_path = './val_data/fgvc_aircraft.pth'
    predictions = torch.load(pth_file_path)

    val_set = []
    for item in predictions:
        for k, v in item.items():
            # Normalize to {image_path: int_label}
            val_set.append({k: int(v['label'])})

    logger.info(f"[Rank {rank}] Val size (global): {len(val_set)}")

    # ---- Shard the validation set by process rank ----
    split_length = math.ceil(len(val_set) / world_size)
    shard = val_set[int(rank * split_length): int((rank + 1) * split_length)]
    logger.info(f"[Rank {rank}] Shard size: {len(shard)} (chunk={split_length})")

    # ---- Counters and outputs ----
    error_count = 0
    right_count = 0
    y_pred = []  # "yes"/"no" per-sample

    # ---- Inference loop ----
    for sample in tqdm(shard, desc=f"Rank {rank}"):
        # Extract path and gold label
        (image_path, image_label), = sample.items()
        image_cate = categories[image_label]

        # Instruction with required <think> and <answer> tags
        question = (
            "This is an image containing an aircraft. Please identify the model of the aircraft based on the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            "<think> ... </think> <answer>species name</answer>\n"
            "Please strictly follow the format."
        )

        # Chat message: image + text
        query = "<image>\n" + question
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image_path}] + [{"type": "text", "text": query}],
            }
        ]

        # Build inputs (text & vision)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate and post-process
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Extract content inside <answer>...</answer>
            match = re.search(r"<answer>(.*?)</answer>", response)
            answer_content = match.group(1) if match else ""

            # Normalize for simple string inclusion check
            image_cate_norm = image_cate.replace(' ', '').replace('_', '').lower()
            answer_norm = answer_content.replace(' ', '').replace('_', '').lower()

            # If either contains the other, count as correct
            if image_cate_norm in answer_norm or answer_norm in image_cate_norm:
                right_count += 1
                y_pred.append("yes")
            else:
                y_pred.append("no")

        except Exception as e:
            # Any decoding/generation error is treated as "no"
            error_count += 1
            y_pred.append("no")

    # Return local results for aggregation
    return {"error_count": error_count, "right_count": right_count, "y_pred": y_pred}


# =========================
# Metrics helper (binary yes/no report)
# =========================
def compute_metrics(y_true, y_pred):
    """
    Compute a set of useful binary classification metrics for labels {"no","yes"}.
    y_true: list[str], ground-truth labels ("yes"/"no")
    y_pred: list[str], predicted labels ("yes"/"no")
    """
    report = classification_report(
        y_true, y_pred,
        labels=["no", "yes"],
        target_names=["no", "yes"],
        zero_division=0,
        output_dict=True
    )

    # Confusion matrix (TN, FP, FN, TP) in the order of labels=["no", "yes"]
    cm = confusion_matrix(y_true, y_pred, labels=["no", "yes"])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Proportion of predicted "yes" (sanity check for bias)
    yes_proportion = round(y_pred.count("yes") / len(y_pred) * 100, 2) if y_pred else 0.0

    metrics = {
        "yes_precision": round(round(float(report["yes"]["precision"]), 4) * 100, 2),
        "yes_recall":    round(round(float(report["yes"]["recall"]),    4) * 100, 2),
        "yes_f1":        round(round(float(report["yes"]["f1-score"]),  4) * 100, 2),

        "no_precision":  round(round(float(report["no"]["precision"]),  4) * 100, 2),
        "no_recall":     round(round(float(report["no"]["recall"]),     4) * 100, 2),
        "no_f1":         round(round(float(report["no"]["f1-score"]),   4) * 100, 2),

        "macro_precision": round(round(float(report["macro avg"]["precision"]), 4) * 100, 2),
        "macro_recall":    round(round(float(report["macro avg"]["recall"]),    4) * 100, 2),
        "macro_f1":        round(round(float(report["macro avg"]["f1-score"]),  4) * 100, 2),

        "total_accuracy":  round(round(float(report["accuracy"]), 4) * 100, 2),
        "yes_proportion":  yes_proportion,

        "fp": fp, "tp": tp, "fn": fn, "tn": tn
    }

    return metrics


# =========================
# Main: launch multi/single process flow,
# gather predictions, build y_true, compute metrics,
# and save to JSON.
# =========================
def main():
    # Ensure "spawn" to be safe across platforms
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    n_gpus = torch.cuda.device_count()
    logger.info(f"Detected GPUs: {n_gpus}")
    use_multiprocess = n_gpus >= 2

    all_y_pred = []
    total_error = 0
    total_right = 0

    if use_multiprocess:
        logger.info('Started generation (multiprocess)')
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        for res in result_lists:
            total_error += int(res["error_count"])
            total_right += int(res["right_count"])
            all_y_pred.extend(res["y_pred"])
    else:
        logger.info("Running on a single GPU or CPU")
        # Treat as world_size=1, rank=0
        res = run(rank=0, world_size=1)
        total_error += int(res["error_count"])
        total_right += int(res["right_count"])
        all_y_pred.extend(res["y_pred"])

    # Ground truth is "yes" for each sample because we evaluate
    # "did the model correctly name the aircraft class?" per item.
    y_true = ["yes"] * len(all_y_pred)

    # Compute metrics
    metrics = compute_metrics(y_true, all_y_pred)

    # Log summary + metrics
    logger.info(f"Error number: {total_error}")
    logger.info(f"Total Right Number: {total_right}")
    logger.info("Metrics:\n" + json.dumps(metrics, ensure_ascii=False, indent=2))

    # Optionally save metrics to disk
    with open("metrics_fgvc_aircraft.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Saved metrics to metrics_fgvc_aircraft.json")


if __name__ == "__main__":
    main()
