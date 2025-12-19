import os
import argparse
from typing import List
import json
import math

import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForCausalLM

from off_api import get_images_for_product, get_product_data


def setup_model(model_id: str = "microsoft/Florence-2-large"):
    """
    Load Florence-2 model and processor, pick best available device (MPS on macOS, else CPU).
    Returns (model, processor, device)
    """
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = (
        AutoModelForCausalLM
        .from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
        .eval()
        .to(device)
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device


def run_phrase_grounding(model, processor, device, image: Image.Image, text_input: str):
    """
    Run Florence-2 '<CAPTION_TO_PHRASE_GROUNDING>' on a single image with a text input.
    Returns the parsed answer dict { '<CAPTION_TO_PHRASE_GROUNDING>': { 'bboxes': [...], 'labels': [...] } }
    """
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = text_input or ""
    # Florence expects the text immediately after the task token; adding a leading space keeps readability.
    prompt = f"{task_prompt} {text_input}" if text_input else task_prompt

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    # Move tensors to the appropriate device and dtype
    # On macOS MPS, float16 is typical; on CPU float32 is safer.
    if device.type == "mps":
        inputs = inputs.to(device, torch.float16)
        pixel_values = inputs["pixel_values"].to(device)
    else:
        inputs = inputs.to(device)
        pixel_values = inputs["pixel_values"].to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=pixel_values,
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    return parsed_answer


def extract_and_save_crops(image: Image.Image, results: dict, task_prompt: str, out_dir: str, image_index: int) -> List[
    str]:
    """
    Extract cropped images from results and save to out_dir. Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    saved_paths: List[str] = []
    if task_prompt in results:
        task_results = results[task_prompt]
        bboxes = task_results.get("bboxes", [])
        labels = task_results.get("labels", [])

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            # Use floor for start coords and ceil for end coords to avoid collapsing very small boxes
            x1 = int(math.floor(x1))
            y1 = int(math.floor(y1))
            x2 = int(math.ceil(x2))
            y2 = int(math.ceil(y2))

            # Clamp to image bounds
            x1 = max(0, min(x1, image.width - 1))
            y1 = max(0, min(y1, image.height - 1))
            x2 = max(0, min(x2, image.width))
            y2 = max(0, min(y2, image.height))

            # Skip invalid or collapsed boxes
            if x2 <= x1 or y2 <= y1:
                print(f"[INFO] Skipping invalid box {bbox} after clamping -> ({x1},{y1},{x2},{y2})")
                continue

            cropped = image.crop((x1, y1, x2, y2))

            raw_label = labels[i] if i < len(labels) else f"crop_{i}"
            safe_label = (
                raw_label.replace(" ", "_")
                .replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-")
            )
            filename = f"{safe_label}_{image_index}_{i}.png"
            out_path = os.path.join(out_dir, filename)
            try:
                cropped.save(out_path)
                saved_paths.append(out_path)
            except Exception:
                out_path = os.path.join(out_dir, f"crop_{image_index}_{i}.png")
                cropped.save(out_path)
                saved_paths.append(out_path)

    return saved_paths


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    return Image.open(resp.raw).convert("RGB")


def process_product(product_id: str, model, processor, device, text_override: str | None = None) -> list[str]:
    """
    For a product_id, run phrase grounding across all image URLs, saving crops under images/{product_id}/.
    Also saves the original images for reference.
    Returns list of saved crop paths.
    """
    urls = get_images_for_product(product_id)
    product_data = get_product_data(product_id)
    product_title = text_override if text_override is not None else product_data.get("title", "") or ""

    base_out_dir = os.path.join("images", product_id)
    originals_dir = os.path.join(base_out_dir, "original")
    crops_dir = os.path.join(base_out_dir, "cropped")
    florence_dir = os.path.join(base_out_dir, "florence_output")
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(florence_dir, exist_ok=True)

    print(f"Processing product {product_id} with title '{product_title}'")
    print(f"Found {len(urls)} image URL(s)")

    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    all_saved: list[str] = []

    for idx, url in enumerate(urls):
        try:
            image = load_image_from_url(url)
        except Exception as e:
            print(f"[WARN] Failed to load URL #{idx} for {product_id}: {e}")
            continue

        # Save original image for reference
        orig_path = os.path.join(originals_dir, f"original_{idx}.png")
        try:
            image.save(orig_path)
        except Exception as e:
            print(f"[WARN] Failed to save original image #{idx}: {e}")

        # Run model
        try:
            results = run_phrase_grounding(model, processor, device, image, text_input=product_title)
        except Exception as e:
            print(f"[WARN] Inference failed for image #{idx}: {e}")
            continue

        # Persist raw results for debugging
        raw_path = os.path.join(florence_dir, f"results_{idx}.json")
        try:
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save raw results #{idx}: {e}")

        # Save crops
        saved = extract_and_save_crops(image, results, task_prompt=task_prompt, out_dir=crops_dir, image_index=idx)
        all_saved.extend(saved)
        bboxes = results.get(task_prompt, {}).get("bboxes", [])
        if not bboxes:
            print(f"Product {product_id} - Image #{idx}: no boxes returned, saved {len(saved)} crops to {crops_dir}")
        else:
            print(f"Product {product_id} - Image #{idx}: {len(bboxes)} boxes, saved {len(saved)} crops to {crops_dir}")

    if not urls:
        print(f"[INFO] No image URLs found for product {product_id}")

    return all_saved


def parse_args():
    p = argparse.ArgumentParser(description="Batch Open Food Facts phrase grounding with Florence-2")
    p.add_argument("product_id", help="Open Food Facts product id to process")
    p.add_argument("--model-id", default="microsoft/Florence-2-large", help="Hugging Face model id")
    p.add_argument("--text-prompt", default=None, help="Optional custom text to ground (defaults to product title)")
    return p.parse_args()


def run_batch_off_inference(product_id: str, model_id: str = "microsoft/Florence-2-large",
                            text_prompt: str | None = None) -> list[str]:
    """
    Public callable function to run phrase grounding for all images of a product.
    This sets up the model and processor and then calls process_product.

    Usage:
        from batch_off_inference import run_batch_off_inference
        run_batch_off_inference("0012000130311", text_prompt="Mountain Dew Baja Blast Tropical Lime")
    Returns the list of saved crop paths.
    """
    print(f"Loading model {model_id}...")
    model, processor, device = setup_model(model_id)
    print(f"Device: {device}")
    saved = process_product(product_id, model, processor, device, text_override=text_prompt)
    print("Done.")
    return saved


def main():
    args = parse_args()
    print(f"Loading model {args.model_id}...")
    model, processor, device = setup_model(args.model_id)
    print(f"Device: {device}")

    process_product(args.product_id, model, processor, device, text_override=args.text_prompt)
    print("Done.")


if __name__ == "__main__":
    run_batch_off_inference("0012000130311")
