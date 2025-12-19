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


def detect_nutrition_facts(model, processor, device, image: Image.Image):
    """
    Run Florence-2 object detection to find nutrition facts panels in an image.
    Uses <OPEN_VOCABULARY_DETECTION> task to detect "nutrition facts" or "nutrition information".
    Returns a dict with detection results and confidence score.
    """
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_input = "nutrition facts panel"
    prompt = f"{task_prompt} {text_input}"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
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


def extract_nutrition_panels(image: Image.Image, nutrition_results: dict, task_prompt: str, out_dir: str,
                             image_index: int) -> List[str]:
    """
    Extract nutrition facts panels from detection results and save to out_dir.
    Only extracts the largest nutrition panel (by area) if multiple panels are detected.
    Returns list with at most one saved nutrition panel path, or empty list if no panels found.
    """
    saved_paths: List[str] = []

    if task_prompt not in nutrition_results:
        return saved_paths

    task_results = nutrition_results[task_prompt]
    bboxes = task_results.get("bboxes", [])
    labels = task_results.get("labels", [])

    if not bboxes:
        # No panels detected, return empty list
        return saved_paths

    # Filter valid bboxes and calculate their areas
    valid_boxes = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
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
            print(f"[INFO] Skipping invalid nutrition panel box {bbox} after clamping -> ({x1},{y1},{x2},{y2})")
            continue

        # Calculate area
        area = (x2 - x1) * (y2 - y1)
        valid_boxes.append((i, x1, y1, x2, y2, area))

    if not valid_boxes:
        return saved_paths

    # Select only the largest panel by area
    largest = max(valid_boxes, key=lambda x: x[5])
    i, x1, y1, x2, y2, area = largest

    # Only create directory if we have a valid panel to save
    os.makedirs(out_dir, exist_ok=True)

    cropped = image.crop((x1, y1, x2, y2))

    raw_label = labels[i] if i < len(labels) else "nutrition_panel"
    safe_label = (
        raw_label.replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )
    filename = f"{safe_label}_{image_index}.png"
    out_path = os.path.join(out_dir, filename)
    try:
        cropped.save(out_path)
        saved_paths.append(out_path)
    except Exception:
        out_path = os.path.join(out_dir, f"nutrition_panel_{image_index}.png")
        cropped.save(out_path)
        saved_paths.append(out_path)

    return saved_paths


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    _, x1_1, y1_1, x2_1, y2_1, _ = box1
    _, x1_2, y1_2, x2_2, y2_2, _ = box2

    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_filtered_boxes(image: Image.Image, results: dict, task_prompt: str, iou_threshold: float = 0.5):
    """
    Extract and filter bounding boxes from results using IoU threshold.
    Returns list of filtered boxes in format: [(original_index, x1, y1, x2, y2, area), ...]
    Only keeps the largest box when boxes are too similar (IoU > threshold).
    """
    if task_prompt not in results:
        return []

    task_results = results[task_prompt]
    bboxes = task_results.get("bboxes", [])

    # Normalize and validate boxes
    valid_boxes = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
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

        area = (x2 - x1) * (y2 - y1)
        valid_boxes.append((i, x1, y1, x2, y2, area))

    if not valid_boxes:
        return []

    # Filter similar boxes using IoU threshold - keep only the largest
    kept_boxes = []
    used_indices = set()

    # Sort by area descending to process largest boxes first
    sorted_boxes = sorted(valid_boxes, key=lambda x: x[5], reverse=True)

    for box in sorted_boxes:
        idx = box[0]
        if idx in used_indices:
            continue

        # Check if this box is similar to any already kept box
        is_similar = False
        for kept_box in kept_boxes:
            iou = calculate_iou(box, kept_box)
            if iou > iou_threshold:
                is_similar = True
                break

        if not is_similar:
            kept_boxes.append(box)
            used_indices.add(idx)

    print(f"[INFO] Filtered {len(valid_boxes)} boxes to {len(kept_boxes)} unique boxes (removed duplicates)")
    return kept_boxes


def extract_and_save_crops(image: Image.Image, results: dict, task_prompt: str, out_dir: str, image_index: int) -> List[
    str]:
    """
    Extract cropped images from results and save to out_dir.
    Filters out similar boxes (using IoU threshold) and keeps only the largest box.
    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    saved_paths: List[str] = []
    if task_prompt not in results:
        return saved_paths

    task_results = results[task_prompt]
    labels = task_results.get("labels", [])

    # Get filtered boxes using the shared function
    kept_boxes = get_filtered_boxes(image, results, task_prompt)

    if not kept_boxes:
        return saved_paths

    # Save crops from filtered boxes
    for box in kept_boxes:
        i, x1, y1, x2, y2, area = box
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


def process_product(product_id: str, model, processor, device, text_override: str | None = None) -> dict:
    """
    For a product_id, run phrase grounding across all image URLs, saving crops under images/{product_id}/.
    Also detects and saves nutrition facts panels to a separate folder.
    Returns a dict with 'crops' and 'nutrition_panels' lists.
    """
    urls = get_images_for_product(product_id)
    product_data = get_product_data(product_id)
    product_title = text_override if text_override is not None else product_data.get("title", "") or ""

    base_out_dir = os.path.join("images", product_id)
    originals_dir = os.path.join(base_out_dir, "original")
    crops_dir = os.path.join(base_out_dir, "cropped")
    nutrition_dir = os.path.join(base_out_dir, "nutrition_panels")
    florence_dir = os.path.join(base_out_dir, "florence_output")
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(nutrition_dir, exist_ok=True)
    os.makedirs(florence_dir, exist_ok=True)

    print(f"Processing product {product_id} with title '{product_title}'")
    print(f"Found {len(urls)} image URL(s)")

    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    nutrition_task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    all_saved_crops: list[str] = []
    all_saved_panels: list[str] = []

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

        # Run phrase grounding for product name
        try:
            results = run_phrase_grounding(model, processor, device, image, text_input=product_title)
        except Exception as e:
            print(f"[WARN] Phrase grounding inference failed for image #{idx}: {e}")
            continue

        # Persist raw results for debugging
        raw_path = os.path.join(florence_dir, f"results_{idx}.json")
        try:
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save raw results #{idx}: {e}")

        # Save crops from phrase grounding
        saved = extract_and_save_crops(image, results, task_prompt=task_prompt, out_dir=crops_dir, image_index=idx)
        all_saved_crops.extend(saved)

        # Get the filtered boxes for nutrition detection (same IoU filtering)
        filtered_boxes = get_filtered_boxes(image, results, task_prompt)

        bboxes = results.get(task_prompt, {}).get("bboxes", [])
        if not bboxes:
            print(
                f"Product {product_id} - Image #{idx}: no product boxes returned, saved {len(saved)} crops to {crops_dir}")
        else:
            print(
                f"Product {product_id} - Image #{idx}: {len(bboxes)} product boxes, saved {len(saved)} crops to {crops_dir}")

            # Run nutrition facts detection only on the filtered unique crops
            for crop_idx, (_, x1, y1, x2, y2, _) in enumerate(filtered_boxes):
                cropped_image = image.crop((x1, y1, x2, y2))

                try:
                    nutrition_results = detect_nutrition_facts(model, processor, device, cropped_image)
                    # Pass idx as base and crop_idx as part of filename generation
                    nutrition_panels = extract_nutrition_panels(cropped_image, nutrition_results, nutrition_task_prompt,
                                                                nutrition_dir, idx * 1000 + crop_idx)
                    if nutrition_panels:
                        print(f"  └─ Crop #{crop_idx}: detected {len(nutrition_panels)} nutrition panel(s)")
                        all_saved_panels.extend(nutrition_panels)
                except Exception as e:
                    print(f"[WARN] Nutrition facts detection failed for image #{idx}, crop #{crop_idx}: {e}")

    if not urls:
        print(f"[INFO] No image URLs found for product {product_id}")

    return {
        "crops": all_saved_crops,
        "nutrition_panels": all_saved_panels
    }


def parse_args():
    p = argparse.ArgumentParser(description="Batch Open Food Facts phrase grounding with Florence-2")
    p.add_argument("product_id", help="Open Food Facts product id to process")
    p.add_argument("--model-id", default="microsoft/Florence-2-large", help="Hugging Face model id")
    p.add_argument("--text-prompt", default=None, help="Optional custom text to ground (defaults to product title)")
    return p.parse_args()


def run_batch_off_inference(product_id: str, model_id: str = "microsoft/Florence-2-large",
                            text_prompt: str | None = None) -> dict:
    """
    Public callable function to run phrase grounding and nutrition facts detection for all images of a product.
    This sets up the model and processor and then calls process_product.

    Usage:
        from batch_off_inference import run_batch_off_inference
        result = run_batch_off_inference("0012000130311", text_prompt="Mountain Dew Baja Blast Tropical Lime")
        print(result['crops'])  # List of cropped product images
        print(result['nutrition_panels'])  # List of detected nutrition fact panels

    Returns a dict with 'crops' and 'nutrition_panels' lists.
    """
    print(f"Loading model {model_id}...")
    model, processor, device = setup_model(model_id)
    print(f"Device: {device}")
    result = process_product(product_id, model, processor, device, text_override=text_prompt)
    print("Done.")
    return result


def main():
    args = parse_args()
    print(f"Loading model {args.model_id}...")
    model, processor, device = setup_model(args.model_id)
    print(f"Device: {device}")

    process_product(args.product_id, model, processor, device, text_override=args.text_prompt)
    print("Done.")


if __name__ == "__main__":
    run_batch_off_inference("0012000130311")
    # run_batch_off_inference("0027379002336")
