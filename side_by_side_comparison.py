"""
Side-by-side image comparison tool.

Creates side-by-side comparisons of original and cropped images from Florence-2 processing output.
Pairs original_i.png with crop_i_y.png files and saves the combined images to a new folder.
"""

import os
import argparse
from typing import List, Tuple
from PIL import Image


def get_crop_pairs(original_dir: str, crops_dir: str) -> List[Tuple[str, str, int, int]]:
    """
    Match original images with their corresponding crops.

    Returns a list of tuples: (original_path, crop_path, original_idx, crop_idx)
    where original_idx is the index from original_i.png and crop_idx is the y index from crop_i_y.png
    """
    pairs = []

    if not os.path.exists(original_dir) or not os.path.exists(crops_dir):
        print(f"[WARN] Original or crops directory does not exist")
        return pairs

    # Get all original images
    originals = {}
    for fname in os.listdir(original_dir):
        if fname.startswith("original_") and fname.endswith(".png"):
            try:
                idx = int(fname.replace("original_", "").replace(".png", ""))
                originals[idx] = os.path.join(original_dir, fname)
            except ValueError:
                continue

    # Get all crops and pair them with originals
    for fname in os.listdir(crops_dir):
        if fname.startswith("crop_") and fname.endswith(".png"):
            try:
                # Parse crop_i_y.png format
                parts = fname.replace("crop_", "").replace(".png", "").split("_")
                if len(parts) == 2:
                    original_idx = int(parts[0])
                    crop_idx = int(parts[1])

                    if original_idx in originals:
                        crop_path = os.path.join(crops_dir, fname)
                        pairs.append((originals[original_idx], crop_path, original_idx, crop_idx))
            except (ValueError, IndexError):
                continue

    # Sort by original index, then crop index
    pairs.sort(key=lambda x: (x[2], x[3]))

    return pairs


def create_side_by_side(original_path: str, crop_path: str, output_path: str,
                        max_width: int = 1200, gap: int = 10) -> bool:
    """
    Create a side-by-side comparison image of original and crop.

    Args:
        original_path: Path to the original image
        crop_path: Path to the cropped image
        output_path: Path to save the combined image
        max_width: Maximum width for each image (will be resized to fit if needed)
        gap: Pixel gap between the two images

    Returns:
        True if successful, False otherwise
    """
    try:
        original = Image.open(original_path).convert("RGB")
        crop = Image.open(crop_path).convert("RGB")

        # Resize images if they exceed max_width while maintaining aspect ratio
        def resize_if_needed(img, max_w):
            if img.width > max_w:
                ratio = max_w / img.width
                new_height = int(img.height * ratio)
                return img.resize((max_w, new_height), Image.Resampling.LANCZOS)
            return img

        original = resize_if_needed(original, max_width)
        crop = resize_if_needed(crop, max_width)

        # Make heights equal by padding the shorter image
        max_height = max(original.height, crop.height)

        def pad_to_height(img, target_height):
            if img.height < target_height:
                padding = (target_height - img.height) // 2
                padded = Image.new("RGB", (img.width, target_height), color=(255, 255, 255))
                padded.paste(img, (0, padding))
                return padded
            return img

        original = pad_to_height(original, max_height)
        crop = pad_to_height(crop, max_height)

        # Create combined image
        combined_width = original.width + crop.width + gap
        combined = Image.new("RGB", (combined_width, max_height), color=(255, 255, 255))

        # Paste images side by side
        combined.paste(original, (0, 0))
        combined.paste(crop, (original.width + gap, 0))

        # Save the combined image
        combined.save(output_path)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to create side-by-side for {original_path} and {crop_path}: {e}")
        return False


def generate_side_by_side_comparisons(product_id: str, base_dir: str = "images",
                                      max_width: int = 1200, gap: int = 10) -> List[str]:
    """
    Generate side-by-side comparison images for a product.

    Args:
        product_id: The product ID/barcode to process
        base_dir: Base directory containing product folders (default: "images")
        max_width: Maximum width for each image in the comparison
        gap: Pixel gap between the two images

    Returns:
        List of paths to generated comparison images
    """
    product_dir = os.path.join(base_dir, product_id)
    original_dir = os.path.join(product_dir, "original")
    crops_dir = os.path.join(product_dir, "cropped")
    output_dir = os.path.join(product_dir, "side-by-side")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get crop pairs
    pairs = get_crop_pairs(original_dir, crops_dir)

    if not pairs:
        print(f"[INFO] No crop-original pairs found for product {product_id}")
        return []

    print(f"[INFO] Found {len(pairs)} crop-original pairs for product {product_id}")

    saved_paths = []
    for original_path, crop_path, orig_idx, crop_idx in pairs:
        # Create output filename
        output_filename = f"comparison_{orig_idx}_{crop_idx}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Create the side-by-side comparison
        if create_side_by_side(original_path, crop_path, output_path, max_width, gap):
            saved_paths.append(output_path)
            print(f"  ✓ Created: {output_filename}")
        else:
            print(f"  ✗ Failed: {output_filename}")

    print(f"\n[SUCCESS] Saved {len(saved_paths)} comparisons to {output_dir}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side comparisons of original and cropped images"
    )
    parser.add_argument("product_id", help="Product ID/barcode to process")
    parser.add_argument("--base-dir", default="images",
                        help="Base directory containing product folders (default: images)")
    parser.add_argument("--max-width", type=int, default=1200, help="Maximum width for each image (default: 1200)")
    parser.add_argument("--gap", type=int, default=10, help="Pixel gap between images (default: 10)")

    args = parser.parse_args()

    generate_side_by_side_comparisons(
        args.product_id,
        base_dir=args.base_dir,
        max_width=args.max_width,
        gap=args.gap
    )


if __name__ == "__main__":
    main()
