# %%
import time
import requests
import os
import zipfile


def zip_images(image_paths, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img in image_paths:
            zipf.write(img, os.path.basename(img))


def _split_barcode_for_path(barcode: str) -> str:
    """
    Splits a 13-digit barcode into the Open Food Facts S3 path format.
    For barcodes with fewer than 13 digits, it returns the barcode as is.

    Example (13 digits): "4012359114303" -> "401/235/911/4303"
    Example (<13 digits): "20065034" -> "20065034"
    """
    if len(barcode) == 13:
        # Split into three groups of 3 digits followed by one group of 4 digits
        return f"{barcode[0:3]}/{barcode[3:6]}/{barcode[6:9]}/{barcode[9:13]}"
    else:
        # For shorter barcodes, the path is just the barcode itself
        return barcode


def get_product_data(product_id: str) -> dict or None:
    """
    Fetches product information from the Open Food Facts API, including image data.

    Args:
        product_id (str): The barcode or product ID of the product.

    Returns:
        dict or None: A dictionary containing product data if found, otherwise None.
    """

    product_id = product_id.zfill(13)

    api_url = f"https://world.openfoodfacts.org/api/v2/product/{product_id}.json"
    headers = {
        "User-Agent": "OpenFoodFactsImageDownloader/1.0"
    }
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get('status') == 1 and data.get('product'):
            product = data['product']
            result = {
                "upc": product.get("code"),
                "title": product.get("product_name"),
                "brand": (product.get("brands") or "").split(",")[0].strip() if product.get("brands") else None,
                "ingredients": product.get("ingredients_text"),
                "data": data['product']
            }
            return result
        else:
            print(f"Product not found or status not 1 for ID: {product_id}. "
                  f"API response status: {data.get('status_verbose', 'N/A')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching product data for {product_id}: {e}")
        time.sleep(1)  # Sleep for a second to avoid overwhelming the API
        return None


def construct_image_url(product_id: str, image_id: str, size: str = 'full') -> str:
    """
    Constructs the direct AWS S3 URL for a specific image of a product.

    Args:
        product_id (str): The barcode or product ID of the product.
        image_id (str): The ID of the image (e.g., '1', 'front', 'nutrition').
        size (str): The desired image size ('full' for raw, '400' for 400px width).

    Returns:
        str: The full URL to the image.
    """
    base_s3_url = "https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/data/"
    product_path = _split_barcode_for_path(product_id)

    if size == 'full':
        return f"{base_s3_url}{product_path}/{image_id}.jpg"
    elif size == '400':
        return f"{base_s3_url}{product_path}/{image_id}.400.jpg"
    else:
        # Default to full size if an unsupported size is requested
        print(f"Warning: Unsupported image size '{size}'. Defaulting to 'full'.")
        return f"{base_s3_url}{product_path}/{image_id}.jpg"


def get_images_for_product(
        product_id: str,
        image_sizes: list = ['full'],
) -> list:
    """
    Retrieves all available image URLs for a given Open Food Facts product ID.

    Args:
        product_id (str): The barcode or product ID of the product.
        image_sizes (list): A list of desired image sizes.
                            Options are 'full' (raw) and '400' (400px width).
                            Defaults to ['full'].

    Returns:
        list: A list of image URLs for the product.
    """
    image_urls = []
    all_data = get_product_data(product_id)

    if not all_data:
        print(f"Could not retrieve data for product ID: {product_id}")
        return image_urls

    product_data = all_data.get('data') if all_data else None

    if not product_data:
        print(f"Could not retrieve data for product ID: {product_id}")
        return image_urls

    # Extract image IDs from the product data
    # The 'images' field contains numeric IDs (e.g., '1', '2') and
    # sometimes specific type IDs like 'front', 'nutrition' which
    # might map to numeric IDs in 'selected_images' or be separate.
    # The 'selected_images' field gives specific image types and their URLs/IDs.

    # We will primarily look for image IDs in the 'images' and 'selected_images' fields.
    all_image_ids = set()

    # Add numeric image IDs from the 'images' dictionary
    if 'images' in product_data and isinstance(product_data['images'], dict):
        for img_id_key in product_data['images']:
            # Check if the key is a string representing a number
            if isinstance(img_id_key, str) and img_id_key.isdigit():
                all_image_ids.add(img_id_key)
            else:
                imgid = product_data['images'][img_id_key]["imgid"]
                all_image_ids.add(str(imgid))

    # Add image IDs from 'selected_images' which often contain 'imgid'
    if 'selected_images' in product_data and isinstance(product_data['selected_images'], dict):
        for image_type, image_info in product_data['selected_images'].items():
            if isinstance(image_info, dict):
                # Look for sizes dictionary, which might contain 'imgid'
                for size_key, size_info in image_info.items():
                    if isinstance(size_info, dict) and 'imgid' in size_info:
                        all_image_ids.add(str(size_info['imgid']))

    # Construct URLs for each image ID and size
    if not all_image_ids:
        print(f"No image IDs found for product ID: {product_id}")
        return image_urls

    print(f"Found image IDs: {', '.join(sorted(list(all_image_ids)))}")

    for img_id in all_image_ids:
        for size in image_sizes:
            image_url = construct_image_url(product_id, img_id, size)
            image_urls.append(image_url)

    return image_urls


if __name__ == "__main__":
    test_product_id = "0012000130311"  # Example product ID
    image_urls = get_images_for_product(test_product_id)
    print(f"Image URLs: {image_urls}")

    product_data = get_product_data(test_product_id)
    print(f"Product Data: {product_data}")
