import os
import urllib.request
import multiprocessing
from pathlib import Path
from tqdm import tqdm


def download_image(args):
    image_link, image_id, savefolder = args
    if isinstance(image_link, str):
        filename = f"{image_id}{Path(image_link).suffix}"
        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f"⚠️ Failed: {image_link}\n{ex}")
    return image_id  # optional return for progress tracking


def download_images(image_links, download_folder, sample_ids):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    args = [(link, sid, download_folder) for link, sid in zip(image_links, sample_ids)]

    with multiprocessing.Pool(100) as pool:
        for _ in tqdm(pool.imap_unordered(download_image, args), total=len(args)):
            pass  # tqdm updates each time a download finishes
