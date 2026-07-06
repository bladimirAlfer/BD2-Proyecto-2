import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from src.config import PMC_IMAGES_CSV

INPUT = PMC_IMAGES_CSV
OUT = PMC_IMAGES_CSV.parent
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

sift = cv2.SIFT_create()

all_descriptors = []
image_descriptor_map = {}

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extrayendo SIFT"):
    image_id = row["image_id"]
    image_path = row["image_path"]

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (512, 512))

    keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        continue

    image_descriptor_map[image_id] = descriptors.astype("float32")
    all_descriptors.append(descriptors.astype("float32"))

if all_descriptors:
    all_descriptors = np.vstack(all_descriptors)
else:
    all_descriptors = np.empty((0, 128), dtype="float32")

np.save(OUT / "pmc_sift_descriptors.npy", all_descriptors)

with open(OUT / "pmc_image_descriptor_map.pkl", "wb") as f:
    pickle.dump(image_descriptor_map, f)

print("Total descriptores SIFT:", all_descriptors.shape)
print("Imágenes con SIFT:", len(image_descriptor_map))
