"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
import numpy as np
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset

URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}

def _download_and_extract(url:str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.join(dest_dir, url.split("/")[-1])
    if not os.path.exists(filename):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, filename)
        print("  Done.")
    print(f"Extracting {filename} ...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(dest_dir)
    print("  Done.")

 
def download_oxford_pet(root: str) -> None:
    """Download and extract images + annotations if not already present."""
    images_dir = os.path.join(root, "images")
    ann_dir    = os.path.join(root, "annotations")
    if not os.path.isdir(images_dir):
        _download_and_extract(URLS["images"], root)
    if not os.path.isdir(ann_dir):
        _download_and_extract(URLS["annotations"], root)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
 
def _to_tensor_normalised(pil_image: Image.Image, size: int) -> torch.Tensor:
    """Resize PIL RGB image to (size x size), normalise, return [3, H, W] float tensor."""
    img = pil_image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, 3], range [0, 1]
    arr = (arr - _MEAN) / _STD                       # ImageNet normalisation
    return torch.from_numpy(arr.transpose(2, 0, 1))  # [3, H, W]
 
 
def _mask_to_tensor(pil_mask: Image.Image, size: int) -> torch.Tensor:
    """Resize trimap mask (NEAREST) and remap values 1/2/3 -> 0/1/2."""
    mask = pil_mask.resize((size, size), Image.NEAREST)
    arr  = np.array(mask, dtype=np.int64) - 1        # 1->0, 2->1, 3->2
    arr  = np.clip(arr, 0, 2)                        # safety clamp
    return torch.from_numpy(arr)                     # [H, W], values in {0,1,2}
 
class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    IMAGE_SIZE = 224
    
    def __init__(self, root: str, split: str = "trainval", download: bool = True, augment: bool = False):

        super().__init__()
        if split not in("trainval", "test"):
            raise ValueError(f"Invalid split '{split}'. Must be 'trainval' or 'test'.")
        
        self.root = root
        self.split = split
        self.augment = augment

        if download:
            download_oxford_pet(root)

        self._img_dir = os.path.join(root, "images")
        self._mask_dir = os.path.join(root, "annotations", "trimaps")
        self._xml_dir  = os.path.join(root, "annotations", "xmls")
        split_file = os.path.join(root, "annotations", f"{split}.txt")

        self._samples: list[tuple[str, int]] = []

        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                stem = parts[0]
                class_id = int(parts[1]) - 1  # remap 1-37 -> 0-36
                self._samples.append((stem, class_id))

        self._bbox_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem, label = self._samples[idx]

        img_path = os.path.join(self._img_dir, f"{stem}.jpg")
        pil_img = Image.open(img_path).convert("RGB")
        orig_w , orig_h = pil_img.size

        image = _to_tensor_normalised(pil_img, self.IMAGE_SIZE)

        mask_path = os.path.join(self._mask_dir, f"{stem}.png")
        pil_mask = Image.open(mask_path)
        mask = _mask_to_tensor(pil_mask, self.IMAGE_SIZE)

        label_tensor = torch.tensor(label, dtype=torch.long)

        bbox = self._get_bbox(stem, orig_w, orig_h)

        if self.augment and torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[2])  # horizontal flip
            mask = torch.flip(mask, dims=[1])

            bbox = bbox.clone()
            bbox[0] = self.IMAGE_SIZE - bbox[0]

        return {
            "image": image,
            "label": label_tensor,
            "bbox": bbox,
            "mask": mask
        }
    
    def _get_bbox(self, stem: str, orig_w: int, orig_h: int) -> torch.Tensor:

        if stem in self._bbox_cache:
            return self._bbox_cache[stem]
        
        xml_path = os.path.join(self._xml_dir, f"{stem}.xml")
        if os.path.exists(xml_path):
            xmin , ymin, xmax, ymax = _parse_voc_xml(xml_path)

            xmin = max(0, xmin); ymin = max(0, ymin)
            xmax = min(orig_w, xmax); ymax = min(orig_h, ymax)

        else :
            # fallback to full image bbox if XML annotation is missing
            xmin, ymin, xmax, ymax = 0, 0, orig_w, orig_h
        
        sx = self.IMAGE_SIZE / orig_w
        sy = self.IMAGE_SIZE / orig_h
        xmin_s = xmin * sx; xmax_s = xmax * sx
        ymin_s = ymin * sy; ymax_s = ymax * sy

        X_center = (xmin_s + xmax_s) / 2.0
        Y_center = (ymin_s + ymax_s) / 2.0
        Width = xmax_s - xmin_s
        Height = ymax_s - ymin_s
        bbox = torch.tensor([X_center, Y_center, Width, Height], dtype=torch.float32)
        self._bbox_cache[stem] = bbox

        return bbox
    
def _parse_voc_xml(xml_path: str) -> tuple[int, int, int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bndbox = root.find(".//bndbox")
    xmin = int(float(bndbox.find("xmin").text))
    ymin = int(float(bndbox.find("ymin").text))
    xmax = int(float(bndbox.find("xmax").text))
    ymax = int(float(bndbox.find("ymax").text))
    return xmin, ymin, xmax, ymax