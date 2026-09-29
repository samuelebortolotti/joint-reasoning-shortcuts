import glob
import os
from torchvision import transforms
import torchvision
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import joblib
from PIL import Image
from torchvision.datasets.folder import pil_loader
import json
import csv
import time
import h5py

def normalize_bounding_box(x, y, box_width, box_height, image_width, image_height):
    center_x = (x + box_width / 2) / image_width
    center_y = (y + box_height / 2) / image_height
    norm_width = box_width / image_width
    norm_height = box_height / image_height

    return center_x, center_y, norm_width, norm_height

class CLEVR_Preprocess(torch.utils.data.Dataset):
    def __init__(self, base_path, split, fast_rcnn=True):

        self.base_path = base_path
        self.split = split
        self.fast_rcnn = fast_rcnn

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.list_images= glob.glob(os.path.join(self.base_path, self.split,"*.png"))
        self.img_number = [i for i in range(len(self.list_images))]

        self.metas=[]
        self.targets=torch.LongTensor([])

        for item in range(len(self.list_images)):
            target_id=os.path.join(self.base_path, self.split, f"CLEVR_{self.split}_{str(self.img_number[item]).zfill(6)}.json")
            with open(target_id, "r") as file:
                meta = json.load(file)
            self.metas.append(meta)
            self.targets = torch.concat([self.targets,torch.LongTensor([meta["label"]])])

    def yolo_dataset_setup(self):
        for item in range(len(self.list_images)):
            target_id=os.path.join(self.base_path, self.split, f"CLEVR_{self.split}_{str(self.img_number[item]).zfill(6)}.json")
            
            with open(target_id, "r") as file:
                meta = json.load(file)

            with open(os.path.join(self.base_path, self.split, f"CLEVR_{self.split}_{str(self.img_number[item]).zfill(6)}.txt"), "w") as file:
                print("Writing...", os.path.join(self.base_path, self.split, f"CLEVR_{self.split}_{str(self.img_number[item]).zfill(6)}.txt"))
                numbers = [[]]
                for i in range(len(meta['objects'])):
                    center_x, center_y, norm_width, norm_height = normalize_bounding_box(meta['objects'][i]["x"], meta['objects'][i]["y"], meta['objects'][i]["width"], meta['objects'][i]["height"], 320, 240)
                    numbers.append([0, center_x, center_y, norm_width, norm_height])

                for number in numbers:
                    file.write(f"{' '.join(str(num) for num in number)}\n")

    @property
    def images_folder(self):
        return os.path.join(self.base_path, self.split)

    @property
    def scenes_path(self):
        return os.path.join(self.base_path)

    def __getitem__(self, item):
        meta=self.metas[item]
        label=self.targets[item]
        concepts = [item for sublist1 in meta["concepts"] for sublist2 in sublist1 for item in sublist2]
        concepts = torch.LongTensor(concepts)

        img_id = self.img_number[item]
        image_id = os.path.join(self.base_path, self.split, f"CLEVR_{self.split}_{str(img_id).zfill(6)}.png")
        
        bbox = []
        for obj in meta['objects']:
            bbox.append(obj['x'])
            bbox.append(obj['y'])
            bbox.append(obj['width'])
            bbox.append(obj['height'])

        for i in range(4 * 4 - len(bbox)):
            bbox.append(1000000)

        bbox = torch.LongTensor(bbox)

        return image_id, bbox, concepts

    def __len__(self):
        return len(self.list_images)
    
class CLEVRDataset():

    def __init__(self, base_path, split):

        self.base_path = base_path
        self.split = split
        self.transform = transforms.ToTensor()

        imgs_path = os.path.join(self.base_path, "preprocess", self.split)
        list_images = sorted([d for d in os.listdir(imgs_path) if os.path.isdir(os.path.join(imgs_path, d))])

        labels, concepts, all_images = [], [], []

        for idx, idf in enumerate(list_images):
            image_folder = os.path.join(imgs_path, f"{idf}")
            meta_scene = os.path.join(self.base_path, "clevr_images", self.split, f"CLEVR_{self.split}_{idf}.json")
            concepts_path = os.path.join(imgs_path, f"{idf}", "ordered_concepts.csv")

            if not os.path.exists(meta_scene):
                print(f"{meta_scene} does not exist.")
                continue

            with open(meta_scene, "r") as file:
                data = json.load(file)
            label = np.array(data["label"])

            with open(concepts_path, mode='r') as file:
                concept_values = [int(c) for c in list(csv.reader(file))[0]]

            labels.append(label)
            concepts.append(np.array(concept_values))

            images = self._get_sorted_file_paths(image_folder)

            rtn_images = [self.transform(pil_loader(im)) for im in images]

            prev_len = len(rtn_images)
            if prev_len < 4:
                rtn_images += [torch.full_like(rtn_images[0], -1) for _ in range(4 - prev_len)]
            else:
                rtn_images = rtn_images[:4]
                prev_len = len(rtn_images)

            assert prev_len == (torch.stack(rtn_images) != -1).all(1).all(1).all(1).sum(), f"Mismatch in {idx}, {prev_len} vs {(torch.stack(rtn_images) == -1).all(1).all(1).sum()}"

            all_images.append(torch.stack(rtn_images).numpy())

        self.labels = np.stack(labels, axis=0)

        self.concepts = np.stack(concepts, axis=0)
        self.list_images = np.stack(all_images, axis=0)


    def _get_sorted_file_paths(self, folder_path):
        return sorted(glob.glob(os.path.join(folder_path, "*.jpg")))


    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, item):
        labels = self.labels[item] - 1
        concepts = self.concepts[item]
        image = self.list_images[item]

        return image, labels, concepts

    def preprocess_and_save(self, h5path):

        self.hdf5_path = h5path
        self.transform = transforms.ToTensor()

        """Preprocesses the dataset and saves it to an HDF5 file, including padded images."""
        imgs_path = os.path.join(self.base_path, "preprocess", self.split)
        list_images = sorted([d for d in os.listdir(imgs_path) if os.path.isdir(os.path.join(imgs_path, d))])

        labels, concepts, all_images = [], [], []

        start = time.time()
        for idx, idf in enumerate(list_images):
            image_folder = os.path.join(imgs_path, f"{idf}")
            meta_scene = os.path.join(self.base_path, "clevr_images", self.split, f"CLEVR_{self.split}_{idf}.json")
            concepts_path = os.path.join(imgs_path, f"{idf}", "ordered_concepts.csv")

            if not os.path.exists(meta_scene):
                print(f"{meta_scene} does not exist.")
                continue

            with open(meta_scene, "r") as file:
                data = json.load(file)
            label = np.array(data["label"])

            with open(concepts_path, mode='r') as file:
                concept_values = [int(c) for c in list(csv.reader(file))[0]]

            labels.append(label)
            concepts.append(np.array(concept_values))

            # Load, transform, and pad images
            images = self._get_sorted_file_paths(image_folder)

            rtn_images = [self.transform(pil_loader(im)) for im in images]

            prev_len = len(rtn_images)
            if prev_len < 4:
                rtn_images += [torch.full_like(rtn_images[0], -1) for _ in range(4 - prev_len)]
            else:
                rtn_images = rtn_images[:4]
                prev_len = len(rtn_images)

            assert prev_len == (torch.stack(rtn_images) != -1).all(1).all(1).all(1).sum(), f"Mismatch in {idx}, {prev_len} vs {(torch.stack(rtn_images) == -1).all(1).all(1).sum()}"

            all_images.append(torch.stack(rtn_images).numpy())

        labels = np.stack(labels, axis=0)
        concepts = np.stack(concepts, axis=0)
        all_images = np.stack(all_images, axis=0)

        with h5py.File(self.hdf5_path, "w") as hf:
            hf.create_dataset("labels", data=labels)
            hf.create_dataset("concepts", data=concepts)
            hf.create_dataset("images", data=all_images, compression="gzip")  # Compress for efficiency

        end = time.time()
        print(f"Preprocessing & saving completed in {end - start:.2f} seconds.")


class CLEVRDatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, which_c=[-1]):
        self.hdf5_path = hdf5_path
        self.hf = h5py.File(self.hdf5_path, "r")

        start = time.time()
        self.labels = self.hf["labels"][:]
        self.concepts = self.hf["concepts"][:]

        if which_c == [-2]:
            self.concepts[:, :] = -1
        elif which_c != [-1]:
            self.concepts = np.split(self.concepts, 4, axis=1)
            concept_ranges = {0: (0, 8), 1: (8, 11), 2: (11, 13), 3: (13, None)}

            for i in range(4):
                for concept_id, (start, end) in concept_ranges.items():
                    if concept_id not in which_c:
                        print("Removing", concept_id)
                        self.concepts[i][:, start:end] = -1

            self.concepts = np.concatenate(self.concepts, axis=1)

        self.images = self.hf["images"][:]
        self.hf.close()
        end = time.time()

        print(f"Reading datasets completed in {end - start:.2f} seconds.")

    def __getitem__(self, item):
        images = self.images[item]
        labels = self.labels[item]
        concepts = self.concepts[item]
        return images, labels, concepts

    def __len__(self):
        return len(self.labels)

    def close(self):
        self.hf.close()

if __name__ == "__main__":
    path = "clevr"
    dest = "clevr/out"

    for split in ["train", "val", "test", "ood"]:
        preprocessor = CLEVRDataset(base_path=path, split=split)
        preprocessor.preprocess_and_save(hdf5_path=f"{dest}clevr_{split}.hdf5")
