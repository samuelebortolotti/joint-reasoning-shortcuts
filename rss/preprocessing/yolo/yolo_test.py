import sys
from datasets.utils.clevr_creation import CLEVR_Preprocess
import cv2
from ultralytics import YOLO
import torch
import os
import numpy as np
import csv

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def crop_bounding_boxes_dataset(model, split):
    dataset = CLEVR_Preprocess("clevr", split)

    save_dir = "clevr/preprocess"
    os.makedirs(save_dir, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4,
        drop_last=False,
    )

    for data, bbox, concepts in dataloader:
        results = model(data)
        os.makedirs(os.path.join(save_dir, split, data[0][-10:-4]), exist_ok=True)

        img = cv2.imread(data[0])
        boxes = results[0].boxes.xyxy.tolist()

        unassigned_box = [a[0] for a in torch.split(bbox, 4, dim=1)]
        unassigned_concepts = [a[0] for a in torch.split(concepts, 15, dim=1)]

        concept_path = os.path.join(save_dir, split, data[0][-10:-4], "ordered_concepts.csv")
        new_concepts = []

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box
            det_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            # check the closes box in bbox and assign the associated concepts
            if len(unassigned_box) != 0:
                ious = [compute_iou(box, gt_box) for gt_box in unassigned_box]
                closest_idx = np.argmax(ious)
                closest_concept = unassigned_concepts[closest_idx]
                
                unassigned_concepts.pop(closest_idx)
                unassigned_box.pop(closest_idx)
                new_concepts.extend(closest_concept)

            ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            resized_object = cv2.resize(ultralytics_crop_object, (128, 128))
            cv2.imwrite(os.path.join(save_dir, split, data[0][-10:-4], str(i) + ".jpg"), resized_object)

        for c in unassigned_concepts:
            new_concepts.extend(c)

        assert len(new_concepts) == 60

        with open(concept_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([t.item() for t in new_concepts])

def main():
    # YOLO 11 FINETUNED
    model = YOLO("weights/best.pt")
    crop_bounding_boxes_dataset(model, "train")
    crop_bounding_boxes_dataset(model, "val")
    crop_bounding_boxes_dataset(model, "test")
    crop_bounding_boxes_dataset(model, "ood")
    

if __name__ == "__main__":
    main()
