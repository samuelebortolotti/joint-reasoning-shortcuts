import sys
from datasets.utils.clevr_creation import CLEVR_Preprocess
from ultralytics import YOLO
import torch
import random
import numpy as np

def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(13)

    dataset = CLEVR_Preprocess("clevr", "train")
    dataset.yolo_dataset_setup()

    model = YOLO("yolo11n.pt")
    model_path = "yolo11n_finetunedv2.pt"
    _ = model.train(
        data="./data.yaml",
        epochs=10,
        project="yolo_training",
        name="yolov11n_finetune",
        device=1,
        save_period=1,
        imgsz=320,
        plots=True
    )

    # Save the fine-tuned model
    model.save(model_path)


if __name__ == "__main__":
    main()
