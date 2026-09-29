from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, CLEVR_get_loader
from datasets.utils.clevr_creation import CLEVRDataset, CLEVR_Preprocess
from backbones.clevrcnn import CLEVRCNN, CLEVRDECODER
import time


class CLEVR(BaseDataset):
    NAME = "clevr"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.return_embeddings = False

    def get_data_loaders(self):
        start = time.time()

        clevr_base_path = "clevr"

        self.dataset_train = CLEVRDataset(
            base_path=clevr_base_path,
            split="train"
        )

        self.dataset_val = CLEVRDataset(
            base_path=clevr_base_path,
            split="val",
        )
        self.dataset_test = CLEVRDataset(
            base_path=clevr_base_path,
            split="test",
        )
        self.dataset_ood = CLEVRDataset(
            base_path=clevr_base_path,
            split="ood",
        )

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))

        keep_order = True if self.return_embeddings else False
        self.train_loader = CLEVR_get_loader(
            self.dataset_train, self.args.batch_size, val_test=keep_order
        )
        self.val_loader = CLEVR_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = CLEVR_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )
        self.ood_loader = CLEVR_get_loader(
            self.dataset_ood, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return CLEVRCNN(num_classes=15), CLEVRDECODER()
        return CLEVRCNN(num_classes=15), CLEVRDECODER()

    def get_split(self):
        return 1, ()

    def get_concept_labels(self):
        return [0, 1]

    def get_labels(self):
        return [0, 1, 2]

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))
        print("Test OOD samples", len(self.dataset_ood))

if __name__ == "__main__":
    dataset = CLEVR()

    for batch_idx, data in enumerate(dataset.train_loader):
        images, labels, concepts = data

        print(images[0].shape)
        print(labels[0])
        print(concepts[0])
        quit()
