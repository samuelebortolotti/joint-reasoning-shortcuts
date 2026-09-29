from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.mnist_creation import load_2MNIST
from backbones.addmnist_joint import MNISTPairsEncoder, MNISTPairsDecoder
from backbones.addmnist_repeated import MNISTRepeatedEncoder
from backbones.addmnist_single import MNISTSingleEncoder, MNISTSingleDecoder
from backbones.mnistcnn import MNISTAdditionCNN
from backbones.disjointmnistcnn import DisjointMNISTAdditionCNN
from backbones.mnist_net import MNIST_Net
import numpy as np
from copy import deepcopy
import torch


class ADDMNIST(BaseDataset):
    NAME = "addmnist"
    DATADIR = "data/raw"

    def get_data_loaders(self):
        dataset_train, dataset_val, dataset_test = load_2MNIST(
            c_sup=self.args.c_sup, which_c=self.args.which_c, args=self.args
        )

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        
        if self.args.task == "sumparityrigged":

            self.pairs = []
            self.negative_pairs = []
            for a in range(10):
                for b in range(10):
                    # skipping even + odd
                    if a % 2 == 0 and b % 2 == 1:
                        self.negative_pairs.append((a, b))
                        continue
                    self.pairs.append((a, b))

            self.dataset_train, self.dataset_val, self.dataset_test, self.ood_test_dataset = (
                self.filtrate(self.dataset_train, self.dataset_val, self.dataset_test, deepcopy(self.dataset_test))
            )
            self.ood_loader = get_loader(self.ood_test_dataset, self.args.batch_size, val_test=True)

        self.train_loader = get_loader(
            dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = get_loader(dataset_val, self.args.batch_size, val_test=True)
        self.test_loader = get_loader(dataset_test, self.args.batch_size, val_test=True)

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.joint:
            if not self.args.splitted:
                return MNISTPairsEncoder(), MNISTPairsDecoder()
            else:
                if self.args.backbone == "neural":
                    return MNISTAdditionCNN(), None
                return MNISTRepeatedEncoder(), MNISTPairsDecoder()
        else:
            if self.args.backbone == "neural":
                return DisjointMNISTAdditionCNN(n_images=self.get_split()[0]), None

            if "dsl" in self.args.model:
                return MNISTSingleEncoder(), MNISTSingleDecoder()

            return MNISTSingleEncoder(), MNISTSingleDecoder()

    def get_split(self):
        if self.args.joint:
            return 1, (10, 10)
        else:
            return 2, (10,)

    def get_concept_labels(self):
        return [str(i) for i in range(10)]

    def get_labels(self):
        return [str(i) for i in range(19)]

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train.data))
        print("Validation samples", len(self.dataset_val.data))
        print("Test samples", len(self.dataset_test.data))


    def construct_mask(self, concepts, pairs):
        pairs_set = set(map(tuple, pairs))  # Convert to set for fast lookup
        concepts_pairs = list(map(tuple, concepts[:, :2]))  # Ensure (a, b) tuples
        
        # Check if each concept exists in the pairs_set
        mask = np.array([pair in pairs_set for pair in concepts_pairs])

        return mask

    def apply_filter(self, dataset, mask):
        dataset.data = dataset.data[mask]
        dataset.concepts = dataset.concepts[mask]
        dataset.real_concepts = dataset.real_concepts[mask]
        dataset.targets = dataset.targets[mask] if isinstance(dataset.targets, np.ndarray) else np.array(dataset.targets)[mask]
        return dataset

    def filtrate(self, train_dataset, val_dataset, test_dataset, ood_set):
        datasets = [train_dataset, val_dataset, test_dataset, ood_set]
        masks = [
            self.construct_mask(d.real_concepts, self.negative_pairs if i == 3 else self.pairs)
            for i, d in enumerate(datasets)
        ]
        return [self.apply_filter(d, m) for d, m in zip(datasets, masks)]


    def get_instance(self, i, j, num_samples=5):
        mask = (self.dataset_test.real_concepts[:, 0] == i) & (self.dataset_test.real_concepts[:, 1] == j)

        valid_indices = np.where(mask)[0]

        if valid_indices.size == 0:
            raise ValueError(f"No matching elements found for x={i}, y={j}")

        sampled_indices = np.random.choice(valid_indices, size=min(num_samples, len(valid_indices)), replace=False)
        sampled_data = self.dataset_test.data[sampled_indices]
        sampled_data = torch.tensor(sampled_data, dtype=torch.float)
        sampled_data = torch.reshape(sampled_data, (num_samples, 1, sampled_data.shape[1], sampled_data.shape[2]))

        return sampled_data