# CBM model for MNIST
import torch
import torch.nn as nn
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from utils.dpl_loss import ADDMNIST_DPL
from models.utils.cbm_module import CBMModule


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistCBM(CBMModule):
    """CBM MODEL FOR MNIST"""

    NAME = "mnistcbm"

    """
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    """

    def __init__(
        self,
        encoder,
        n_images=2,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=20,
        nr_classes=19,
    ):
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder
            n_images (int, default=2): number of images
            c_split: concept splits
            args: command line arguments
            model_dict (default=None): model dictionary
            n_facts (int, default=20): number of concepts
            nr_classes (int, nr_classes): number of classes

        Returns:
            None: This function does not return a value.
        """
        super(MnistCBM, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split

        # Worlds-queries matrix
        if args.task == "addition":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 19 if not args.dataset in ["halfmnist", "restrictedmnist"] else 9
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.nr_classes = 3
        elif args.task in ["sumparity", "sumparityrigged"]:
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 2

        # opt and device
        self.opt = None
        self.device = get_device()
        self.multi_linear = args.multi_linear

        if args.multi_linear:
            self.classifier = nn.Sequential(
                nn.Linear(self.n_facts**2, self.nr_classes, bias=False),
                nn.Softmax(dim=1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.n_facts*2, self.nr_classes, bias=False),
                nn.Softmax(dim=1),
            )

        torch.nn.init.normal_(self.classifier[0].weight, mean=0.0, std=3.0)

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
        """
        # Image encoding
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        cs = [self.encoder(xi)[0] for xi in xs]
        cs = torch.stack(cs, dim=1) if cs[0].dim() == 2 else torch.cat(cs, dim=1)

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)
        # get the result of the inference
        py = self.cmb_inference(pCs)  # cs

        return {"CS": cs, "YS": py, "pCS": pCs, "KNOWLEDGE": self.classifier}


    def cmb_inference(self, pCs, query=None):
        """Performs inference inference

        Args:
            self: instance
            cs: concepts logits
            query (default=None): query

        Returns:
            query_prob: query probability
        """
        if self.multi_linear:
            combined_features = pCs[:, 0, :].unsqueeze(2).multiply(pCs[:, 1, :].unsqueeze(1)).view(pCs.shape[0], -1)
        else:
            combined_features = torch.cat([pCs[:, 0, :], pCs[:, 1, :]], dim=-1)
        return self.classifier(combined_features)


    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents
            split (int, default=2): numbers of split

        Returns:
            vec: normalized concepts
        """
        # Extract probs for each digit
        eps = 1e-5
        prob_digits = F.softmax(z, dim=-1) + eps
        prob_digits /= prob_digits.sum(dim=-1, keepdim=True)
        return prob_digits.view(-1, 2, self.n_facts)

    @staticmethod
    def get_loss(args):
        """Loss function for the architecture

        Args:
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError if the loss function is not available
        """
        if args.dataset in [
            "addmnist",
            "shortmnist",
            "restrictedmnist",
            "halfmnist",
            "clipshortmnist",
        ]:
            return ADDMNIST_DPL(ADDMNIST_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initialize optimizer

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )
