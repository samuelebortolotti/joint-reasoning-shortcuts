# DSL model for MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from utils.dpl_loss import ADDMNIST_DPL
from models.utils.cbm_module import CBMModule
import models.utils.madgrad as madgrad

def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistDSLDPLRec(CBMModule):
    """DPL variant of DSL MODEL FOR MNIST"""

    NAME = "mnistdsldplrec"

    """
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    """

    def __init__(
        self,
        encoder,
        decoder,
        n_images=2,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=10,
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
        super(MnistDSLDPLRec, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )
        # add decoder
        self.decoder = decoder

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split
        
        if args.task == "addition":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.nr_classes = 3
        elif args.task == "sumparity" or args.task == "sumparityrigged":
            self.n_facts = 10
            self.nr_classes = 19

        # opt and device
        self.opt = None
        self.device = get_device()
        self.weights = torch.nn.Parameter(torch.randn([self.n_facts, self.n_facts, self.nr_classes], requires_grad=True).to(self.device))
        
    def get_rules_matrix(self, eval=False):
        return torch.max(torch.nn.functional.softmax(self.weights, dim=2), dim=2, keepdim=True)

    def forward(self, x):
        """Forward method

        Args:
            x (torch.tensor): input vector

        Returns:
            dict: output dictionary
        """
        xs = x.chunk(self.n_images, dim=-1)

        encoded = [self.encoder(img) for img in xs]
        cs, mus, logvars = zip(*encoded)

        eps = [torch.randn_like(lv) for lv in logvars]
        latents = [(mu + e * lv.exp()).view(len(e), -1) for mu, e, lv in zip(mus, eps, logvars)]
        discrete_latents = [F.gumbel_softmax(c, tau=1, hard=True, dim=-1).squeeze(dim=1) for c in cs]
        recs = [self.decoder(torch.cat([l, d], dim=1)) for l, d in zip(latents, discrete_latents)]
        cs = torch.stack(cs, dim=1) if cs[0].ndim == 2 else torch.cat(cs, dim=1)
        mus = torch.stack(mus, dim=-1) if mus[0].ndim == 2 else torch.cat(mus, dim=1)
        logvars = torch.stack(logvars, dim=-1) if logvars[0].ndim == 2 else torch.cat(logvars, dim=1)
        recs = torch.cat(recs, dim=-1)

        pCs = self.normalize_concepts(cs)
        py = F.softmax(self.weights, dim=2)
        label_prob, label_predicted = self._dpl_inference(pCs[:, 0, :], pCs[:, 1, :], py)

        return {
            "CS": cs,
            "YS": label_prob,
            "pCS": pCs,
            "PRED": label_predicted,
            "MUS": mus,
            "LOGVARS": logvars,
            "RECS": recs,
        }

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

    def _dpl_inference(self, c1, c2, py, eval=False):
        y = torch.einsum("bi,bj,ijk->bk", c1, c2, py)
        y += 1e-5
        Z = y.sum(dim=1, keepdim=True)
        y /= Z
        return y, torch.argmax(y, dim=-1)

    def start_optim(self, args):
        """Initialize optimizer

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = madgrad.MADGRAD(
            [
                {"params": list(self.parameters())[:1]},
                {"params": list(self.parameters())[1:], "lr": 1e-3},
            ],
            lr=args.lr,
        )