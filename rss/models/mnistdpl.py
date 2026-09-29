# DPL model for MNIST
import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.deepproblog_modules import GraphSemiring
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import ADDMNIST_DPL


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistDPL(DeepProblogModel):
    """DPL MODEL FOR MNIST"""

    NAME = "mnistdpl"
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
        super(MnistDPL, self).__init__(
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
            self.w_q = build_worlds_queries_matrix(2, self.n_facts, "addmnist")
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.w_q = build_worlds_queries_matrix(2, self.n_facts, "productmnist")
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.w_q = build_worlds_queries_matrix(2, self.n_facts, "multiopmnist")
            self.nr_classes = 3
        elif args.task in ["sumparity", "sumparityrigged"]:
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.w_q = build_worlds_queries_matrix(2, self.n_facts, "sumparity")
            self.nr_classes = 2

        # opt and device
        self.opt = None
        self.device = get_device()
        self.w_q = self.w_q.to(self.device)

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

        # Problog inference to compute worlds and query probability distributions
        py, _ = self.problog_inference(pCs)

        return {"CS": cs, "YS": py, "pCS": pCs}

    def problog_inference(self, pCs, query=None):
        """Performs ProbLog inference to retrieve the worlds probability distribution P(w). Works with two encoded bits.

        Args:
            self: instance
            pCs: probability of concepts
            query (default=None): query

        Returns:
            query_prob: query probability
            worlds_prob: worlds probability
        """
        # Extract first and second digit probability
        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1, :]

        # Compute world probability P(w) using outer product
        worlds_prob = (prob_digit1.unsqueeze(-1) * prob_digit2.unsqueeze(1)).reshape(-1, self.n_facts**2)

        # Compute query probability P(q)
        query_prob = torch.stack(
            [self.compute_query(i, worlds_prob).view(-1) for i in range(self.nr_classes)],
            dim=-1
        )

        # Normalize query probabilities with a small offset
        query_prob = (query_prob + 1e-5) / query_prob.sum(dim=-1, keepdim=True)
        return query_prob, worlds_prob


    def compute_query(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        return torch.sum(self.w_q[:, query] * worlds_prob, dim=1, keepdim=True)

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents
            split (int, default=2): numbers of split

        Returns:
            vec: normalized concepts
        """
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
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
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

    # override
    def to(self, device):
        super().to(device)
        self.w_q = self.w_q.to(device)
