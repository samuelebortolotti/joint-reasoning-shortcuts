# DSL model for MNIST
import torch
import torch.nn as nn
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


class MnistDSL(CBMModule):
    """DSL MODEL FOR MNIST"""

    NAME = "mnistdsl"

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
        super(MnistDSL, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split
        
        if args.task == "addition":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 19 if not args.dataset in ["halfmnist", "restrictedmnist"] else 9
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 37 if not args.dataset in ["halfmnist", "restrictedmnist"] else 10
        elif args.task == "multiop":
            self.n_facts = 5
            self.nr_classes = 3
        elif args.task in ["sumparity", "sumparityrigged"]:
            self.n_facts = 10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            self.nr_classes = 2

        # opt and device
        self.opt = None
        self.args = args
        self.device = get_device()
        self.weights = torch.nn.Parameter(torch.randn([self.n_facts, self.n_facts, self.nr_classes], requires_grad=True).to(self.device))
        self.epsilon_digits = args.eps_sym
        self.epsilon_rules = args.eps_rul
        args.model_m = self


    def epsilon_greedy(self, t, eval, dim=1):
        if eval:
            # In evaluation mode, select the max value (greedy action)
            truth_values, chosen_symbols = torch.max(t, dim=dim)
        else:
            # In training mode, use epsilon-greedy
            random_selection = torch.rand((t.shape[0],), device=self.device) < self.epsilon_digits
            symbol_index_random = torch.randint(t.shape[1], (t.shape[0],), device=self.device)
            _, symbol_index_max = torch.max(t, dim=dim)

            # Select randomly or greedily based on epsilon
            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)
            truth_values = torch.gather(t, dim, chosen_symbols.unsqueeze(1))

        return truth_values, chosen_symbols

    def get_rules_matrix(self, eval):
        if eval:
            # In evaluation mode, return max over the softmax values of weights
            return torch.max(torch.nn.functional.softmax(self.weights, dim=2), dim=2, keepdim=True)
        else:
            # In training mode, use epsilon-greedy to select the rules
            n_digits, n_output_symbols = self.weights.shape[0], self.weights.shape[2]

            random_selection = torch.rand((n_digits, n_digits), device=self.device) < self.epsilon_rules
            symbol_index_random = torch.randint(n_output_symbols, (n_digits, n_digits), device=self.device)
            _, symbol_index_max = torch.max(self.weights, dim=2)

            # Select rules based on epsilon-greedy strategy
            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            # Gather the corresponding truth values
            truth_values = torch.gather(torch.nn.functional.softmax(self.weights, dim=2),
                                        2, chosen_symbols.unsqueeze(2)).squeeze(2)

            return truth_values, chosen_symbols


    def forward(self, x, eval=False):
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

        # get the result of the inference via DSL
        rules_weights, g_matrix = self.get_rules_matrix(eval)

        # if self.n_images == 2:
        truth_values_x, chosen_symbols_x = self.epsilon_greedy(pCs[:, 0, :], eval)
        truth_values_y, chosen_symbols_y = self.epsilon_greedy(pCs[:, 1, :], eval)

        symbols_truth_values = torch.concat(
            [rules_weights[chosen_symbols_x, chosen_symbols_y].view(-1, 1),
            truth_values_x.view(-1, 1),
            truth_values_y.view(-1, 1)], dim=1)
        
        pred = g_matrix[chosen_symbols_x, chosen_symbols_y]

        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)    
        py = predictions_truth_values

        return {"CS": cs, "YS": py, "pCS": pCs, "PRED": pred, "KNOWLEDGE": torch.nn.functional.softmax(self.weights, dim=2)}


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
        self.opt = madgrad.MADGRAD(
            [{'params': list(self.parameters())[:1]}, {'params': list(self.parameters())[1:], 'lr': 1e-3}], lr=args.lr
        )