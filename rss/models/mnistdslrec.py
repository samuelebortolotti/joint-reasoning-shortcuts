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


class MnistDSLRec(CBMModule):
    """DSL MODEL FOR MNIST"""

    NAME = "mnistdslrec"

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
        super(MnistDSLRec, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split
        self.decoder = decoder
        
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

    def epsilon_greedy(self, t, eval, dim=1):
        if eval:
            truth_values, chosen_symbols = torch.max(t, dim=dim)
        else:
            random_selection = torch.rand((t.shape[0],)) < self.epsilon_digits
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(t.shape[1], (t.shape[0],))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(t, dim=dim)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)
            truth_values = torch.gather(t, dim, chosen_symbols.view(-1, 1))

        return truth_values, chosen_symbols

    def get_rules_matrix(self, eval):
        if eval:
            return torch.max(torch.nn.functional.softmax(self.weights, dim=2), dim=2, keepdim=True)
        else:
            n_digits = self.weights.shape[0]
            n_output_symbols = self.weights.shape[2]
            random_selection = torch.rand((n_digits, n_digits)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, (n_digits, n_digits))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(self.weights, dim=2)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            truth_values = torch.gather(torch.nn.functional.softmax(self.weights, dim=2),
                                        2, chosen_symbols.view(n_digits, n_digits, 1)).view(n_digits, n_digits)


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
        cs, mus, logvars = [], [], []
        latents = []
        recs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            c, mu, logvar = self.encoder(xs[i])
            cs.append(c)
            mus.append(mu)
            logvars.append(logvar)

            # extract decodings

            # 1) add variational vars + discrete
            eps = torch.randn_like(logvar)
            L = len(eps)
            latents.append((mu + eps * logvar.exp()).view(L, -1))

            for i in range(len(self.c_split)):
                latents.append(F.gumbel_softmax(c[:, i, :], tau=1, hard=True, dim=-1))

        latents = torch.cat(latents, dim=1)

        # 2) pass to decoder
        recs = self.decoder(latents)

        # return everything
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)
        mus = torch.stack(mus, dim=-1) if clen == 2 else torch.cat(mus, dim=1)
        logvars = (
            torch.stack(logvars, dim=-1) if clen == 2 else torch.cat(logvars, dim=1)
        )

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

        # else:
        #     truth_values, chosen_symbols = self.epsilon_greedy(pCs[:, 0, :], eval)
        #     symbols_truth_values = torch.concat(
        #         [rules_weights[chosen_symbols].view(-1, 1),
        #         truth_values.view(-1, 1)], dim=1)
        #     pred = g_matrix[chosen_symbols]

        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)    
        py = predictions_truth_values

        return {"CS": cs, "YS": py, "pCS": pCs, "PRED": pred, "MUS": mus, "LOGVARS": logvars, "RECS": recs}

    def get_layer_representation(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
        """
        # Image encoding
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _, _ = self.encoder(xs[i])  # sizes are ok
            cs.append(lc)
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        # stacking concepts one on top of the other
        return cs.view(-1, cs.shape[-1])

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

        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        prob_digit1 = nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = nn.Softmax(dim=1)(prob_digit2)

        # Clamp digits_probs to avoid ProbLog underflow
        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1  # Normalization
        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2  # Normalization

        return torch.stack([prob_digit1, prob_digit2], dim=1).view(-1, 2, self.n_facts)

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