# MNIST CBM with Reconstruction
import torch
from utils.dpl_loss import ADDMNIST_DPL
from utils.args import *
from utils.conf import get_device
from models.utils.deepproblog_modules import GraphSemiring
from models.utils.utils_problog import *
from utils.losses import *
from models.utils.cbm_module import CBMModule


def get_parser() -> ArgumentParser:
    """Argument parser

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistCBMRec(CBMModule):
    """CBM with Reconstruction for MNIST"""

    NAME = "mnistcbmrec"
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
            decoder (nn.Module): decoder
            n_images (int, default=2): number of images
            c_split: concept splits
            args: command line arguments
            model_dict (default=None): model dictionary
            n_facts (int, default=20): number of concepts
            nr_classes (int, nr_classes): number of classes

        Returns:
            None: This function does not return a value.
        """
        super(MnistCBMRec, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split

        # add decoder
        self.decoder = decoder

        # Worlds-queries matrix
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
                nn.Linear(self.n_facts**2, self.nr_classes),
                nn.Softmax(dim=1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.n_facts*2, self.nr_classes),
                nn.Softmax(dim=1),
            )

        torch.nn.init.normal_(self.classifier[0].weight, mean=0.0, std=3.0)

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            output_dict: output dictionary
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

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)
        # get the result of the inference
        py = self.cmb_inference(pCs) 

        return {
            "YS": py,
            "CS": cs,
            "pCS": pCs,
            "MUS": mus,
            "LOGVARS": logvars,
            "RECS": recs,
            "KNOWLEDGE": self.classifier
        }


    def cmb_inference(self, pCs, query=None):
        """Performs inference inference

        Args:
            self: instance
            cs: concepts logits
            query (default=None): query

        Returns:
            query_prob: query probability
        """

        combined_features = pCs[:, 0, :].unsqueeze(2).multiply(pCs[:, 1, :].unsqueeze(1)).view(pCs.shape[0], -1)
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

