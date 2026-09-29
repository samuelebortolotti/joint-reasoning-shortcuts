import torch
import torch.autograd.functional as F
from utils.args import *
from utils.conf import get_device
from utils.dpl_loss import ADDMNIST_DPL
from utils.losses import *
from models.utils.cbm_module import CBMModule
from backbones.addmnist_single import MNISTSingleEncoder

def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class MnistSENN(CBMModule):

    NAME = "mnistsenn"

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
        super(MnistSENN, self).__init__(
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
        # knowledge encoder
        if args.dataset == "addmnist":
            self.relevance_score = MNISTSingleEncoder(c_dim=self.n_facts * self.nr_classes)
            torch.nn.init.normal_(self.relevance_score.dense_c.weight, mean=0.0, std=3.0)
        else:
            raise NotImplementedError()
        self.decoder = decoder

    def forward(self, x):
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

        ks = [self.relevance_score(xi)[0] for xi in xs]
        ks = torch.stack(ks, dim=1) if ks[0].dim() == 2 else torch.cat(ks, dim=1)
        ks = ks.reshape(ks.shape[0], ks.shape[1], self.n_facts, self.nr_classes)
        wx = ks[:, 0, :].unsqueeze(2).multiply(ks[:, 1, :].unsqueeze(1))
        wx = wx.reshape(wx.shape[0], self.n_facts ** 2, self.nr_classes)

        pCs = self.normalize_concepts(cs)
        phi = pCs[:, 0, :].unsqueeze(2).multiply(pCs[:, 1, :].unsqueeze(1)).view(pCs.shape[0], -1).unsqueeze(-1) # pCs.reshape(pCs.shape[0], pCs.shape[1] * pCs.shape[2]).unsqueeze(-1)

        dot = torch.sum(wx * phi, dim=1)
        py = F.softmax(dot, dim=-1)

        py = py + 1e-5
        with torch.no_grad():
            Z = torch.sum(py, dim=-1, keepdim=True)
        py = py / Z

        return {
            "CS": cs, 
            "YS": py, 
            "pCS": pCs, 
            "WX": wx, 
            "PHIX": phi,
            "MUS": mus,
            "LOGVARS": logvars,
            "RECS": recs
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
    
    def get_loss(self, args):
        """Returns the loss function for this architecture

        Args:
            self: instance
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
        """Initializes the optimizer for this architecture

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )
