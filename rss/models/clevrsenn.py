# CBM model for CLEVR
import torch
import torch.nn as nn
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from utils.dpl_loss import CLEVR_DPL
from models.utils.cbm_module import CBMModule
from torchvision.transforms import ToPILImage
from backbones.clevrcnn import CLEVRCNN


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class ClevrSENN(CBMModule):
    """CBM MODEL FOR CLEVR"""

    NAME = "clevrsenn"

    def __init__(
        self,
        encoder,
        decoder,
        n_images=4,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=15,
        nr_classes=4,
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
        super(ClevrSENN, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.c_split = c_split

        # facts and classes
        self.n_facts = n_facts
        self.nr_classes = nr_classes
        self.n_images = n_images

        # opt and device
        self.opt = None
        self.device = get_device()

        # decoder and relevance score computation
        self.relevance_score = CLEVRCNN(c_dim=15)
        torch.nn.init.normal_(self.relevance_score.dense_c.weight, mean=0.0, std=3.0)

        self.decoder = decoder

        self.classifier = nn.Sequential(
            nn.Linear((self.n_facts + 1) * 4, self.nr_classes),
            nn.Softmax(dim=1),
        )

    def get_pred_from_prob(self, pCs, presence=True):
        # get the result of the inference
        py = self.cmb_inference(pCs, presence)

        return py

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            output_dict: output dictionary
        """
        # Pre-allocate tensors
        n_images = x.shape[1]
        batch_size = x.shape[0]
        mask = (x == -1).all(-1).all(-1).all(-1)

        cs = torch.zeros((batch_size, n_images, 1, self.n_facts), device=x.device)
        mus = torch.zeros((batch_size, n_images, 15, 16), device=x.device)
        logvars = torch.zeros((batch_size, n_images, 15, 16), device=x.device)
        recs = torch.full((batch_size, n_images, *x.shape[2:]), -1, device=x.device, dtype=torch.float32)

        # Iterate over images
        for i in range(n_images):
            current_images = x[:, i]
            valid_indices = ~mask[:, i]

            if valid_indices.any():
                valid_predictions, mu, logvar = self.encoder(current_images[valid_indices])
                
                # Update the placeholders with valid predictions
                cs[valid_indices, i] = valid_predictions
                mus[valid_indices, i] = mu
                logvars[valid_indices, i] = logvar
                
                # Add variational variables + discrete predictions
                eps = torch.randn_like(logvars[valid_indices, i])

                latents = [
                    (mus[valid_indices, i] + eps * logvars[valid_indices, i].exp()).view(valid_indices.sum(), -1),
                    F.gumbel_softmax(cs[valid_indices, i, 0, :], tau=1, hard=True, dim=-1)
                ]
                latents = torch.cat(latents, dim=1)
                
                recs[valid_indices, i] = self.decoder(latents)  # Update reconstructed images

        # TODO

        cs = cs.view(x.shape[0], n_images, self.n_facts)
        pCs = self.normalize_concepts(cs)
        py = self.cmb_inference(pCs)

        epsilon = 1e-6
        py = py + epsilon
        py = py / py.sum(dim=1, keepdim=True)

        pCs = pCs.view(pCs.shape[0], pCs.shape[1] * pCs.shape[2])

        return {
            "YS": py,
            "CS": cs,
            "pCS": pCs,
            "MUS": mus,
            "LOGVARS": logvars,
            "RECS": recs,
        }

    
    def cmb_inference(self, cs, presence=False):
        """Performs inference inference

        Args:
            self: instance
            cs: concepts logits
            query (default=None): query

        Returns:
            query_prob: query probability
        """
        if presence:
            return self.classifier(cs.view(cs.shape[0], -1))

        presence_mask = torch.all(cs == -1, dim=-1, keepdim=True).to(cs.dtype)
        result = torch.cat([presence_mask, cs], dim=-1)
        query_prob = self.classifier(result.view(result.shape[0], -1))

        return query_prob

    def normalize_concepts(self, z, split=2):
        eps = 1e-5
        mask = torch.all(z != 0, dim=-1)

        colors, shapes, materials, sizes = [x.clone() for x in torch.split(z, [8, 3, 2, 2], dim=-1)]

        for feature in [colors, shapes, materials, sizes]:
            feature[mask] = nn.functional.softmax(feature[mask], dim=-1)
            feature[mask] = feature[mask] + eps
            feature[mask] = feature[mask] / feature[mask].sum(dim=-1, keepdim=True)

        return torch.cat([colors, shapes, materials, sizes], dim=-1)


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
            "clevr",
        ]:
            return CLEVR_DPL(CLEVR_Cumulative)
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

