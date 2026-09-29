# CBM model for CLEVR
import torch
import torch.nn as nn
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from utils.dpl_loss import CLEVR_DPL
from models.utils.cbm_module import CBMModule
from torchvision.transforms import ToPILImage


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class ClevrCBM(CBMModule):
    """CBM MODEL FOR CLEVR"""

    NAME = "clevrcbm"

    def __init__(
        self,
        encoder,
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
        super(ClevrCBM, self).__init__(
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

        self.classifier = nn.Sequential(
            nn.Linear((self.n_facts + 1) * 4, self.nr_classes),
            nn.Softmax(dim=1),
        )

    def get_pred_from_prob(self, pCs, presence=True):
        # get the result of the inference
        py = self.cmb_inference(pCs, presence)

        # Add a small offset
        # epsilon = 1e-6
        # py = py + epsilon

        # # Renormalize probabilities
        # py = py / py.sum(dim=1, keepdim=True)

        return py

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
        """
        # Image encoding
        cs = []
        n_images = len(x[0])

        for i in range(n_images):
            current_images = x[:, i]

            mask = (current_images == -1).all(dim=1).all(dim=1).all(dim=1)

            lc = torch.zeros((current_images.shape[0], 1, self.n_facts), device=x.device)  # Placeholder for all predictions

            if not mask.all():
                valid_indices = ~mask
                valid_predictions, _, _ = self.encoder(current_images[valid_indices])
                lc[valid_indices] = valid_predictions

            cs.append(lc)

        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)

        # get the result of the inference
        py = self.cmb_inference(pCs)  # cs

        # Add a small offset
        epsilon = 1e-6
        py = py + epsilon

        # Renormalize probabilities
        py = py / py.sum(dim=1, keepdim=True)

        pCs = pCs.view(pCs.shape[0], pCs.shape[1] * pCs.shape[2])

        return {"CS": cs, "YS": py, "pCS": pCs}

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
            flattened_cs = cs.view(cs.shape[0], cs.shape[1] * cs.shape[2])
            query_prob = self.classifier(flattened_cs)
            return query_prob

        presence_mask = torch.all(cs == -1, dim=-1)
        presence_mask = presence_mask.float().unsqueeze(-1)  # Shape: [128, 4, 1]
        result = torch.cat([presence_mask, cs], dim=-1)  # Shape: [128, 4, 16]

        # flatten the cs
        flattened_cs = result.view(result.shape[0], result.shape[1] * result.shape[2])
        query_prob = self.classifier(flattened_cs)

        return query_prob

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents
            split (int, default=2): numbers of split

        Returns:
            vec: normalized concepts
        """

        colors, shapes, materials, sizes = z[:, :, :8], z[:, :, 8:11], z[:, :, 11:13], z[:, :, 13:15]

        mask = torch.all(z != 0, dim=-1)

        colors[mask] = nn.Softmax(dim=-1)(colors[mask])
        shapes[mask] = nn.Softmax(dim=-1)(shapes[mask])
        materials[mask] = nn.Softmax(dim=-1)(materials[mask])
        sizes[mask] = nn.Softmax(dim=-1)(sizes[mask])

        # Clamp digits_probs to avoid ProbLog underflow
        eps = 1e-5
        colors[mask] = colors[mask] + eps
        with torch.no_grad():
            Z1 = torch.sum(colors[mask], dim=-1, keepdim=True)
        colors[mask] = colors[mask] / Z1  # Normalization

        shapes[mask] = shapes[mask] + eps
        with torch.no_grad():
            Z2 = torch.sum(shapes[mask], dim=-1, keepdim=True)
        shapes[mask] = shapes[mask] / Z2  # Normalization

        sizes[mask] = sizes[mask] + eps
        with torch.no_grad():
            Z3 = torch.sum(sizes[mask], dim=-1, keepdim=True)
        sizes[mask] = sizes[mask] / Z3  # Normalization

        materials[mask] = materials[mask] + eps
        with torch.no_grad():
            Z4 = torch.sum(materials[mask], dim=-1, keepdim=True)
        materials[mask] = materials[mask] / Z4  # Normalization

        cat = torch.cat([colors, shapes, materials, sizes], dim=-1)
        return cat

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