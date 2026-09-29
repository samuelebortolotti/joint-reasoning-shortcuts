# DSL model for CLEVR
import torch
import torch.nn as nn
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from utils.dpl_loss import CLEVR_DPL
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


class ClevrDSLDPLRec(CBMModule):
    """DPL variant of DSL MODEL FOR CLEVR"""

    NAME = "clevrdsldplrec"

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
        super(ClevrDSLDPLRec, self).__init__(
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
        
        self.n_facts = n_facts
        self.nr_classes = nr_classes

        # opt and device
        self.opt = None
        self.device = get_device()
        self.weights_unary = torch.nn.Parameter(torch.randn([8, 3, 2, 2, 8], requires_grad=True).to(self.device))
                                                                # conc x 4
        self.weights_aggregation = torch.nn.Parameter(torch.randn([8, 8, 8, 8, 4], requires_grad=True).to(self.device))
        
    def get_rules_matrix(self, eval=False):
        return torch.max(torch.nn.functional.softmax(self.weights, dim=2), dim=2, keepdim=True)

    def get_pred_from_prob(self, pCs, presence=True):
        py_unary_predicate = torch.nn.functional.softmax(self.weights_unary, dim=-1)

        unary_probs = torch.stack([
            self._dpl_inference(pCs[:, i, 1:], py_unary_predicate)[0] for i in range(4)
        ], dim=1)

        unary_probs[:, :, 1] = pCs[:, :, 0]
        unary_probs[:, :, 0] = 1 - pCs[:, :, 0]

        py_agg_predicate = torch.nn.functional.softmax(self.weights_aggregation, dim=-1)

        label_prob, _ = self._dpl_inference_agg(
            unary_probs[:, 0], unary_probs[:, 1], unary_probs[:, 2], unary_probs[:, 3],
            py_agg_predicate, eval
        )

        return label_prob

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
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


        cs = cs.view(x.shape[0], n_images, self.n_facts)
        pCs = self.normalize_concepts(cs)

        py_unary_predicate = torch.nn.functional.softmax(self.weights_unary, dim=-1)
        py_agg_predicate = torch.nn.functional.softmax(self.weights_aggregation, dim=-1)

        # inference
        unary_prob1, _ = self._dpl_inference(pCs[:, 0, :], py_unary_predicate)
        unary_prob2, _ = self._dpl_inference(pCs[:, 1, :], py_unary_predicate)
        unary_prob3, _ = self._dpl_inference(pCs[:, 2, :], py_unary_predicate)
        unary_prob4, _ = self._dpl_inference(pCs[:, 3, :], py_unary_predicate)


        absence_mask = torch.all(pCs == -1, dim=-1)  # Shape: [128, 4]

        unary_prob1[absence_mask[:, 0], :] = 0
        unary_prob1[absence_mask[:, 0], 0] = 1
        unary_prob2[absence_mask[:, 1], :] = 0
        unary_prob2[absence_mask[:, 1], 0] = 1
        unary_prob3[absence_mask[:, 2], :] = 0
        unary_prob3[absence_mask[:, 2], 0] = 1
        unary_prob4[absence_mask[:, 3], :] = 0
        unary_prob4[absence_mask[:, 3], 0] = 1

        label_prob, label_predicted = self._dpl_inference_agg(
            unary_prob1,
            unary_prob2,
            unary_prob3,
            unary_prob4,
            py_agg_predicate,
            eval
        )

        pCs = pCs.view(pCs.shape[0], pCs.shape[1] * pCs.shape[2])

        return {"CS": cs, "YS": label_prob, "pCS": pCs, "PRED": label_predicted, "MUS": mus, "LOGVARS": logvars, "RECS": recs}


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

    def _dpl_inference(self, t, py, eval=False):
        colors, shapes, materials, sizes = t[:, :8], t[:, 8:11], t[:, 11:13], t[:, 13:15]
        y = torch.einsum("bi,bj,bk,bl->bijkl", colors, shapes, materials, sizes)
        y = y.reshape(t.shape[0], -1) @ py.view(-1, py.shape[-1])

        # Normalize
        y = (y + 1e-5) / (y.sum(dim=1, keepdim=True) + 1e-5)
        return y, torch.argmax(y, dim=-1)

    def _dpl_inference_agg(self, pred1, pred2, pred3, pred4, py, eval=False):
        y = torch.einsum("bi,bj,bk,bl->bijkl", pred1, pred2, pred3, pred4)
        y = y.reshape(pred1.shape[0], -1) @ py.view(-1, py.shape[-1])

        y = (y + 1e-5) / (y.sum(dim=1, keepdim=True) + 1e-5)
        return y, torch.argmax(y, dim=-1)

    def _old_dpl_inference_agg(self, pres1, pred1, pres2, pred2, pres3, pred3, pres4, pred4, py, eval=False):
        y = torch.einsum("bi,bj,bk,bl->bijkl", pred1, pred2, pred3, pred4)
        
        py_selected = py[pres1, :, pres2, :, pres3, :, pres4, :, :].view(-1, py.shape[-1])
        y = y.reshape(pred1.shape[0], -1) @ py_selected

        y = (y + 1e-5) / (y.sum(dim=1, keepdim=True) + 1e-5)
        return y, torch.argmax(y, dim=-1)

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