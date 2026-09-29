# DPL model for CLEVR
import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.deepproblog_modules import GraphSemiring
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import CLEVR_DPL
from models.utils.ops import outer_product


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

def build_unary_predicates(device):
    
    possible_worlds = list(product(range(8), range(3), range(2), range(2)))
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, 8))
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    predicate = torch.zeros(n_worlds, n_queries) # missing
    for w in range(n_worlds):
        color, shape, material, size = look_up[w]
        if shape == 0 and size == 0:
            predicate[w, 2] = 1
        elif shape == 2 and size == 0:
            predicate[w, 3] = 1    
        elif shape == 0 and size == 1 and material == 1:
            predicate[w, 4] = 1
        elif shape == 1 and size == 1:
            predicate[w, 5] = 1
        elif shape == 1 and size == 0 and color == 2:
            predicate[w, 6] = 1
        elif shape == 1 and size == 1 and color == 7:
            predicate[w, 7] = 1
        else:
            predicate[w, 1] = 1 # everything else
    return predicate.to(device)


def build_aggregation_predicate(device):
    possible_worlds = list(
        product(
            range(8), range(8), range(8), range(8)
        )
    )
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, 4))
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    aggregate = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        (
            predicate1, 
            predicate2,
            predicate3,
            predicate4,
        ) = look_up[w]

        if (
            (predicate1 == 2 and predicate2 == 3) or 
            (predicate1 == 3 and predicate2 == 2) or
            (predicate1 == 2 and predicate3 == 3) or
            (predicate1 == 3 and predicate3 == 2) or
            (predicate1 == 2 and predicate4 == 3) or
            (predicate1 == 3 and predicate4 == 2) or
            (predicate2 == 2 and predicate3 == 3) or
            (predicate2 == 3 and predicate3 == 2) or
            (predicate2 == 2 and predicate4 == 3) or
            (predicate2 == 3 and predicate4 == 2) or
            (predicate3 == 2 and predicate4 == 3) or
            (predicate3 == 3 and predicate4 == 2)
        ):
            aggregate[w, 0] = 1
        elif (
            (predicate1 == 4 and predicate2 == 5) or 
            (predicate1 == 5 and predicate2 == 4) or
            (predicate1 == 4 and predicate3 == 5) or
            (predicate1 == 5 and predicate3 == 4) or
            (predicate1 == 4 and predicate4 == 5) or
            (predicate1 == 5 and predicate4 == 4) or
            (predicate2 == 4 and predicate3 == 5) or
            (predicate2 == 5 and predicate3 == 4) or
            (predicate2 == 4 and predicate4 == 5) or
            (predicate2 == 5 and predicate4 == 4) or
            (predicate3 == 4 and predicate4 == 5) or
            (predicate3 == 5 and predicate4 == 4)
        ):
            aggregate[w, 1] = 1
        elif (
            (predicate1 == 6 and predicate2 == 7) or 
            (predicate1 == 7 and predicate2 == 6) or
            (predicate1 == 6 and predicate3 == 7) or
            (predicate1 == 7 and predicate3 == 6) or
            (predicate1 == 6 and predicate4 == 7) or
            (predicate1 == 7 and predicate4 == 6) or
            (predicate2 == 6 and predicate3 == 7) or
            (predicate2 == 7 and predicate3 == 6) or
            (predicate2 == 6 and predicate4 == 7) or
            (predicate2 == 7 and predicate4 == 6) or
            (predicate3 == 6 and predicate4 == 7) or
            (predicate3 == 7 and predicate4 == 6)
        ):
            aggregate[w, 2] = 1
        else:
            aggregate[w, 3] = 1

    return aggregate.to(device)


class CLEVRDPL(DeepProblogModel):
    """CLEVR MODEL FOR MNIST"""

    NAME = "clevrdpl"

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
        super(CLEVRDPL, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split
        self.n_facts = n_facts

        self.device = get_device()

        # Worlds-queries matrix
        self.unary_predicates = build_unary_predicates(self.device)
        self.agg_predicate = build_aggregation_predicate(self.device)

        # opt and device
        self.opt = None

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
        """
        n_images = x.shape[1]
        mask = (x == -1).all(-1).all(-1).all(-1)

        cs = torch.zeros((x.shape[0], n_images, 1, self.n_facts), device=x.device)

        for i in range(n_images):
            current_images = x[:, i]
            valid_indices = ~mask[:, i]
            if valid_indices.any():
                valid_predictions, _, _ = self.encoder(current_images[valid_indices])
                cs[valid_indices, i] = valid_predictions

        cs = cs.view(x.shape[0], n_images, self.n_facts)
        pCs = self.normalize_concepts(cs)
        py = self.problog_inference(pCs)

        pCs = pCs.view(pCs.shape[0], pCs.shape[1] * pCs.shape[2])

        return {"CS": cs, "YS": py, "pCS": pCs}

    def compute_unary_predicate(self, prob_img):

        colors, shapes, materials, sizes = prob_img[:, :8], prob_img[:, 8:11], prob_img[:, 11:13], prob_img[:, 13:15]

        res = colors.unsqueeze(2).multiply(shapes.unsqueeze(1)).view(prob_img.shape[0], -1)
        res = res.unsqueeze(2).multiply(materials.unsqueeze(1)).view(prob_img.shape[0], -1)
        res = res.unsqueeze(2).multiply(sizes.unsqueeze(1)).view(prob_img.shape[0], -1)

        predicates_prob = torch.zeros(size=(len(res), 8), device=prob_img.device)
        for i in range(8):
            predicates_prob[:, i] = self.compute_query(i, res, self.unary_predicates).view(-1)
    
        predicates_prob += 1e-5
        with torch.no_grad():
            Z = torch.sum(predicates_prob, dim=-1, keepdim=True)
        predicates_prob = predicates_prob / Z

        return predicates_prob

    def compute_aggregation(self, preds_1, preds_2, preds_3, preds_4):
        res = preds_1.unsqueeze(2).multiply(preds_2.unsqueeze(1)).view(preds_1.shape[0], -1)
        res = res.unsqueeze(2).multiply(preds_3.unsqueeze(1)).view(preds_1.shape[0], -1)
        res = res.unsqueeze(2).multiply(preds_4.unsqueeze(1)).view(preds_1.shape[0], -1)

        wq = torch.zeros(size=(len(res), 4), device=res.device)
        for i in range(4):
            wq[:, i] = self.compute_query(i, res, self.agg_predicate).view(-1)

        wq += 1e-5
        with torch.no_grad():
            Z = torch.sum(wq, dim=-1, keepdim=True)
        wq = wq / Z

        return wq

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
        prob_im = pCs[:, :4, :]  # Shape: [128, 4, N] - 4 components (prob_im1, prob_im2, prob_im3, prob_im4)

        # Apply compute_unary_predicate to all probabilities at once
        preds = torch.stack([self.compute_unary_predicate(prob_im[:, i, :]) for i in range(4)], dim=1)

        absence_mask = torch.all(pCs == 0, dim=-1)  # Shape: [128, 4]
        for img in range(preds.shape[1]):
            current_absence_mask = absence_mask[:, img]
            preds[current_absence_mask, img, :] = 0
            preds[current_absence_mask, img, 0] = 1

        # Compute query probability via aggregation
        query_prob = self.compute_aggregation(preds[:, 0, :], preds[:, 1, :], preds[:, 2, :], preds[:, 3, :])

        return query_prob


    def compute_query(self, query, worlds_prob, w_q):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        return torch.sum(w_q[:, query] * worlds_prob, dim=1, keepdim=True)

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents
            split (int, default=2): numbers of split

        Returns:
            vec: normalized concepts
        """
        # Split tensor into the respective components
        colors, shapes, materials, sizes = z[:, :, :8], z[:, :, 8:11], z[:, :, 11:13], z[:, :, 13:15]

        # Mask where z is non-zero (all components in the last dimension)
        mask = torch.all(z != 0, dim=-1)
        eps = 1e-5

        components = [colors, shapes, materials, sizes]
        for i, comp in enumerate(components):
            comp[mask] = nn.Softmax(dim=-1)(comp[mask]) + eps
            with torch.no_grad():
                Z = torch.sum(comp[mask], dim=-1, keepdim=True) 
            comp[mask] = comp[mask] / Z

        # Concatenate the components back together
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
        if args.dataset in ["clevr"]:
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

    # override
    def to(self, device):
        super().to(device)
        self.unary_predicates.to(device)
        self.agg_predicate.to(device)
