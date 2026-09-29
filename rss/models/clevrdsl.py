# DSL model for CLEVR
import torch
import torch.nn as nn
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from utils.dpl_loss import CLEVR_DPL
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


class ClevrDSL(CBMModule):
    """DSL MODEL FOR CLEVR"""

    NAME = "clevrdsl"

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
        super(ClevrDSL, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split
        
        self.n_facts = n_facts
        self.nr_classes = nr_classes

        # opt and device
        self.opt = None
        self.args = args
        self.device = get_device()                          # col, sha, siz, mat, pred
        # self.weights_unary = torch.nn.Parameter(torch.randn([8, 3, 2, 2, 7], requires_grad=True).to(self.device))
        #                                                            # pres, conc x 4
        # self.weights_aggregation = torch.nn.Parameter(torch.randn([2, 7, 2, 7, 2, 7, 2, 7, self.nr_classes], requires_grad=True).to(self.device))
        self.epsilon_digits = args.eps_sym
        self.epsilon_rules = args.eps_rul
        # new redefinition
        self.weights_unary = torch.nn.Parameter(torch.randn([8, 3, 2, 2, 8], requires_grad=True).to(self.device)) # 0 is missing
                                                                   # pres, conc x 4
        self.weights_aggregation = torch.nn.Parameter(torch.randn([8, 8, 8, 8, self.nr_classes], requires_grad=True).to(self.device))
        


    def epsilon_greedy(self, t, eval, dim=1):
        colors, shapes, materials, sizes = t[:, :8], t[:, 8:11], t[:, 11:13], t[:, 13:15]

        if eval:
            (truth_values_1, chosen_symbols_1) = torch.max(colors, dim=dim)
            (truth_values_2, chosen_symbols_2) = torch.max(shapes, dim=dim)
            (truth_values_3, chosen_symbols_3) = torch.max(materials, dim=dim)
            (truth_values_4, chosen_symbols_4) = torch.max(sizes, dim=dim)

            return (
                truth_values_1, chosen_symbols_1,
                truth_values_2, chosen_symbols_2,
                truth_values_3, chosen_symbols_3,
                truth_values_4, chosen_symbols_4
            )
        else:
            random_selection = torch.rand((t.shape[0],)) < self.epsilon_digits
            random_selection = random_selection.to(self.device)
            
            # 1
            color_index_random = torch.randint(colors.shape[1], (t.shape[0],))
            color_index_random = color_index_random.to(self.device)
            _, color_index_max = torch.max(colors, dim=dim)

            chosen_color = torch.where(random_selection, color_index_random, color_index_max)
            color_truth_values = torch.gather(colors, dim, chosen_color.view(-1, 1))

            # 2 shapes
            shapes_index_random = torch.randint(shapes.shape[1], (shapes.shape[0],))
            shapes_index_random = shapes_index_random.to(self.device)
            _, shapes_index_max = torch.max(shapes, dim=dim)

            chosen_shapes = torch.where(random_selection, shapes_index_random, shapes_index_max)
            shapes_truth_values = torch.gather(shapes, dim, chosen_shapes.view(-1, 1))

            # 3 materials
            materials_index_random = torch.randint(materials.shape[1], (materials.shape[0],))
            materials_index_random = materials_index_random.to(self.device)
            _, materials_index_max = torch.max(materials, dim=dim)

            chosen_materials = torch.where(random_selection, materials_index_random, materials_index_max)
            materials_truth_values = torch.gather(materials, dim, chosen_materials.view(-1, 1))

            # 4 sizes
            sizes_index_random = torch.randint(sizes.shape[1], (sizes.shape[0],))
            sizes_index_random = sizes_index_random.to(self.device)
            _, sizes_index_max = torch.max(sizes, dim=dim)

            chosen_sizes = torch.where(random_selection, sizes_index_random, sizes_index_max)
            sizes_truth_values = torch.gather(sizes, dim, chosen_sizes.view(-1, 1))

            return (
                color_truth_values, chosen_color,
                shapes_truth_values, chosen_shapes,
                materials_truth_values, chosen_materials,
                sizes_truth_values, chosen_sizes
            )

    def get_unary_predicate_rules_matrix(self, weights, eval):
        if eval:
            return torch.max(torch.nn.functional.softmax(weights, dim=-1), dim=-1, keepdim=True)
        else:
            n_colors = weights.shape[0]
            n_shape = weights.shape[1]
            n_materials = weights.shape[2]
            n_sizes = weights.shape[3]
            n_output_symbols = weights.shape[4]
            random_selection = torch.rand((n_colors, n_shape, n_materials, n_sizes)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, (n_colors, n_shape, n_materials, n_sizes))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(weights, dim=-1)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            truth_values = torch.gather(torch.nn.functional.softmax(weights, dim=-1),
                                        -1, chosen_symbols.view(n_colors, n_shape, n_materials, n_sizes, 1)).view(n_colors, n_shape, n_materials, n_sizes)


            return truth_values, chosen_symbols


    def old_epsilon_greedy_aggregation(self, pres1, pred1, pres2, pred2, pres3, pred3, pres4, pred4, eval, dim=1):

        if eval:
            (truth_values_1, chosen_symbols_1) = torch.max(pred1, dim=dim)
            (truth_values_pres_1, chosen_symbols_pres_1) = torch.max(pres1, dim=0)

            (truth_values_2, chosen_symbols_2) = torch.max(pred2, dim=dim)
            (truth_values_pres_2, chosen_symbols_pres_2) = torch.max(pres2, dim=0)

            (truth_values_3, chosen_symbols_3) = torch.max(pred3, dim=dim)
            (truth_values_pres_3, chosen_symbols_pres_3) = torch.max(pres3, dim=0)

            (truth_values_4, chosen_symbols_4) = torch.max(pred4, dim=dim)
            (truth_values_pres_4, chosen_symbols_pres_4) = torch.max(pres4, dim=0)

            return (
                truth_values_1, chosen_symbols_1,
                truth_values_pres_1, chosen_symbols_pres_1,

                truth_values_2, chosen_symbols_2,
                truth_values_pres_2, chosen_symbols_pres_2,

                truth_values_3, chosen_symbols_3,
                truth_values_pres_3, chosen_symbols_pres_3,

                truth_values_4, chosen_symbols_4,
                truth_values_pres_4, chosen_symbols_pres_4
            )
        else:
            random_selection = torch.rand((pres1.shape[0],)) < self.epsilon_digits
            random_selection = random_selection.to(self.device)
            
            # presence
            (truth_values_pres_1, chosen_symbols_pres_1) = torch.max(pres1, dim=0)
            (truth_values_pres_2, chosen_symbols_pres_2) = torch.max(pres2, dim=0)
            (truth_values_pres_3, chosen_symbols_pres_3) = torch.max(pres3, dim=0)
            (truth_values_pres_4, chosen_symbols_pres_4) = torch.max(pres4, dim=0)
            
            # 1
            first_index_random = torch.randint(pred1.shape[1], (pred1.shape[0],))
            first_index_random = first_index_random.to(self.device)
            _, first_index_max = torch.max(pred1, dim=dim)

            chosen_first = torch.where(random_selection, first_index_random, first_index_max)
            first_truth_values = torch.gather(pred1, dim, chosen_first.view(-1, 1))

            # 2
            second_index_random = torch.randint(pred2.shape[1], (pred2.shape[0],))
            second_index_random = second_index_random.to(self.device)
            _, second_index_max = torch.max(pred2, dim=dim)

            chosen_second = torch.where(random_selection, second_index_random, second_index_max)
            second_truth_values = torch.gather(pred2, dim, chosen_second.view(-1, 1))

            # 3
            third_index_random = torch.randint(pred3.shape[1], (pred3.shape[0],))
            third_index_random = third_index_random.to(self.device)
            _, third_index_max = torch.max(pred3, dim=dim)

            chosen_third = torch.where(random_selection, third_index_random, third_index_max)
            third_truth_values = torch.gather(pred3, dim, chosen_third.view(-1, 1))

            # 4 sizes
            fourth_index_random = torch.randint(pred4.shape[1], (pred4.shape[0],))
            fourth_index_random = fourth_index_random.to(self.device)
            _, fourth_index_max = torch.max(pred4, dim=dim)

            chosen_fourth = torch.where(random_selection, fourth_index_random, fourth_index_max)
            fourth_truth_values = torch.gather(pred4, dim, chosen_fourth.view(-1, 1))

            return (
                first_truth_values, chosen_first,
                truth_values_pres_1, chosen_symbols_pres_1,
                second_truth_values, chosen_second,
                truth_values_pres_2, chosen_symbols_pres_2,
                third_truth_values, chosen_third,
                truth_values_pres_3, chosen_symbols_pres_3,
                fourth_truth_values, chosen_fourth,
                truth_values_pres_4, chosen_symbols_pres_4,
            )

    def epsilon_greedy_aggregation(self, pred1, pred2, pred3, pred4, eval, dim=1):

        if eval:
            (truth_values_1, chosen_symbols_1) = torch.max(pred1, dim=dim)
            (truth_values_2, chosen_symbols_2) = torch.max(pred2, dim=dim)
            (truth_values_3, chosen_symbols_3) = torch.max(pred3, dim=dim)
            (truth_values_4, chosen_symbols_4) = torch.max(pred4, dim=dim)

            return (
                truth_values_1, chosen_symbols_1,
                truth_values_2, chosen_symbols_2,
                truth_values_3, chosen_symbols_3,
                truth_values_4, chosen_symbols_4,
            )
        else:
            random_selection = torch.rand((pred1.shape[0],)) < self.epsilon_digits
            random_selection = random_selection.to(self.device)
            
            # 1
            first_index_random = torch.randint(pred1.shape[1], (pred1.shape[0],))
            first_index_random = first_index_random.to(self.device)
            _, first_index_max = torch.max(pred1, dim=dim)

            chosen_first = torch.where(random_selection, first_index_random, first_index_max)
            first_truth_values = torch.gather(pred1, dim, chosen_first.view(-1, 1))

            # 2
            second_index_random = torch.randint(pred2.shape[1], (pred2.shape[0],))
            second_index_random = second_index_random.to(self.device)
            _, second_index_max = torch.max(pred2, dim=dim)

            chosen_second = torch.where(random_selection, second_index_random, second_index_max)
            second_truth_values = torch.gather(pred2, dim, chosen_second.view(-1, 1))

            # 3
            third_index_random = torch.randint(pred3.shape[1], (pred3.shape[0],))
            third_index_random = third_index_random.to(self.device)
            _, third_index_max = torch.max(pred3, dim=dim)

            chosen_third = torch.where(random_selection, third_index_random, third_index_max)
            third_truth_values = torch.gather(pred3, dim, chosen_third.view(-1, 1))

            # 4 sizes
            fourth_index_random = torch.randint(pred4.shape[1], (pred4.shape[0],))
            fourth_index_random = fourth_index_random.to(self.device)
            _, fourth_index_max = torch.max(pred4, dim=dim)

            chosen_fourth = torch.where(random_selection, fourth_index_random, fourth_index_max)
            fourth_truth_values = torch.gather(pred4, dim, chosen_fourth.view(-1, 1))

            return (
                first_truth_values, chosen_first,
                second_truth_values, chosen_second,
                third_truth_values, chosen_third,
                fourth_truth_values, chosen_fourth,
            )

    def get_old_aggregation_predicate_rules_matrix(self, weights, eval):
        if eval:
            return torch.max(torch.nn.functional.softmax(weights, dim=-1), dim=-1, keepdim=True)
        else:
            n_pres_1 = weights.shape[0]
            n_pred1 = weights.shape[1]
            n_pres_2 = weights.shape[2]
            n_pred2 = weights.shape[3]
            n_pres_3 = weights.shape[4]
            n_pred3 = weights.shape[5]
            n_pres_4 = weights.shape[6]
            n_pred4 = weights.shape[7]
            n_output_symbols = weights.shape[8]
            random_selection = torch.rand((n_pres_1, n_pred1, n_pres_2, n_pred2, n_pres_3, n_pred3, n_pres_4, n_pred4)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, (n_pres_1, n_pred1, n_pres_2, n_pred2, n_pres_3, n_pred3, n_pres_4, n_pred4))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(weights, dim=-1)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            truth_values = torch.gather(torch.nn.functional.softmax(weights, dim=-1),
                                        -1, chosen_symbols.view(n_pres_1, n_pred1, n_pres_2, n_pred2, n_pres_3, n_pred3, n_pres_4, n_pred4, 1)).view(n_pres_1, n_pred1, n_pres_2, n_pred2, n_pres_3, n_pred3, n_pres_4, n_pred4)


            return truth_values, chosen_symbols

    def get_aggregation_predicate_rules_matrix(self, weights, eval):
        if eval:
            return torch.max(torch.nn.functional.softmax(weights, dim=-1), dim=-1, keepdim=True)
        else:
            n_pred1 = weights.shape[0]
            n_pred2 = weights.shape[1]
            n_pred3 = weights.shape[2]
            n_pred4 = weights.shape[3]
            n_output_symbols = weights.shape[4]
            random_selection = torch.rand((n_pred1, n_pred2, n_pred3, n_pred4)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, (n_pred1, n_pred2, n_pred3, n_pred4))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(weights, dim=-1)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            truth_values = torch.gather(torch.nn.functional.softmax(weights, dim=-1),
                                        -1, chosen_symbols.view(n_pred1, n_pred2, n_pred3, n_pred4, 1)).view(n_pred1, n_pred2, n_pred3, n_pred4)


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
        cs = []
        n_images = len(x[0])

        for i in range(n_images):
            current_images = x[:, i]

            mask = (current_images == -1).all(dim=1).all(dim=1).all(dim=1)

            lc = torch.zeros((current_images.shape[0], 1, self.n_facts), device=x.device)  # Placeholder for all predictions

            if not mask.all():  # If there are valid images
                valid_indices = ~mask  # Get valid image indices
                valid_predictions, _, _ = self.encoder(current_images[valid_indices])  # Process valid images
                lc[valid_indices] = valid_predictions  # Assign predictions to valid indices

            cs.append(lc)  # Add predictions (with zeros for missing images)
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)

        # get the result of the inference via DSL
        unary_rules_weights, unary_g_matrix = self.get_unary_predicate_rules_matrix(self.weights_unary, eval)

        (
            color_truth_values_1, chosen_color_1,
            shapes_truth_values_1, chosen_shapes_1,
            materials_truth_values_1, chosen_materials_1,
            sizes_truth_values_1, chosen_sizes_1
        ) = self.epsilon_greedy(pCs[:, 0, :], eval)

        (
            color_truth_values_2, chosen_color_2,
            shapes_truth_values_2, chosen_shapes_2,
            materials_truth_values_2, chosen_materials_2,
            sizes_truth_values_2, chosen_sizes_2
        ) = self.epsilon_greedy(pCs[:, 1, :], eval)

        (
            color_truth_values_3, chosen_color_3,
            shapes_truth_values_3, chosen_shapes_3,
            materials_truth_values_3, chosen_materials_3,
            sizes_truth_values_3, chosen_sizes_3
        ) = self.epsilon_greedy(pCs[:, 2, :], eval)

        (
            color_truth_values_4, chosen_color_4,
            shapes_truth_values_4, chosen_shapes_4,
            materials_truth_values_4, chosen_materials_4,
            sizes_truth_values_4, chosen_sizes_4
        ) = self.epsilon_greedy(pCs[:, 3, :], eval)
        
        # compute the presence
        absence_mask = torch.all(pCs == -1, dim=-1)  # Shape: [128, 4]

        first_pred_prob = self.weights_unary[chosen_color_1, chosen_shapes_1, chosen_materials_1, chosen_sizes_1]
        first_pred_prob[absence_mask[:, 0], :] = 0
        first_pred_prob[absence_mask[:, 0], 0] = 1

        second_pred_prob = self.weights_unary[chosen_color_2, chosen_shapes_2, chosen_materials_2, chosen_sizes_2]
        second_pred_prob[absence_mask[:, 1], :] = 0
        second_pred_prob[absence_mask[:, 1], 0] = 1

        third_pred_prob = self.weights_unary[chosen_color_3, chosen_shapes_3, chosen_materials_3, chosen_sizes_3]
        third_pred_prob[absence_mask[:, 2], :] = 0
        third_pred_prob[absence_mask[:, 2], 0] = 1

        fourth_pred_prob = self.weights_unary[chosen_color_4, chosen_shapes_4, chosen_materials_4, chosen_sizes_4]
        fourth_pred_prob[absence_mask[:, 3], :] = 0
        fourth_pred_prob[absence_mask[:, 3], 0] = 1

        aggregation_rules_weights, aggregation_g_matrix = self.get_aggregation_predicate_rules_matrix(self.weights_aggregation, eval)

        (
            first_truth_values, chosen_first,
            second_truth_values, chosen_second,
            third_truth_values, chosen_third,
            fourth_truth_values, chosen_fourth,
        ) = self.epsilon_greedy_aggregation(
            first_pred_prob,
            second_pred_prob,
            third_pred_prob,
            fourth_pred_prob,
            eval
        )

        symbols_truth_values = torch.concat(
            [unary_rules_weights[chosen_color_1, chosen_shapes_1, chosen_materials_1, chosen_sizes_1].view(-1, 1),
            color_truth_values_1.view(-1, 1),
            shapes_truth_values_1.view(-1, 1),
            materials_truth_values_1.view(-1, 1),
            sizes_truth_values_1.view(-1, 1),
            unary_rules_weights[chosen_color_2, chosen_shapes_2, chosen_materials_2, chosen_sizes_2].view(-1, 1),
            color_truth_values_2.view(-1, 1),
            shapes_truth_values_2.view(-1, 1),
            materials_truth_values_2.view(-1, 1),
            sizes_truth_values_2.view(-1, 1),
            unary_rules_weights[chosen_color_3, chosen_shapes_3, chosen_materials_3, chosen_sizes_3].view(-1, 1),
            color_truth_values_3.view(-1, 1),
            shapes_truth_values_3.view(-1, 1),
            materials_truth_values_3.view(-1, 1),
            sizes_truth_values_3.view(-1, 1),
            unary_rules_weights[chosen_color_4, chosen_shapes_4, chosen_materials_4, chosen_sizes_4].view(-1, 1),
            color_truth_values_4.view(-1, 1),
            shapes_truth_values_4.view(-1, 1),
            materials_truth_values_4.view(-1, 1),
            sizes_truth_values_4.view(-1, 1),
            aggregation_rules_weights[chosen_first, chosen_second, chosen_third, chosen_fourth].view(-1, 1),
            first_truth_values.view(-1, 1),
            second_truth_values.view(-1, 1),
            third_truth_values.view(-1, 1),
            fourth_truth_values.view(-1, 1)
        ], dim=1)

        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)    
        py = predictions_truth_values

        pred = aggregation_g_matrix[chosen_first, chosen_second, chosen_third, chosen_fourth]

        pCs = pCs.view(pCs.shape[0], pCs.shape[1] * pCs.shape[2])

        return {"CS": cs, "YS": py, "pCS": pCs, "PRED": pred, "KNOWLEDGE": torch.nn.functional.softmax(self.weights_aggregation, dim=-1)}

        
    def old_forward(self, x, eval=False):
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

            if not mask.all():  # If there are valid images
                valid_indices = ~mask  # Get valid image indices
                valid_predictions, _, _ = self.encoder(current_images[valid_indices])  # Process valid images
                lc[valid_indices] = valid_predictions  # Assign predictions to valid indices

            cs.append(lc)  # Add predictions (with zeros for missing images)
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)

        # get the result of the inference via DSL
        unary_rules_weights, unary_g_matrix = self.get_unary_predicate_rules_matrix(self.weights_unary, eval)

        (
            color_truth_values_1, chosen_color_1,
            shapes_truth_values_1, chosen_shapes_1,
            materials_truth_values_1, chosen_materials_1,
            sizes_truth_values_1, chosen_sizes_1
        ) = self.epsilon_greedy(pCs[:, 0, :], eval)

        (
            color_truth_values_2, chosen_color_2,
            shapes_truth_values_2, chosen_shapes_2,
            materials_truth_values_2, chosen_materials_2,
            sizes_truth_values_2, chosen_sizes_2
        ) = self.epsilon_greedy(pCs[:, 1, :], eval)

        (
            color_truth_values_3, chosen_color_3,
            shapes_truth_values_3, chosen_shapes_3,
            materials_truth_values_3, chosen_materials_3,
            sizes_truth_values_3, chosen_sizes_3
        ) = self.epsilon_greedy(pCs[:, 2, :], eval)

        (
            color_truth_values_4, chosen_color_4,
            shapes_truth_values_4, chosen_shapes_4,
            materials_truth_values_4, chosen_materials_4,
            sizes_truth_values_4, chosen_sizes_4
        ) = self.epsilon_greedy(pCs[:, 3, :], eval)
        
        # compute the presence
        presence_mask = torch.all(pCs != -1, dim=-1).float()  # Shape: [128, 4]

        aggregation_rules_weights, aggregation_g_matrix = self.get_aggregation_predicate_rules_matrix(self.weights_aggregation, eval)

        (
            first_truth_values, chosen_first,
            _, chosen_symbols_pres_1,
            second_truth_values, chosen_second,
            _, chosen_symbols_pres_2,
            third_truth_values, chosen_third,
            _, chosen_symbols_pres_3,
            fourth_truth_values, chosen_fourth,
            _, chosen_symbols_pres_4,
        ) = self.epsilon_greedy_aggregation(
            presence_mask[:, 0],
            self.weights_unary[chosen_color_1, chosen_shapes_1, chosen_materials_1, chosen_sizes_1],
            presence_mask[:, 1],
            self.weights_unary[chosen_color_2, chosen_shapes_2, chosen_materials_2, chosen_sizes_2],
            presence_mask[:, 2],
            self.weights_unary[chosen_color_3, chosen_shapes_3, chosen_materials_3, chosen_sizes_3],
            presence_mask[:, 3],
            self.weights_unary[chosen_color_4, chosen_shapes_4, chosen_materials_4, chosen_sizes_4],
            eval
        )

        symbols_truth_values = torch.concat(
            [unary_rules_weights[chosen_color_1, chosen_shapes_1, chosen_materials_1, chosen_sizes_1].view(-1, 1),
            color_truth_values_1.view(-1, 1),
            shapes_truth_values_1.view(-1, 1),
            materials_truth_values_1.view(-1, 1),
            sizes_truth_values_1.view(-1, 1),
            unary_rules_weights[chosen_color_2, chosen_shapes_2, chosen_materials_2, chosen_sizes_2].view(-1, 1),
            color_truth_values_2.view(-1, 1),
            shapes_truth_values_2.view(-1, 1),
            materials_truth_values_2.view(-1, 1),
            sizes_truth_values_2.view(-1, 1),
            unary_rules_weights[chosen_color_3, chosen_shapes_3, chosen_materials_3, chosen_sizes_3].view(-1, 1),
            color_truth_values_3.view(-1, 1),
            shapes_truth_values_3.view(-1, 1),
            materials_truth_values_3.view(-1, 1),
            sizes_truth_values_3.view(-1, 1),
            unary_rules_weights[chosen_color_4, chosen_shapes_4, chosen_materials_4, chosen_sizes_4].view(-1, 1),
            color_truth_values_4.view(-1, 1),
            shapes_truth_values_4.view(-1, 1),
            materials_truth_values_4.view(-1, 1),
            sizes_truth_values_4.view(-1, 1),
            aggregation_rules_weights[chosen_symbols_pres_1, chosen_first, chosen_symbols_pres_2, chosen_second, chosen_symbols_pres_3, chosen_third, chosen_symbols_pres_4, chosen_fourth].view(-1, 1),
            first_truth_values.view(-1, 1),
            second_truth_values.view(-1, 1),
            third_truth_values.view(-1, 1),
            fourth_truth_values.view(-1, 1)
        ], dim=1)

        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)    
        py = predictions_truth_values

        pred = aggregation_g_matrix[chosen_symbols_pres_1, chosen_first, chosen_symbols_pres_2, chosen_second, chosen_symbols_pres_3, chosen_third, chosen_symbols_pres_4, chosen_fourth]

        pCs = pCs.view(pCs.shape[0], pCs.shape[1] * pCs.shape[2])

        return {"CS": cs, "YS": py, "pCS": pCs, "PRED": pred, "KNOWLEDGE": torch.nn.functional.softmax(self.weights_aggregation, dim=-1)}


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
            "clevr"
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
        self.opt = madgrad.MADGRAD(
            [{'params': list(self.parameters())[:1]}, {'params': list(self.parameters())[1:], 'lr': 1e-3}], lr=args.lr
        )