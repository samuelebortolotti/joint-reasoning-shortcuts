from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from utils.losses import clevr_logic
from utils.boia_knowledge import get_boia_actions_from_concepts

def build_confusion_matrix(preds1, preds2, n_classes):
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(preds1.shape[0]):
        confusion_matrix[preds1[i].item(), preds2[i].item()] += 1
    return confusion_matrix

def permutation_matrix_from_predictions(preds1, preds2, n_classes):
    confusion_matrix = build_confusion_matrix(preds1, preds2, n_classes)

    row_indices, col_indices = linear_sum_assignment(confusion_matrix, maximize=True)

    perm_matrix = np.zeros((n_classes, n_classes), dtype=np.float32)
    perm_matrix[row_indices, col_indices] = 1

    return torch.tensor(perm_matrix)

def build_boia_confusion_matrix(preds, gts, n_concepts):
    confusion = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    for i in range(n_concepts):
        for j in range(n_concepts):
            confusion[i, j] = np.sum((preds[:, i] == 1) & (gts[:, j] == 1))
    return confusion

def find_boia_permutation(preds, gts, n_concepts):
    confusion_matrix = build_boia_confusion_matrix(preds, gts, n_concepts)
    _, col_indices = linear_sum_assignment(confusion_matrix, maximize=True)
    return col_indices

def rearrange_predictions_with_confusion_clevr(pred, gt):
    N, _, _ = pred.shape

    global_cost_matrix = np.zeros((4, 4))

    for i in range(N):

        local_cost_matrix = np.zeros((4, 4))
        for j in range(4):
            for k in range(4):
                matches = (pred[i, :, j] == gt[i, :, k]) & (gt[i, :, k] != -1)
                local_cost_matrix[j, k] = matches.sum()

        global_cost_matrix += local_cost_matrix

    row_ind, col_ind = linear_sum_assignment(-global_cost_matrix)
    pred_rearranged = np.copy(pred)
    for i in range(N):
        pred_rearranged[i] = pred[i, :, col_ind]

    return pred_rearranged, col_ind


def get_hungarian_permutation(predicted_concepts, true_concepts, dataset_name):
    if dataset_name == "clevr":

        _, perm_idx = rearrange_predictions_with_confusion_clevr(predicted_concepts, true_concepts)
        
        perm_color = permutation_matrix_from_predictions(
            predicted_concepts[:, :, 0].flatten(), true_concepts[:, :, 0].flatten(), 8
        ).numpy()
        perm_shapes = permutation_matrix_from_predictions(
            predicted_concepts[:, :, 1].flatten(), true_concepts[:, :, 1].flatten(), 3
        ).numpy()
        perm_material = permutation_matrix_from_predictions(
            predicted_concepts[:, :, 2].flatten(), true_concepts[:, :, 2].flatten(), 2
        ).numpy()
        perm_sizes = permutation_matrix_from_predictions(
            predicted_concepts[:, :, 3].flatten(), true_concepts[:, :, 3].flatten(), 2
        ).numpy()

        return (perm_idx, perm_color, perm_shapes, perm_material, perm_sizes)
    elif dataset_name == "boia":
        return find_boia_permutation(
            predicted_concepts, true_concepts, 21
        )
    else:
        return permutation_matrix_from_predictions(
            predicted_concepts, true_concepts, 10
        ).numpy()

def get_gt_knowledge(args):
    w = []
    for i in range(10):
        for j in range(10):
            if args.task == "sumparity" or args.task == "sumparityrigged":
                w.append((i + j) % 2)
            elif args.task == "addition":
                w.append(i + j)
            else:
                raise NotImplementedError()
    return np.array(w)

def get_cbm_knowledge(w, device, args):
    knowledge = torch.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            x = torch.nn.functional.one_hot(torch.tensor([i]), num_classes=10).float().to(device)
            y = torch.nn.functional.one_hot(torch.tensor([j]), num_classes=10).float().to(device)
            if args.multi_linear:
                xy = x.unsqueeze(2).multiply(y.unsqueeze(1)).view(x.shape[0], -1)
            else:
                xy = torch.cat([x, y], dim=-1)
            knowledge[i, j] = torch.argmax(w(xy), dim=-1)

    return knowledge

def get_senn_knowledge(model_relevance, device, args, dataset):
    n_classes = 19 if args.task == "addition" else 2

    knowledge = torch.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            g1 = torch.nn.functional.one_hot(torch.tensor([i]), num_classes=10).float().to(device)
            g2 = torch.nn.functional.one_hot(torch.tensor([j]), num_classes=10).float().to(device)
            g = g1.unsqueeze(2).multiply(g2.unsqueeze(1)).view(g1.shape[0], -1).unsqueeze(2)
            g = g.repeat(5, 1, 1)

            x = dataset.get_instance(i, j, num_samples=5).to(device)
            xs = x.chunk(2, dim=-1)

            ks = [model_relevance(xi)[0] for xi in xs]
            ks = torch.stack(ks, dim=1) if ks[0].dim() == 2 else torch.cat(ks, dim=1)
            ks = ks.reshape(ks.shape[0], ks.shape[1], 10, n_classes)
            wx = ks[:, 0, :].unsqueeze(2).multiply(ks[:, 1, :].unsqueeze(1))
            wx = wx.reshape(wx.shape[0], 10 ** 2, n_classes)

            dot = torch.sum(wx * g, dim=1)
            py = torch.nn.functional.softmax(dot, dim=-1).mean(0)

            knowledge[i, j] = torch.argmax(py, dim=-1)
    return knowledge


def low_taper_fade(model, pi, is_cbm, args, dataset=None,index=None):
    if is_cbm:
        w = get_cbm_knowledge(model.classifier, model.device, args)
    else:
        if "senn" in args.model:
            w = get_senn_knowledge(model.relevance_score, model.device, args, dataset)
        else:
            w = torch.argmax(torch.nn.functional.softmax(model.weights, dim=2), dim=2)
    w_aligned = np.dot(pi.T, np.dot(w.cpu().numpy(), pi)).flatten()
    w_gt = get_gt_knowledge(args).flatten()

    mask = torch.zeros(100).to(torch.bool)
    for a in range(10):
        for b in range(10):
                if a % 2 == 0 and b % 2 == 1:
                    if index == 1:
                        mask[a*10 + b] = True
                else:
                    if index == 0:
                        mask[a*10 + b] = True

    if args.task == "sumparityrigged":
        w_gt, w_aligned = w_gt[mask], w_aligned[mask]

    print(w_gt.shape, w_aligned.shape)

    w_acc = accuracy_score(w_gt, w_aligned)
    w_f1 = f1_score(w_gt, w_aligned, average="macro")

    return w_acc, w_f1

def retrive_concepts_and_labels_hungarian(predicted_concepts, true_concepts, perm_matrix, dataset_name):

    if dataset_name == "clevr":

        (_, perm_color, perm_shapes, perm_material, perm_sizes) = perm_matrix

        predicted_colors = perm_color[predicted_concepts[:, :, 0]]
        predicted_colors = np.argmax(predicted_colors, axis=-1)

        predicted_shapes = perm_shapes[predicted_concepts[:, :, 1]]
        predicted_shapes = np.argmax(predicted_shapes, axis=-1)

        predicted_materials = perm_material[predicted_concepts[:, :, 2]]
        predicted_materials = np.argmax(predicted_materials, axis=-1)

        predicted_sizes = perm_sizes[predicted_concepts[:, :, 3]]
        predicted_sizes = np.argmax(predicted_sizes, axis=-1)

        predicted_concepts = np.stack(
            [predicted_colors, predicted_shapes, predicted_materials, predicted_sizes],
            axis=-1
        )

    elif dataset_name == "boia":
        predicted_concepts = predicted_concepts[:, perm_matrix]
    else:
        predicted_concepts = perm_matrix[predicted_concepts]
        predicted_concepts = np.argmax(predicted_concepts, axis=1)

    return true_concepts, predicted_concepts  

def compute_nll(predictions, targets):
    loss_fn = torch.nn.NLLLoss(reduction="mean")
    return loss_fn(torch.tensor(predictions).log(), torch.tensor(targets).long()).item()

def compute_concept_collapse(true_concepts, predicted_concepts, multilabel=False):
    return 1 - compute_coverage(confusion_matrix(true_concepts, predicted_concepts))

def compute_coverage(confusion_matrix):
    max_values = np.max(confusion_matrix, axis=0)
    clipped_values = np.clip(max_values, 0, 1)

    coverage = np.sum(clipped_values) / len(clipped_values)
    return coverage

def get_concepts_label_clevr():
    samples, labels = [], []

    for _ in range(100):
        presence = np.random.randint(0, 2, size=4)
        color = np.random.randint(0, 8, size=4)
        shape = np.random.randint(0, 3, size=4)
        material = np.random.randint(0, 2, size=4)
        size = np.random.randint(0, 2, size=4)
        
        objects = np.stack([presence, color, shape, material, size], axis=-1)
        y = clevr_logic(objects)
        labels.append(torch.tensor([y]).to("cpu").long())

        logits = []
        for obj in objects:
            l = []
            presence, color, shape, material, size = torch.tensor(obj)

            l.append(torch.tensor([presence]).to("cpu").float())
            l.append(torch.nn.functional.one_hot(color, 8).to("cpu").float())
            l.append(torch.nn.functional.one_hot(shape, 3).to("cpu").float())
            l.append(torch.nn.functional.one_hot(material, 2).to("cpu").float())
            l.append(torch.nn.functional.one_hot(size, 2).to("cpu").float())

            l = torch.cat(l, dim=0)
            logits.append(l)
        logits = torch.stack(logits, dim=1)
        samples.append(logits)

    samples = torch.stack(samples, dim=0)
    labels = torch.cat(labels, dim=0)
    return samples, labels


def sample_boia_config():
    config = torch.randint(0, 2, (1000, 21))
    labels = get_boia_actions_from_concepts(config)
    return config, labels


def evaluate_knowledge_boia(model, pi, model_name):
    def invert_permutation(permutation):
        inv = np.zeros_like(permutation)
        inv[permutation] = np.arange(len(permutation))
        return inv
    
    sampled_configurations, labels = sample_boia_config()

    inv_perm = invert_permutation(pi)
    predicted = sampled_configurations[:, inv_perm]

    y = model.get_pred_from_prob(predicted.to(model.device).to(torch.float32), False).detach().cpu()

    if 'cbm' in model_name:
        y = (y > 0.5).long()
    else:
        y_pred_split = torch.split(y, 2, dim=1)
        y = torch.stack([pred.argmax(dim=1) for pred in y_pred_split], dim=1)

    acc = accuracy_score(labels.numpy().flatten(), y.numpy().flatten())
    f1 = f1_score(labels.numpy().flatten(), y.numpy().flatten(), average="macro")
    
    return acc, f1


def evaluate_knowledge_clevr(model, pi):
    samples, labels = get_concepts_label_clevr()

    perm_idx, perm_color, perm_shapes, perm_material, perm_sizes = pi
    perm_idx, perm_color, perm_shapes, perm_material, perm_sizes = perm_idx.astype(int), perm_color.astype(int) , perm_shapes.astype(int) , perm_material.astype(int) , perm_sizes.astype(int) 

    true_concepts = torch.tensor(samples)
    full_concept_vector = torch.zeros((true_concepts.shape[0], 4, 16), dtype=float)

    for idx in range(4):
        concept_img = true_concepts[:, idx, :]
        mask = (concept_img == -1).all(dim=1)
        mask = ~mask

        concept_vector = torch.zeros((concept_img.shape[0], 16), dtype=float)
    
        colors = concept_img[:, 0].to(int)
        shapes = concept_img[:, 1].to(int)
        materials = concept_img[:, 2].to(int)
        sizes = concept_img[:, 3].to(int)
        
        if mask.sum() != 0:
            concept_vector[mask, 0] = torch.tensor(mask, dtype=float)
            concept_vector[mask, 1:9] = torch.tensor(perm_color.T[colors[mask]], dtype=float)
            concept_vector[mask, 9:12] = torch.tensor(perm_shapes.T[shapes[mask]], dtype=float)
            concept_vector[mask, 12:14] = torch.tensor(perm_material.T[materials[mask]], dtype=float)
            concept_vector[mask, 14:] = torch.tensor(perm_sizes.T[sizes[mask]], dtype=float)
        else:
            concept_vector[:, 0] = torch.tensor(mask, dtype=float)
            
        full_concept_vector[:, idx, :] = concept_vector
        
    y = model.get_pred_from_prob(full_concept_vector.to(model.device).to(torch.float32), True).detach().cpu().numpy()
    y = np.argmax(y, axis=-1)
    
    return accuracy_score(labels.numpy(), y), f1_score(labels.numpy(), y, average="macro")

def compute_clevr_collapse(true_concepts, predicted_concepts):
    mask_color = true_concepts[:, :, 0].reshape(-1) != -1
    mask_shapes = true_concepts[:, :, 1].reshape(-1) != -1
    mask_materials = true_concepts[:, :, 2].reshape(-1) != -1
    mask_sizes = true_concepts[:, :, 3].reshape(-1) != -1

    filtered_true_colors = true_concepts[:, :, 0].reshape(-1)[mask_color]
    filtered_predicted_colors = predicted_concepts[:, :, 0].reshape(-1)[mask_color]

    filtered_true_shapes = true_concepts[:, :, 1].reshape(-1)[mask_shapes]
    filtered_predicted_shapes = predicted_concepts[:, :, 1].reshape(-1)[mask_shapes]

    filtered_true_materials = true_concepts[:, :, 2].reshape(-1)[mask_materials]
    filtered_predicted_materials = predicted_concepts[:, :, 2].reshape(-1)[mask_materials]

    filtered_true_sizes = true_concepts[:, :, 3].reshape(-1)[mask_sizes]
    filtered_predicted_sizes = predicted_concepts[:, :, 3].reshape(-1)[mask_sizes]

    # Compute collapses
    collapse_color = compute_concept_collapse(
        filtered_true_colors,
        filtered_predicted_colors,
        False,
    )
    collapse_shapes = compute_concept_collapse(
        filtered_true_shapes,
        filtered_predicted_shapes,
        False,
    )
    collapse_materials = compute_concept_collapse(
        filtered_true_materials,
        filtered_predicted_materials,
        False,
    )
    collapse_sizes = compute_concept_collapse(
        filtered_true_sizes,
        filtered_predicted_sizes,
        False,
    )

    return np.mean(
        [collapse_color, collapse_shapes, collapse_materials, collapse_sizes]
    )

def compute_clevr_accuracy(true_concepts, predicted_concepts):
    mask_color = true_concepts[:, :, 0].reshape(-1) != -1
    mask_shapes = true_concepts[:, :, 1].reshape(-1) != -1
    mask_materials = true_concepts[:, :, 2].reshape(-1) != -1
    mask_sizes = true_concepts[:, :, 3].reshape(-1) != -1

    filtered_true_colors = true_concepts[:, :, 0].reshape(-1)[mask_color]
    filtered_predicted_colors = predicted_concepts[:, :, 0].reshape(-1)[mask_color]

    filtered_true_shapes = true_concepts[:, :, 1].reshape(-1)[mask_shapes]
    filtered_predicted_shapes = predicted_concepts[:, :, 1].reshape(-1)[mask_shapes]

    filtered_true_materials = true_concepts[:, :, 2].reshape(-1)[mask_materials]
    filtered_predicted_materials = predicted_concepts[:, :, 2].reshape(-1)[mask_materials]

    filtered_true_sizes = true_concepts[:, :, 3].reshape(-1)[mask_sizes]
    filtered_predicted_sizes = predicted_concepts[:, :, 3].reshape(-1)[mask_sizes]
    
    concept_accuracy_color = accuracy_score(
        filtered_true_colors, filtered_predicted_colors,
    )
    concept_accuracy_shape = accuracy_score(
        filtered_true_shapes, filtered_predicted_shapes,
    )
    concept_accuracy_materials = accuracy_score(
        filtered_true_materials, filtered_predicted_materials,
    )
    concept_accuracy_sizes = accuracy_score(
        filtered_true_sizes, filtered_predicted_sizes,
    )

    concept_f1_macro_color = f1_score(
        filtered_true_colors, filtered_predicted_colors,
        average="macro",
    )
    concept_f1_macro_shape = f1_score(
        filtered_true_shapes, filtered_predicted_shapes,
        average="macro",
    )
    concept_f1_macro_sizes = f1_score(
        filtered_true_sizes, filtered_predicted_sizes,
        average="macro",
    )
    concept_f1_macro_materials = f1_score(
        filtered_true_materials, filtered_predicted_materials,
        average="macro",
    )

    concept_accuracy = np.mean([concept_accuracy_color, concept_accuracy_shape, concept_accuracy_materials, concept_accuracy_sizes])
    concept_f1_macro = np.mean([concept_f1_macro_color, concept_f1_macro_shape, concept_f1_macro_materials, concept_f1_macro_sizes])

    return concept_accuracy, concept_f1_macro