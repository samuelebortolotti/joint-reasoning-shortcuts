# This module contains the computation of the metrics

import torch
import numpy as np
import math
from numpy import ndarray
from typing import Tuple, Dict, List
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.special import softmax


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)

    # pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = pred.eq(target.long())

    # if topk = (1,5), then, k=1 and k=5
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_binary(y_pred, y_true):

    # Split predictions into separate tensors for each label
    y_pred_split = torch.split(y_pred, 2, dim=1)
    y_pred_binary = torch.stack([pred.argmax(dim=1) for pred in y_pred_split], dim=1)

    acc = (
        (y_true.detach().cpu() == y_pred_binary.detach().cpu())
        .all(dim=1)
        .float()
        .mean()
        .item()
    )

    f1 = f1_score(y_true.cpu().numpy(), y_pred_binary.cpu().numpy(), average="micro")

    return acc * 100, f1 * 100


def evaluate_mix(true, pred):
    """Evaluate f1 and accuracy

    Args:
        true: Groundtruth values
        pred: Predicted values

    Returns:
        ac: accuracy
        f1: f1 score
    """
    try:
        ac = accuracy_score(true, pred)
    except ValueError:
        ac = (true == pred).astype(float).mean().item()

    try:
        f1 = f1_score(true, pred, average="weighted")
    except ValueError:
        # Compute true positives, false positives, and false negatives
        tp = np.sum(pred & true, axis=1)
        fp = np.sum(pred & ~true, axis=1)
        fn = np.sum(~pred & true, axis=1)

        # Compute precision, recall, and F1 score for each sample
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        # Take the average F1 score across all samples
        f1 = np.mean(f1)

    return ac, f1


def evaluate_metrics(
    model,
    loader,
    args,
    last=False,
    concatenated_concepts=True,
    apply_softmax=False,
    cf1=False,
):
    """Evaluate metrics of the frequentist model

    Args:
        model (nn.Module): network
        loader: dataloader
        args: command line arguments
        last (bool, default=False): evaluate metrics on the test set
        concatenated_concepts (bool, default=True): return concatenated concepts
        apply_softmax (bool, default=False): whether to apply softmax

    Returns:
        y_true: groundtruth labels, if last is specified
        gs: groundtruth concepts, if last is specified
        ys: label predictions, if last is specified
        cs: concept predictions, if last is specified
        p_cs: concept probability, if last is specified
        p_ys: label probability, if last is specified
        p_cs_all: concept proabability all, if last is specified
        p_ys_all: label probability all, if last is specified
        tloss: test loss, otherwise
        cacc: concept accuracy, otherwise
        yacc: label accuracy, otherwise
        f1sc: f1 on concept, otherwise
    """
    L = len(loader)
    tloss, cacc, yacc = 0, 0, 0
    f1sc, f1 = 0, 0
    fcf1 = 0
    for i, data in enumerate(loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        if args.model in ["mnistdsl"]:
            out_dict = model(images, eval=True)
        else:
            out_dict = model(images)
        out_dict.update({"INPUTS": images, "LABELS": labels, "CONCEPTS": concepts})

        if last and i == 0:
            y_true = labels.detach().cpu().numpy()
            c_true = concepts.detach().cpu().numpy()
            y_pred = out_dict["YS"].detach().cpu().numpy()
            c_pred = out_dict["CS"].detach().cpu().numpy()
            pc_pred = out_dict["pCS"].detach().cpu().numpy()
            if args.model in ["mnistdsl", "clevrdsl"]:
                ys = out_dict["PRED"].detach().cpu().numpy()
            if args.dataset in ["minikandinsky", "clipkandinksy"]:
                pc_pred = pc_pred.reshape(pc_pred.shape[0], pc_pred.shape[1], 18)
        elif last and i > 0:
            y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
            c_true = np.concatenate([c_true, concepts.detach().cpu().numpy()], axis=0)
            y_pred = np.concatenate(
                [y_pred, out_dict["YS"].detach().cpu().numpy()], axis=0
            )
            c_pred = np.concatenate(
                [c_pred, out_dict["CS"].detach().cpu().numpy()], axis=0
            )
            o = out_dict["pCS"].detach().cpu().numpy()
            if args.model in ["mnistdsl"]:
                ys = np.concatenate(
                    [ys, out_dict["PRED"].detach().cpu().numpy()], axis=0
                )
            if args.dataset in ["minikandinsky", "clipkandinsky"]:
                pc_pred = np.concatenate(
                    [pc_pred, o.reshape(o.shape[0], o.shape[1], 18)], axis=0
                )
            else:
                pc_pred = np.concatenate([pc_pred, o], axis=0)

        if args.dataset == "addmnist" and not last:
            loss, ac, acc, f1 = ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts, args)
        elif args.dataset == "boia":
            loss, ac, acc, f1 = SDDOIA_eval_tloss_cacc_acc(out_dict)
        elif args.dataset == "clevr":
            loss, ac, acc, f1 = CLEVR_eval_tloss_cacc_acc(out_dict, concepts)
        else:
            NotImplementedError()

        if not last:
            tloss += loss.item()
            cacc += ac
            yacc += acc
            f1sc += f1

    if apply_softmax:
        y_pred = softmax(y_pred, axis=1)

    if last:

        if args.dataset == "boia":
            y_pred_split = np.split(y_pred, 4, axis=1)
            ys = np.stack([pred.argmax(axis=1) for pred in y_pred_split], axis=1)
            p_ys = np.stack([pred.max(axis=1) for pred in y_pred_split], axis=1)

            gs = c_true
            cs = c_pred
            p_cs = pc_pred
        elif args.model in ["mnistdsl", "mnistdslrec"]:
            gs = np.split(c_true, c_true.shape[1], axis=1)
            cs = np.split(c_pred, c_pred.shape[1], axis=1)
            p_cs = np.split(pc_pred, pc_pred.shape[1], axis=1)
            p_ys = y_pred
        elif args.dataset in ["clevr"]:
            ys = np.argmax(y_pred, axis=1)
            gs = np.reshape(c_true, (c_true.shape[0], 4, -1)) #[c_true[:, 8], c_true[:, 8:11], c_true[:, 11:13], c_true[:, 13:15]]
            cs = c_pred # [c_pred[:, 8], c_pred[:, 8:11], c_pred[:, 11:13], c_pred[:, 13:15]]
            p_cs = pc_pred # [pc_pred[:, 8], pc_pred[:, 8:11], pc_pred[:, 11:13], pc_pred[:, 13:15]]
            p_ys = y_pred
        else:
            ys = np.argmax(y_pred, axis=1)
            gs = np.split(c_true, c_true.shape[1], axis=1)
            cs = np.split(c_pred, c_pred.shape[1], axis=1)
            p_cs = np.split(pc_pred, pc_pred.shape[1], axis=1)
            p_ys = y_pred

        p_cs_all = p_cs
        p_ys_all = y_pred

        assert len(gs) == len(cs), f"gs: {len(gs)}, cs: {len(cs)}"

        if args.dataset not in [
            "boia",
            "clevr"
        ]:
            gs = np.concatenate(gs, axis=0).squeeze(1)
        if args.dataset == "boia":
            cs = (cs >= 0.5).astype(int)
        elif args.dataset != "clevr":
            cs = np.concatenate(cs, axis=0).squeeze(1).argmax(axis=-1)
            p_cs = (
                np.concatenate(p_cs, axis=0).squeeze(1).max(axis=-1)
            )  # should take the maximum as it is the associated probability
        else:
            gs = np.stack([np.argmax(gs[:, :, :8], axis=-1), np.argmax(gs[:, :, 8:11], axis=-1), np.argmax(gs[:, :, 11:13], axis=-1), np.argmax(gs[:, :, 13:15], axis=-1)], axis=-1)
            cs = np.stack([np.argmax(cs[:, :, :8], axis=-1), np.argmax(cs[:, :, 8:11], axis=-1), np.argmax(cs[:, :, 11:13], axis=-1), np.argmax(cs[:, :, 13:15], axis=-1)], axis=-1)

        if args.dataset not in [
            "boia",
            "clevr"
        ]:
            p_cs_all = np.concatenate(p_cs_all, axis=0).squeeze(
                1
            )  # all the items of probabilities are considered
            if args.model not in ["mnistdsl"]:
                p_ys = p_ys.max(axis=1)
        
        assert gs.shape == cs.shape, f"gs: {gs.shape}, cs: {cs.shape}"

        if not concatenated_concepts:
            # by default, consider the concept to return as separated internally
            gs = c_true
            cs = c_pred.argmax(axis=2)

        return y_true, gs, ys, cs, p_cs, p_ys, p_cs_all, p_ys_all
    else:
        if args.dataset in ["boia", "addmnist", "clevr"]:
            if cf1:
                return tloss / L, cacc / L, yacc / L, f1sc / L, fcf1 / L
            else:
                return tloss / L, cacc / L, yacc / L, f1sc / L
        else:
            return tloss / L, cacc / L, yacc / L, 0


def ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts, args):
    """ADDMMNIST evaluation

    Args:
        out_dict (Dict[str]): dictionary of outputs
        concepts: concepts

    Returns:
        loss: loss
        cacc: concept accuracy
        acc: label accuracy
        zero: 0
    """
    reprs = out_dict["CS"]
    L = len(reprs)

    objs = torch.split(
        reprs,
        1,
        dim=1,
    )
    g_objs = torch.split(concepts, 1, dim=1)

    if args.dataset != "clipshortmnist":

        assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

        loss, cacc = 0, 0
        for j in range(len(objs)):
            # enconding + ground truth
            obj_enc = objs[j].squeeze(dim=1)
            g_obj = g_objs[j].to(torch.long).view(-1)

            # evaluate loss
            loss += torch.nn.CrossEntropyLoss()(obj_enc, g_obj)

            # concept accuracy of object
            c_pred = torch.argmax(obj_enc, dim=1)

            assert (
                c_pred.size() == g_obj.size()
            ), f"size c_pred: {c_pred.size()}, size g_objs: {g_obj.size()}"

            correct = (c_pred == g_obj).sum().item()
            cacc += correct / len(objs[j])
    else:
        cacc = 0.0
        loss = torch.tensor([0.0])

    y = out_dict["YS"]
    y_true = out_dict["LABELS"]

    if args.model in ["mnistdsl", "mnistdslrec"]:
        y_pred = out_dict["PRED"].squeeze()
    else:
        y_pred = torch.argmax(y, dim=-1)
    assert (
        y_pred.size() == y_true.size()
    ), f"size y_pred: {y_pred.size()}, size y_true: {y_true.size()}"

    acc = (y_pred.detach().cpu() == y_true.detach().cpu()).sum().item() / len(y_true)

    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")

    return loss / len(objs), cacc / len(objs) * 100, acc * 100, f1 * 100


def SDDOIA_eval_tloss_cacc_acc(out_dict):
    """SDDOIA evaluation

    Args:
        out_dict (Dict[str]): dictionary of outputs

    Returns:
        loss: loss
        cacc: concept accuracy
        acc: label accuracy
        f1: f1 accuracy
    """
    cs = out_dict["CS"]
    c_pred = (cs >= 0.5).to(torch.long)
    c_true = out_dict["CONCEPTS"].to(torch.long)

    y_pred = out_dict["YS"]
    y_true = out_dict["LABELS"]

    # case for fully neural approach
    if torch.all(cs < 0).item():
        loss = torch.tensor(0.0, device=cs.device)
    else:
        loss = F.binary_cross_entropy(cs, c_true.float())

    cacc = (c_true == c_pred).all(dim=1).float().mean().item()

    # Split predictions into separate tensors for each label
    y_pred_split = torch.split(y_pred, 2, dim=1)
    y_pred_binary = torch.stack([pred.argmax(dim=1) for pred in y_pred_split], dim=1)

    acc = accuracy_score(y_true.cpu().numpy(), y_pred_binary.cpu().numpy())
    f1 = f1_score(y_true.cpu().numpy(), y_pred_binary.cpu().numpy(), average="micro")

    return loss, cacc * 100, acc * 100, f1 * 100


def CLEVR_eval_tloss_cacc_acc(out_dict, concepts):
    """CLEVR evaluation

    Args:
        out_dict (Dict[str]): dictionary of outputs

    Returns:
        loss: loss
        cacc: concept accuracy
        acc: label accuracy
        f1: f1 accuracy
    """
    reprs = out_dict["CS"]

    concepts_flat = concepts

    L = len(reprs)

    objs = torch.split(
        reprs,
        1,
        dim=1,
    )
    g_objs = torch.split(concepts_flat, 15, dim=1)

    assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

    loss, cacc = 0, 0

    for j in range(len(objs)):
        # enconding + ground truth
        obj_enc = objs[j].squeeze()
        g_obj = g_objs[j].to(torch.long)

        # evaluate loss

        colors, shapes, materials, sizes = g_obj[:, :8], g_obj[:, 8:11], g_obj[:, 11:13], g_obj[:, 13:15]
        pcolors, pshapes, pmaterials, psizes = obj_enc[:, :8], obj_enc[:, 8:11], obj_enc[:, 11:13], obj_enc[:, 13:15]

        loss += torch.nn.CrossEntropyLoss()(pcolors, torch.argmax(colors, dim=-1))
        loss += torch.nn.CrossEntropyLoss()(pshapes, torch.argmax(shapes, dim=-1))
        loss += torch.nn.CrossEntropyLoss()(pmaterials, torch.argmax(materials, dim=-1))
        loss += torch.nn.CrossEntropyLoss()(psizes, torch.argmax(sizes, dim=-1))

        correct = 0
        correct += (torch.argmax(colors, dim=-1) == torch.argmax(pcolors, dim=-1)).sum().item()
        correct += (torch.argmax(shapes, dim=-1) == torch.argmax(pshapes, dim=-1)).sum().item()
        correct += (torch.argmax(materials, dim=-1) == torch.argmax(pmaterials, dim=-1)).sum().item()
        correct += (torch.argmax(sizes, dim=-1) == torch.argmax(psizes, dim=-1)).sum().item()

        cacc += correct / (len(objs[j]) * 4)

    y = out_dict["YS"]
    y_true = out_dict["LABELS"].to(torch.long)

    if "PRED" in out_dict:
        y_pred = out_dict["PRED"]
    else:
        y_pred = torch.argmax(y, dim=-1)

    acc = (y_pred.flatten().detach().cpu() == y_true.flatten().detach().cpu()).sum().item() / len(y_true.flatten())

    f1 = f1_score(y_true.flatten().cpu().numpy(), y_pred.flatten().cpu().numpy(), average="macro")

    return loss / len(objs), cacc / len(objs) * 100, acc * 100, f1 * 100


def entropy(probabilities: ndarray, n_values: int):
    """Entropy

    Args:
        probabilities (ndarray): probability vector
        n_values (int): n values

    Returns:
        entropy_values: entropy vector
    """
    # Compute entropy along the columns
    probabilities += 1e-5
    probabilities /= 1 + (n_values * 1e-5)

    entropy_values = -np.sum(probabilities * np.log(probabilities), axis=1) / np.log(
        n_values
    )
    return entropy_values


def mean_entropy(probabilities: ndarray, n_values: int) -> float:
    """Mean Entropy

    Args:
        probabilities (ndarray): probability vector
        n_values (int): n values

    Returns:
        entropy_values: mean entropy
    """
    # Accepts a ndarray of dim n_examples * n_classes (or equivalently n_concepts)
    entropy_values = entropy(probabilities, n_values).mean()
    return entropy_values.item()


