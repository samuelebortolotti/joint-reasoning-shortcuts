# Losses module
import torch
import numpy as np
import torch.nn.functional as F
from info_nce import InfoNCE, info_nce
from utils.contrastive import apply_augmentation
import os
from PIL import Image
from utils.boia_knowledge import get_boia_actions_from_concepts


K_MNIST = [(5, 5, 10), (8, 7, 15), (2, 8, 10), (7, 4, 11), (6, 2, 8), (7, 5, 12), (5, 3, 8), (6, 7, 13), (9, 6, 15), (8, 6, 14), (1, 8, 9), (0, 9, 9), (1, 1, 2), (4, 9, 13), (7, 2, 9), (0, 7, 7), (2, 0, 2), (0, 2, 2), (1, 7, 8), (6, 4, 10), (8, 8, 16), (9, 5, 14), (5, 4, 9), (6, 1, 7), (3, 9, 12), (3, 2, 5), (6, 6, 12), (0, 1, 1), (2, 7, 9), (4, 4, 8), (9, 8, 17), (0, 8, 8), (1, 2, 3), (9, 4, 13), (2, 5, 7), (8, 0, 8), (2, 6, 8), (0, 3, 3), (8, 1, 9), (3, 5, 8), (3, 1, 4), (3, 8, 11), (8, 4, 12), (1, 3, 4), (4, 5, 9), (8, 5, 13), (2, 4, 6), (2, 9, 11), (3, 4, 7), (7, 3, 10), (6, 3, 9), (7, 0, 7), (2, 1, 3), (7, 6, 13), (7, 7, 14), (0, 4, 4), (1, 5, 6), (8, 2, 10), (8, 9, 17), (3, 7, 10), (6, 0, 6), (1, 6, 7), (9, 7, 16), (9, 3, 12), (4, 2, 6), (7, 1, 8), (3, 3, 6), (5, 2, 7), (5, 0, 5), (1, 0, 1), (0, 6, 6), (4, 8, 12), (1, 9, 10), (6, 9, 15), (9, 2, 11), (7, 8, 15), (7, 9, 16), (0, 0, 0), (5, 8, 13), (4, 1, 5), (9, 9, 18), (9, 1, 10), (5, 1, 6), (2, 2, 4), (2, 3, 5), (4, 3, 7), (5, 9, 14), (3, 6, 9), (8, 3, 11), (6, 8, 14), (4, 7, 11), (0, 5, 5), (5, 7, 12), (9, 0, 9), (4, 6, 10), (6, 5, 11), (5, 6, 11), (4, 0, 4), (1, 4, 5), (3, 0, 3)]
K_MNIST_SP = [(5, 5, 0), (8, 7, 1), (2, 8, 0), (7, 4, 1), (6, 2, 0), (7, 5, 0), (5, 3, 0), (6, 7, 1), (9, 6, 1), (8, 6, 0), (1, 8, 1), (0, 9, 1), (1, 1, 0), (4, 9, 1), (7, 2, 1), (0, 7, 
1), (2, 0, 0), (0, 2, 0), (1, 7, 0), (6, 4, 0), (8, 8, 0), (9, 5, 0), (5, 4, 1), (6, 1, 1), (3, 9, 0), (3, 2, 1), (6, 6, 0), (0, 1, 1), (2, 7, 1), (4, 4, 0), (9, 8, 1), (0, 
8, 0), (1, 2, 1), (9, 4, 1), (2, 5, 1), (8, 0, 0), (2, 6, 0), (0, 3, 1), (8, 1, 1), (3, 5, 0), (3, 1, 0), (3, 8, 1), (8, 4, 0), (1, 3, 0), (4, 5, 1), (8, 5, 1), (2, 4, 0), (
2, 9, 1), (3, 4, 1), (7, 3, 0), (6, 3, 1), (7, 0, 1), (2, 1, 1), (7, 6, 1), (7, 7, 0), (0, 4, 0), (1, 5, 0), (8, 2, 0), (8, 9, 1), (3, 7, 0), (6, 0, 0), (1, 6, 1), (9, 7, 0)
, (9, 3, 0), (4, 2, 0), (7, 1, 0), (3, 3, 0), (5, 2, 1), (5, 0, 1), (1, 0, 1), (0, 6, 0), (4, 8, 0), (1, 9, 0), (6, 9, 1), (9, 2, 1), (7, 8, 1), (7, 9, 0), (0, 0, 0), (5, 8,
 1), (4, 1, 1), (9, 9, 0), (9, 1, 0), (5, 1, 0), (2, 2, 0), (2, 3, 1), (4, 3, 1), (5, 9, 0), (3, 6, 1), (8, 3, 1), (6, 8, 0), (4, 7, 1), (0, 5, 1), (5, 7, 0), (9, 0, 1), (4,
 6, 0), (6, 5, 1), (5, 6, 1), (4, 0, 0), (1, 4, 1), (3, 0, 1)]

def ADDMNIST_Classification(out_dict: dict, args):
    """Addmnist classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)

    if args.model in [
        "mnistdpl",
        "mnistdplrec",
        "mnistpcbmdpl",
        "mnistclip",
        "mnistnn",
        "mnistcbm",
        "mnistcbmrec",
        "mnistdsldpl",
        "mnistdsldplrec",
        "mnistsenn"
    ]:
        loss = F.nll_loss(out.log(), labels, reduction="mean")
    elif args.model in [
        "mnistsl",
        "mnistslrec",
        "mnistpcbmsl",
    ]:
        loss = F.cross_entropy(out, labels, reduction="mean")
    elif args.model in [
        "mnistdsl",
        "mnistdslrec"
    ]:
        # compute loss
        pred = out_dict["PRED"].squeeze()
        model_labels = torch.where(torch.eq(labels, pred), 1.0, 0.0).view(-1)
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(torch.logit(out, 0.0001), model_labels)
    else:
        print("MALe", args.model)
        print()
        quit()
        loss = torch.tensor(1e-5)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses

def CUB_Classification(out_dict: dict, args):
    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)

    if args.model in [
        "cubcbm"
    ]:
        loss = F.nll_loss(out.log(), labels, reduction="mean")
    else:
        print("MALe", args.model, "cubcbm")
        print()
        quit()
        loss = torch.tensor(1e-5)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def CUB_Concept_Match(out_dict: dict, args):
    reprs = out_dict["CS"].squeeze()
    concepts = out_dict["CONCEPTS"].to(torch.float32)

    mask = concepts != -1
    loss = torch.tensor(0.0)

    if mask.any():
        loss = torch.nn.functional.cross_entropy(reprs[mask], concepts[mask])

    loss_value = loss
    if isinstance(loss, torch.Tensor):
        loss_value = loss.item()

    return loss, {"c-loss": loss_value}

def ADDMNIST_Concept_Match(out_dict: dict, args):
    """Addmnist concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["CS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    objs = reprs.unbind(dim=1)
    g_objs = concepts.unbind(dim=1)

    assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

    loss = sum(
        torch.nn.functional.cross_entropy(objs[j][mask].squeeze(1), g_objs[j][mask].view(-1))
        for j, mask in enumerate(g_objs[j] != -1 for j in range(len(g_objs)))
        if mask.any()
    )

    loss /= len(objs)

    loss_value = loss
    if isinstance(loss, torch.Tensor):
        loss_value = loss.item()

    return loss, {"c-loss": loss_value}

def debug_save_images(tensor, titles, args, output_dir="./output"):
    if args.model == "mnistsenn":
        output_dir = "./output_senn"
    elif args.w_rec == 0.1:
        output_dir = "./output_current"
    os.makedirs(output_dir, exist_ok=True)

    left_image, right_image = tensor[:, :, :, :28], tensor[:, :, :, 28:]

    images = torch.cat([left_image, right_image], dim=0)
    for i in range(images.shape[0]):
        img = images[i].squeeze().detach().cpu().numpy() * 255.0
        img = Image.fromarray(img).convert("L")
        filename = os.path.join(output_dir, f"{titles[i]}.png")
        img.save(filename)


def CUB_Knowledge_Match(out_dict: dict, args):
    return NotImplementedError


def ADDMNIST_Knowledge_Match(out_dict: dict, args):
    """Addmnist knowledge match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """

    dataset = K_MNIST if args.task == "addition" else K_MNIST_SP
    num_to_select = int(len(dataset) * args.perc_k)
    selected_combinations = dataset[:num_to_select]

    if not selected_combinations:
        return torch.tensor(0, device=args.device), {"knowledge-loss": 0}

    c1, c2, labels = map(torch.tensor, zip(*selected_combinations))
    c1, c2, labels = c1.to(args.device), c2.to(args.device), labels.to(args.device)

    if "senn" in args.model:
        c1, c2 = F.one_hot(c1, n).float(), F.one_hot(c2, n).float()
        phi = torch.cat([c1, c2], dim=-1).unsqueeze(-1)

        wx = out_dict["WX"]
        dot = torch.sum(wx * phi, dim=1)
        y = F.softmax(dot, dim=-1)
    else:
        knowledge = out_dict["KNOWLEDGE"]
        if "cbm" in args.model:
            n = 10 if args.dataset in {"addmnist", "shortmnist"} else 5
            c1, c2 = F.one_hot(c1, n).float(), F.one_hot(c2, n).float()
            y = knowledge((c1.unsqueeze(2) * c2.unsqueeze(1)).view(c1.shape[0], -1))
        else:
            y = knowledge[c1, c2]

    loss = F.nll_loss(torch.log(y), labels)
    return loss, {"knowledge-loss": loss.item()}


def CUB_Contrastive(out_dict: dict, model):
    criterion = InfoNCE()

    logits = out_dict["CS"]
    inputs = out_dict["INPUTS"]
    model = out_dict['MODEL']

    model.eval()
    new_inputs = apply_augmentation(inputs, "cub")
    positive = model(new_inputs)['CS']
    loss = criterion(logits, positive)
    model.train()

    return loss, {"contrastive-loss": loss.item()}


def ADDMNIST_Contrastive(out_dict: dict, model):
    """Addmnist contrastive loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    criterion = InfoNCE()

    logits = out_dict["CS"]
    logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
    inputs = out_dict["INPUTS"]
    model = out_dict['MODEL']

    model.eval()

    new_inputs = apply_augmentation(inputs)
    positive = model(new_inputs)['CS']
    positive = positive.view(positive.shape[0] * positive.shape[1], positive.shape[2])
    loss = criterion(logits, positive)
    model.train()

    return loss, {"contrastive-loss": loss.item()}


def CLEVR_Contrastive(out_dict: dict, model):
    """Addmnist contrastive loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    criterion = InfoNCE()
    logits = out_dict["CS"]
    inputs = out_dict["INPUTS"]
    model = out_dict['MODEL']

    model.eval()
    new_inputs = apply_augmentation(inputs, "clevr")
    model.train()
    positive = model(new_inputs)['CS']

    mask = (inputs == -1).all(-1).all(-1).all(-1)
    valid_indices = ~mask 

    logits = logits[valid_indices]
    positive = positive[valid_indices]

    loss = criterion(logits, positive)
    losses = {"contrastive-loss": loss.item()}

    return loss, losses



def ADDMNIST_Concept_CLIP(out_dict: dict, args):
    """Addmnist concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"]
    objs = torch.split(reprs, 1, dim=1)
    g_objs = torch.split(concepts, 1, dim=1)
    loss = torch.tensor(0.0, device=reprs.device)

    assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

    for j in range(len(objs)):

        input_prob = objs[j].squeeze(1)
        targt_prob = g_objs[j].squeeze(1)

        loss += F.kl_div(input_prob.log(), targt_prob)
    losses = {"clip-loss": loss.item()}

    return loss / len(objs), losses


def ADDMNIST_REC_Match(out_dict: dict, args):
    """Addmnist concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    recs, inputs, mus, logvars = (
        out_dict["RECS"], out_dict["INPUTS"], out_dict["MUS"], out_dict["LOGVARS"]
    )

    assert inputs.shape == recs.shape, f"{inputs.shape}-{recs.shape}"

    L = recs.shape[0]
    recon = F.binary_cross_entropy(recs.view(L, -1), inputs.view(L, -1))
    kld = (0.5 * (mus.pow(2) + logvars.exp() - logvars - 1).sum(1).mean()).abs()

    losses = {"recon-loss": recon.item(), "kld": kld.item()}

    if out_dict["EPOCH"] > 20:
        concepts = out_dict["CONCEPTS"][:10, :2].flatten().tolist()
        conc_list = [str(el) for el in concepts]
        debug_save_images(recs[:10], conc_list, args)

    return recon + args.beta * kld, losses


def CUB_REC_Match(out_dict: dict, args):
    recs, inputs, mus, logvars = (
        out_dict["RECS"], out_dict["INPUTS"], out_dict["MUS"], out_dict["LOGVARS"]
    )

    assert inputs.shape == recs.shape, f"{inputs.shape}-{recs.shape}"

    L = recs.shape[0]
    recon = F.binary_cross_entropy(recs.view(L, -1), inputs.view(L, -1))
    kld = (0.5 * (mus.pow(2) + logvars.exp() - logvars - 1).sum(1).mean()).abs()

    losses = {"recon-loss": recon.item(), "kld": kld.item()}

    if out_dict["EPOCH"] > 20:
        concepts = out_dict["CONCEPTS"][:10, :2].flatten().tolist()
        conc_list = [str(el) for el in concepts]
        debug_save_images(recs[:10], conc_list)

    return recon + args.beta * kld, losses


def ADDMNIST_Entropy(out_dict, args):
    """Addmnist entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    l = pCs.size(-1)
    p_mean = torch.mean(pCs, dim=0).view(-1, l) + 1e-5
    p_mean /= p_mean.sum(dim=1, keepdim=True)
    loss = -(p_mean * p_mean.log()).sum(dim=1).mean() / np.log(pCs.shape[-1])

    final_loss = 1 - loss.item()

    assert final_loss > -1e-5, f"Final loss is out of range: {final_loss}"

    return final_loss, {"H-loss": final_loss}


def CUB_Entropy(out_dict, args):
    """Addmnist entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    l = pCs.size(-1)
    p_mean = torch.mean(pCs, dim=0).view(-1, l) + 1e-5
    p_mean /= p_mean.sum(dim=1, keepdim=True)
    loss = -(p_mean * p_mean.log()).sum(dim=1).mean() / np.log(pCs.shape[-1])

    final_loss = 1 - loss.item()

    assert final_loss > -1e-5, f"Final loss is out of range: {final_loss}"

    return final_loss, {"H-loss": final_loss}


def senn_robustness_loss(x, model):
    x.requires_grad_(True)

    out_dict = model(x)
    aggregates = out_dict["YS"].squeeze(-1)  # (batch_size, num_classes)
    relevances = out_dict["WX"]  # (batch_size, num_concepts, num_classes)
    concepts = out_dict["PHIX"].squeeze(-1)  # (batch_size, num_concepts)

    batch_size, channels, height, width = x.shape
    num_features = channels * height * width
    num_classes = aggregates.shape[1]
    num_concepts = concepts.shape[1]

    def compute_jacobian(output, x, output_dim):
        jacobian = []
        for i in range(output_dim):
            grad_outputs = torch.zeros_like(output).to(x.device)
            grad_outputs[:, i] = 1.0 
            grad = torch.autograd.grad(outputs=output, inputs=x, grad_outputs=grad_outputs, 
                                       retain_graph=True, create_graph=True, only_inputs=True)[0]
            jacobian.append(grad.view(batch_size, num_features, 1))
        return torch.cat(jacobian, dim=2)

    J_yx = compute_jacobian(aggregates, x, num_classes)  # (batch_size, num_features, num_classes)
    J_hx = compute_jacobian(concepts, x, num_concepts)  # (batch_size, num_features, num_concepts)

    robustness_loss = J_yx - torch.bmm(J_hx, relevances)  # (batch_size, num_features, num_classes)
    return robustness_loss.norm(p='fro')


def ADDMNIST_SENN(out_dict, args):
    """Addmnist senn loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """

    x = out_dict["INPUTS"]
    model = out_dict["MODEL"]

    robustness_loss = senn_robustness_loss(x, model)

    return robustness_loss, {"loss-jacobian": robustness_loss.item()}


def ADDMNIST_rec_class(out_dict: dict, args):
    """Addmnist rec class

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss1, losses1 = ADDMNIST_Classification(out_dict, args)
    loss2, losses2 = ADDMNIST_REC_Match(out_dict, args)

    losses1.update(losses2)

    return loss1 + args.gamma * loss2, losses1


def ADDMNIST_Cumulative(out_dict: dict, args):
    """Addmnist cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = ADDMNIST_Classification(out_dict, args)
    mitigation = 0
    if args.model in ["mnistdplrec", "mnistcbmrec", "mnistslrec", "mnistltnrec", "mnistdslrec", "mnistdsldplrec", "mnistsenn"]:
        loss1, losses1 = ADDMNIST_REC_Match(out_dict, args)
        mitigation += args.w_rec * loss1
        losses.update(losses1)
    if args.entropy:
        loss2, losses2 = ADDMNIST_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = ADDMNIST_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)
    if args.k_sup > 0 and ("dsl" in args.model or "cbm" in args.model):
        loss4, losses4 = ADDMNIST_Knowledge_Match(out_dict, args)
        mitigation += args.k_sup * loss4
        losses.update(losses4)
    if args.contrastive:
        loss5, losses5 = ADDMNIST_Contrastive(out_dict, args)
        mitigation += args.w_con * loss5
        losses.update(losses5)
    if args.model in ["mnistsenn"] and False:
        loss6, losses6 = ADDMNIST_SENN(out_dict, args)
        loss += loss6 * args.w_senn
        losses.update(losses6)

    return loss + args.gamma * mitigation, losses


def CUB_Cumulative(out_dict: dict, args):

    loss, losses = CUB_Classification(out_dict, args)

    mitigation = 0
    if args.model in ["cubcbmrec"]:
        loss1, losses1 = CUB_REC_Match(out_dict, args)
        mitigation += args.w_rec * loss1
        losses.update(losses1)
    if args.entropy:
        loss2, losses2 = CUB_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = CUB_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)
    if args.k_sup > 0 and ("dsl" in args.model or "cbm" in args.model):
        loss4, losses4 = CUB_Knowledge_Match(out_dict, args)
        mitigation += args.k_sup * loss4
        losses.update(losses4)
    if args.contrastive:
        loss5, losses5 = CUB_Contrastive(out_dict, args)
        mitigation += args.w_con * loss5
        losses.update(losses5)

    return loss + args.gamma * mitigation, losses

def ADDMNIST_DSL(out_dict: dict, args):
    """Addmnist DSL loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = ADDMNIST_DSL(out_dict, args)
    mitigation = 0
    if args.model in ["mnistdplrec", "mnistslrec", "mnistltnrec", "mnistdslrec", "mnistdsldplrec"]:
        loss1, losses1 = ADDMNIST_REC_Match(out_dict, args)
        mitigation += args.w_rec * loss1
        losses.update(losses1)
    if args.entropy:
        loss2, losses2 = ADDMNIST_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = ADDMNIST_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    return loss + args.gamma * mitigation, losses


def KAND_Classification(out_dict: dict, args):
    """Kandinsky classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    out = out_dict["YS"]
    # preds = out_dict["PREDS"]
    final_labels = out_dict["LABELS"][:, -1].to(torch.long)
    # inter_labels = out_dict["LABELS"][:, :-1].to(torch.long)

    if args.task in ["patterns"]:
        weight = torch.tensor(
            [
                1 / 0.04938272,
                1 / 0.14814815,
                1 / 0.02469136,
                1 / 0.14814815,
                1 / 0.44444444,
                1 / 0.07407407,
                1 / 0.02469136,
                1 / 0.07407407,
                1 / 0.01234568,
            ],
            device=out.device,
        )
        final_weight = torch.tensor([0.5, 0.5], device=out.device)
    elif args.task == "red_triangle":
        weight = torch.tensor([0.35538, 1 - 0.35538], device=out.device)
        final_weight = torch.tensor([0.04685, 1 - 0.04685], device=out.device)
    else:
        weight = torch.tensor([0.5, 0.5], device=out.device)
        final_weight = torch.tensor([0.5, 0.5], device=out.device, dtype=torch.float64)

    if args.model in [
        "kanddpl",
        "kandcbm",
        "kandnn",
        "kandclip",
        "minikanddpl",
        "kandcbm",
    ]:
        ## ADD SMALL OFFSET
        # out += 1e-5
        # with torch.no_grad():
        #     Z = torch.sum(out, dim=1, keepdim=True)
        # out /= Z

        if args.model in ["kandcbm"]:
            criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=final_weight.float()
            )
            loss = criterion(out, final_labels)
        else:
            loss = F.nll_loss(
                out.log(), final_labels, reduction="mean"  # , weight=final_weight
            )
    else:
        loss = torch.tensor(1e-5)

    assert loss > 0, f"{loss}, {out}, {final_labels}"

    losses = {"y-loss": loss.item()}

    return loss, losses


def KAND_Concept_Match(out_dict: dict):
    """Kandinsky concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["CS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    objs = torch.split(reprs, 1, dim=1)
    g_objs = torch.split(concepts, 1, dim=1)

    loss = torch.tensor(0.0, device=reprs.device)

    assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

    for j in range(len(g_objs)):

        # Loop though the figures

        cs = torch.split(objs[j], 3, dim=-1)
        gs = torch.split(g_objs[j], 1, dim=-1)

        assert len(cs) == len(gs), f"{len(cs)}-{len(gs)}"

        for k in range(len(gs)):
            target = gs[k].view(-1)
            mask = target != -1
            if mask.sum() > 0:
                loss += torch.nn.CrossEntropyLoss()(
                    cs[k][mask].squeeze(1), target[mask].view(-1)
                )

    loss /= len(g_objs) * len(gs)

    losses = {"c-loss": loss.item()}

    return loss, losses


def KAND_Entropy(out_dict, args):
    """Kandinsky entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]

    pc_i = torch.cat(torch.split(pCs, 3, dim=-1), dim=1)

    p_mean = torch.mean(pc_i, dim=0)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=1, keepdim=True)
    p_mean /= Z

    loss = 0
    for i in range(p_mean.size(0)):
        loss -= torch.sum(p_mean[i] * p_mean[i].log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > 0, loss

    return 1 - loss, losses


def KAND_Cumulative(out_dict: dict, args):
    """Kandinsky cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = KAND_Classification(out_dict, args)

    mitigation = 0
    if args.model in ["kandplrec", "kandslrec", "kandltnrec"]:
        return NotImplementedError("not available")
    if args.entropy:
        loss2, losses2 = KAND_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = KAND_Concept_Match(out_dict)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    # return mitigation, losses
    return loss + args.gamma * mitigation, losses


def SDDOIA_Classification(out_dict: dict, args):
    """SDDOIA classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = 0, {}
    mitigation = 0

    if args.boia_model in ["ce"]:
        loss, losses1 = SDDOIA_CE(out_dict, args)
        losses.update(losses1)
    elif args.boia_model in ["bce"]:
        loss, losses1 = SDDOIA_BCE(out_dict, args)
        losses.update(losses1)
    else:
        raise NotImplementedError("Not implemented loss")

    if args.k_sup > 0 and ("cbm" in args.model or "dsl" in args.model):
        loss2, losses2 = BOIA_Knowledge_Match(out_dict, args)
        mitigation += args.k_sup * loss2
        losses.update(losses2)

    return loss + args.gamma * mitigation, losses


def SDDOIA_Entropy(out_dict, args):
    """SDDOIA entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    p_mean = torch.mean(pCs, dim=0)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=0, keepdim=True)
    p_mean /= Z

    loss = -torch.sum(p_mean * p_mean.log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > -0.00001, loss

    return 1 - loss, losses


def SDDOIA_Concept_Match(out_dict: dict, args):
    """SDDOIA concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    loss = torch.tensor(0.0, device=reprs.device)

    probs_list = torch.split(reprs, 2, dim=1)

    for i, rep in enumerate(probs_list):
        # Create a mask to filter out concepts with -1
        mask = concepts[:, i] != -1
        if mask.sum() > 0:  # Proceed only if there are valid entries
            # Apply the mask
            filtered_rep = rep[mask]
            filtered_concepts = concepts[mask, i]
            loss += torch.nn.NLLLoss()(filtered_rep.log(), filtered_concepts)

    print("Concept supervision loss", loss.item())

    losses = {"c-loss": loss.item()}

    return loss, losses


def SDDOIA_Cumulative(out_dict: dict, args):
    """SDDOIA cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = SDDOIA_Classification(out_dict, args)

    mitigation = 0
    if args.entropy:
        loss2, losses2 = SDDOIA_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = SDDOIA_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    return loss + args.gamma * mitigation, losses


def SDDOIA_BCE(out_dict: dict, args):
    """SDDOIA bce

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """

    def BCE_forloop(tar, pred):
        loss = F.binary_cross_entropy(tar[0, :4], pred[0, :4])

        for i in range(1, len(tar)):
            loss = loss + F.binary_cross_entropy(tar[i, :4], pred[i, :4])
        return loss

    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)

    loss = BCE_forloop(out, labels)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def SDDOIA_CE(out_dict: dict, args):
    """SDDOIA bce

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """

    def CE_forloop(y_pred, y_true):

        y_trues = torch.split(y_true, 1, dim=-1)
        y_preds = torch.split(y_pred, 2, dim=-1)

        loss = 0
        for i in range(4):

            true = y_trues[i].view(-1)
            pred = y_preds[i]

            ## add small offset to avoid NaNs
            pred = pred + 1e-5
            with torch.no_grad():
                Z = torch.sum(pred, dim=0, keepdim=True)
            pred = pred / Z

            assert torch.max(pred) < 1, pred
            assert torch.min(pred) > 0, pred

            loss_i = F.nll_loss(pred.log(), true.to(torch.long))
            loss += loss_i / 4

            assert loss_i > 0, pred.log()

        return loss

    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)

    loss = CE_forloop(out, labels)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses

def XOR_Classification(out_dict: dict, args):
    """XOR classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = 0, {}

    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)
    
    if args.model in [
        "xorsl",
        "xorcbm",
        "xornn",
        "xordpl",
    ]:
        loss = F.cross_entropy(out, labels, reduction="mean")
    else:
        loss = torch.tensor(1e-5)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def XOR_Entropy(out_dict, args):
    """XOR entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    p_mean = torch.mean(pCs, dim=0)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=0, keepdim=True)
    p_mean /= Z

    loss = -torch.sum(p_mean * p_mean.log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > -0.00001, loss

    return 1 - loss, losses


def XOR_Concept_Match(out_dict: dict, args):
    """XOR concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    loss = torch.tensor(0.0, device=reprs.device)

    for i, rep in enumerate(range(reprs.size(1))):
        # Create a mask to filter out concepts with -1
        filtered_rep = reprs[:, i]
        filtered_concepts = concepts[:, i]
        loss += torch.nn.NLLLoss()(filtered_rep.log(), filtered_concepts)

    print("Concept supervision loss", loss.item())

    losses = {"c-loss": loss.item()}

    return loss, losses

def XOR_Cumulative(out_dict: dict, args):
    """Xor cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = XOR_Classification(out_dict, args)

    mitigation = 0
    if args.entropy:
        loss2, losses2 = XOR_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = XOR_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    return loss + args.gamma * mitigation, losses

def MNMATH_Classification(out_dict: dict, args):
    """XOR classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = 0, {}

    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.float)

    if args.model in [
        "mnmathnn",
        "mnmathcbm",
        "mnmathsl",
        "mnmathdpl",
    ]:
        loss += torch.nn.BCELoss()(out, labels)
    else:
        loss = torch.tensor(1e-5)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def MNMATH_Entropy(out_dict, args):
    """XOR entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    p_mean = torch.mean(pCs, dim=0)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=0, keepdim=True)
    p_mean /= Z

    loss = -torch.sum(p_mean * p_mean.log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > -0.00001, loss

    return 1 - loss, losses


def MNMATH_Concept_Match(out_dict: dict, args):
    """XOR concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)
    concepts = concepts.view(concepts.size(0), concepts.size(1) * concepts.size(2), 1)

    loss = torch.tensor(0.0, device=reprs.device)

    for i, rep in enumerate(range(reprs.size(1))):
        # Create a mask to filter out concepts with -1
        filtered_rep = reprs[:, i]
        filtered_concepts = concepts[:, i].squeeze(1)

        specific_concepts = [0, 5, 9]
        mask = torch.isin(filtered_concepts, torch.tensor(specific_concepts).to(filtered_concepts.device)).to(filtered_concepts.device)

        filtered_concepts = filtered_concepts[mask]
        filtered_predictions = filtered_rep[mask]

        if filtered_concepts.size(0) > 0:
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(filtered_predictions, filtered_concepts)

    print("Concept supervision loss", loss.item())

    losses = {"c-loss": loss.item()}

    return loss, losses

def MNMATH_Cumulative(out_dict: dict, args):
    """Xor cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = MNMATH_Classification(out_dict, args)

    mitigation = 0
    if args.entropy:
        loss2, losses2 = MNMATH_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = MNMATH_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    return loss + args.gamma * mitigation, losses

def CLEVR_Cumulative(out_dict: dict, args):
    """CLEVR cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = CLEVR_Classification(out_dict, args)
    mitigation = 0
    if args.model in ["clevrcbmrec", "clevrdslrec", "clevrdsldplrec"]:
        loss1, losses1 = CLEVR_REC_Match(out_dict, args)
        mitigation += args.w_rec * loss1
        losses.update(losses1)
    if args.entropy:
        loss2, losses2 = CLEVR_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = CLEVR_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)
    if args.k_sup > 0 and ("cbm" in args.model or "dsl" in args.model):
        loss4, losses4 = CLEVR_Knowledge_Match(out_dict, args)
        mitigation += args.k_sup * loss4
        losses.update(losses4)
    if args.contrastive:
        loss5, losses5 = CLEVR_Contrastive(out_dict, args)
        mitigation += args.w_con * loss5
        losses.update(losses5)

    return loss + args.gamma * mitigation, losses


def CLEVR_Classification(out_dict: dict, args):
    """CLEVR classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)

    if args.model in [
        "clevrcbm",
        "clevrcbmrec",
        "clevrdsldpl",
        "clevrdsldplrec",
        "clevrdpl"
    ]:
        loss = F.nll_loss(out.log(), labels, reduction="mean")
    elif args.model in [
        "clevrdsl",
        "clevrdslrec"
    ]:
        pred = out_dict["PRED"].squeeze()
        model_labels = torch.where(torch.eq(labels, pred), 1.0, 0.0).view(-1)
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(torch.logit(out, 0.0001), model_labels)
    else:
        loss = torch.tensor(1e-5)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def CLEVR_Entropy(out_dict, args):
    """CLEVR entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]

    loss = torch.tensor(0.0).to(args.device)

    pCs = pCs.chunk(4, dim=1)
    for pimg in pCs:
        for p, dim in zip(torch.split(pimg, [8, 3, 2, 2], dim=-1), [8, 3, 2, 2]):
            p_mean = p.mean(dim=0) + 1e-5  
            Z = p_mean.sum(dim=0, keepdim=True)
            p_mean /= Z
            loss += 1 + torch.sum(p_mean * torch.log_softmax(p_mean, dim=0)) / np.log(dim)

    losses = {"H-loss": loss.item()}

    return loss, losses


def CLEVR_Concept_Match(out_dict: dict, args):
    """CLEVR concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    loss = torch.tensor(0.0, device=reprs.device)
    
    gt_img_1, gt_img_2, gt_img_3, gt_img_4 = concepts[:, :15], concepts[:, 15:30], concepts[:, 30:45], concepts[:, 45:60]
    p_img_1, p_img_2, p_img_3, p_img_4 = reprs[:, :15], reprs[:, 15:30], reprs[:, 30:45], reprs[:, 45:60]

    for pred, gt in zip([p_img_1, p_img_2, p_img_3, p_img_4], [gt_img_1, gt_img_2, gt_img_3, gt_img_4]):
        
        colors, shapes, materials, sizes = pred[:, :8], pred[:, 8:11], pred[:, 11:13], pred[:, 13:15]
        gt_colors, gt_shapes, gt_materials, gt_sizes = gt[:, :8], gt[:, 8:11], gt[:, 11:13], gt[:, 13:15]

        for pel, gtel in zip([colors, shapes, materials, sizes], [gt_colors, gt_shapes, gt_materials, gt_sizes]):

            filtered_gt = gtel[~torch.all(gtel == -1, dim=-1)]
            filtered_pred = pel[~torch.all(gtel == -1, dim=-1)]

            assert torch.all(gtel == -1, dim=-1).sum() == torch.all(pel == 0, dim=-1).sum(), f"{(torch.all(gtel == -1, dim=-1) != torch.all(pel == 0, dim=-1)).nonzero(as_tuple=True)[0]}"

            if filtered_gt.size(0) > 0:
                loss += torch.nn.functional.nll_loss(filtered_pred.log(), torch.argmax(filtered_gt, dim=-1))

            assert loss != float("inf"), "Loss gone to infinity"

    print("Concept supervision loss", loss.item())

    losses = {"c-loss": loss.item()}

    return loss, losses

    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    loss = torch.tensor(0.0, device=reprs.device)

    # Split into 4 images per sample
    gt_parts = concepts.chunk(4, dim=1)  # Each is [batch, 15]
    pred_parts = reprs.chunk(4, dim=1)

    for pel, gtel in zip(pred_parts, gt_parts):
        mask = ~torch.all(gtel == -1, dim=-1)
        if mask.any():
            filtered_pred = pel[mask]
            filtered_gt = gtel[mask]

            filtered_pred = torch.clamp(filtered_pred + 1e-5, min=1e-5)

            with torch.no_grad():
                Z = torch.sum(filtered_pred, dim=-1, keepdim=True) + 1e-5  # Prevent division by zero
        filtered_pred /= Z

        filtered_pred = torch.clamp(filtered_pred, min=1e-5)

        # Debugging checks
        assert torch.all(torch.isfinite(filtered_pred)), "filtered_pred contains NaN or Inf!"
        assert torch.all(filtered_gt.argmax(dim=-1) >= 0), "filtered_gt contains negative indices!"
        assert torch.all(filtered_gt.argmax(dim=-1) < filtered_pred.shape[-1]), "filtered_gt has out-of-bounds indices!"

        loss += torch.nn.functional.nll_loss(filtered_pred.log(), filtered_gt.argmax(dim=-1))

    loss /= 4 # 4 objects circa

    assert loss.isfinite()

    print("Concept supervision loss:", loss.item())

    losses = {"c-loss": loss.item()}

    return loss, losses



def CLEVR_REC_Match(out_dict: dict, args):
    """CLEVR concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    # Extract data
    recs, inputs, mus, logvars = (
        out_dict["RECS"],
        out_dict["INPUTS"],
        out_dict["MUS"],
        out_dict["LOGVARS"],
    )

    assert inputs.shape == recs.shape, f"{inputs.shape}-{recs.shape}"

    criterion = torch.nn.MSELoss()

    # Compute mask in batch
    mask = (inputs == -1).all(dim=2).all(dim=2).all(dim=2)
    valid_indices = ~mask

    if valid_indices.any():
        recon = criterion(recs[valid_indices], inputs[valid_indices])
    else:
        recon = torch.tensor(0.0, device=inputs.device)

    # Compute KLD Loss
    kld = (-0.5 * (1 + logvars[valid_indices] - mus[valid_indices]**2 - logvars[valid_indices].exp()).sum(1).mean()).abs()

    # Print loss info
    print("Reconstruction Loss:", recon.item())

    # Compute total loss
    losses = {"recon-loss": recon.item(), "kld": kld.item()}
    total_loss = recon + args.beta * kld

    return total_loss, losses


def clevr_logic(vector):
    # Predefined mappings for faster access
    colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    shapes = ["cube", "sphere", "cylinder"]
    materials = ["rubber", "metal"]
    sizes = ["large", "small"]

    # Track the found classes
    class_1_found = {'large_cube': False, 'large_cylinder': False}
    class_2_found = {'small_metal_cube': False, 'small_sphere': False}
    class_3_found = {'large_blue_sphere': False, 'small_yellow_sphere': False}

    # Loop through the vector and update class found flags
    for obj in vector:
        presence, color_idx, shape_idx, material_idx, size_idx = obj

        if presence == 0:
            continue

        # Directly retrieve properties from the indexes
        color = colors[color_idx]
        shape = shapes[shape_idx]
        material = materials[material_idx]
        size = sizes[size_idx]

        # Check Class 1 conditions
        if size == 'large' and shape == 'cube' and color == 'gray':
            class_1_found['large_cube'] = True
        elif size == 'large' and shape == 'cylinder':
            class_1_found['large_cylinder'] = True

        # Check Class 2 conditions
        if size == 'small' and material == 'metal' and shape == 'cube':
            class_2_found['small_metal_cube'] = True
        elif size == 'small' and shape == 'sphere' and material == 'metal':
            class_2_found['small_sphere'] = True

        # Check Class 3 conditions
        if size == 'large' and color == 'blue' and shape == 'sphere':
            class_3_found['large_blue_sphere'] = True
        elif size == 'small' and color == 'yellow' and shape == 'sphere':
            class_3_found['small_yellow_sphere'] = True

    class_1 = all(class_1_found.values())
    class_2 = all(class_2_found.values())
    class_3 = all(class_3_found.values())

    # Return appropriate class number based on the found conditions
    if sum([class_1, class_2, class_3]) == 1:
        if class_1:
            return 0
        elif class_2:
            return 1
        elif class_3:
            return 2

    return 3  # No matching class or not interesting


def CLEVR_Knowledge_Match(out_dict: dict, args):
    """Addmnist knowledge match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    model = out_dict["MODEL"]

    n_samples = int(500 * args.perc_k) # 500 at most for knowledge supervision
    if n_samples == 0:
        return torch.tensor(0.0), {"knowledge-loss": 0.0}

    presence = torch.randint(0, 2, (n_samples, 4), device=args.device).squeeze()
    color = torch.randint(0, 8, (n_samples, 4), device=args.device).squeeze()
    shape = torch.randint(0, 3, (n_samples, 4), device=args.device).squeeze()
    material = torch.randint(0, 2, (n_samples, 4), device=args.device).squeeze()
    size = torch.randint(0, 2, (n_samples, 4), device=args.device).squeeze()

    objects = torch.stack([presence, color, shape, material, size], dim=-1)
    labels = []
    for i in range(len(objects)):
        labels.append(clevr_logic(objects[i]))
    labels = torch.tensor(labels, device=args.device).long()

    logits = []
    for i in range(n_samples):
        l = [
            presence[i:i+1].float().unsqueeze(-1),
            torch.nn.functional.one_hot(color[i:i+1], 8).float(),
            torch.nn.functional.one_hot(shape[i:i+1], 3).float(),
            torch.nn.functional.one_hot(material[i:i+1], 2).float(),
            torch.nn.functional.one_hot(size[i:i+1], 2).float(),
        ]
        logits.append(torch.cat(l, dim=-1))

    logits = torch.stack(logits, dim=0).squeeze(1)

    loss = F.nll_loss(torch.log(model.get_pred_from_prob(logits, presence=True)), labels)

    print("The loss is", loss.item())

    losses = {"knowledge-loss": loss.item()}
    return loss, losses



def BOIA_Knowledge_Match(out_dict: dict, args):
    """Addmnist knowledge match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    model = out_dict["MODEL"]

    n_samples = int(2000 * args.perc_k) # 1000 at most for knowledge supervision
    if n_samples == 0:
        return torch.tensor(0.0), {"knowledge-loss": 0.0}
    
    config = torch.randint(0, 2, (n_samples, 21))
    labels = get_boia_actions_from_concepts(config)

    config = torch.tensor(config).to(args.device).to(torch.float32)
    labels = torch.tensor(labels).to(args.device)

    def CE_forloop(y_pred, y_true):

        y_trues = torch.split(y_true, 1, dim=-1)
        y_preds = torch.split(y_pred, 2, dim=-1)

        loss = 0
        for i in range(4):

            true = y_trues[i].view(-1)
            pred = y_preds[i]

            ## add small offset to avoid NaNs
            pred = pred + 1e-5
            with torch.no_grad():
                Z = torch.sum(pred, dim=0, keepdim=True)
            pred = pred / Z

            assert torch.max(pred) < 1, pred
            assert torch.min(pred) > 0, pred

            loss_i = F.nll_loss(pred.log(), true.to(torch.long))
            loss += loss_i / 4

            assert loss_i > 0, pred.log()

        return loss

    loss = CE_forloop(torch.log(model.get_pred_from_prob(config, full=True)), labels)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    print("The loss is", loss.item())

    losses = {"knowledge-loss": loss.item()}

    return loss, losses
