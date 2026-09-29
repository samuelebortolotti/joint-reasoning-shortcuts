# Module which contains the code for training a model
import torch
import numpy as np
import wandb
import csv
import os
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.wandb_logger import *
from utils.status import progress_bar
from datasets.utils.base_dataset import BaseDataset
from models.mnistdpl import MnistDPL
from utils.dpl_loss import ADDMNIST_DPL
from utils.metrics import (
    evaluate_metrics,
    evaluate_mix,
    mean_entropy,
    accuracy_binary,
)
from utils.generative import conditional_gen, recon_visaulization
from utils import fprint
import matplotlib.pyplot as plt
from warmup_scheduler import GradualWarmupScheduler
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, f1_score, accuracy_score
import numpy as np
from matplotlib import colors
from utils.jrs_utils import *


def visualize_rules(rules, name, N):
    fig, ax = plt.subplots(figsize=(7, 7))
    rules = rules.detach().cpu().numpy().squeeze()
    cmap = colors.ListedColormap(["#7b68ee"])
    
    ax.set_xticks([i for i in range(N)])
    ax.set_yticks([i for i in range(N)])
    ax.set_yticklabels([i for i in range(N)], rotation=90)

    ax.invert_xaxis()
    ax.xaxis.tick_top()

    ax.tick_params(axis="both", which="both", length=0, labelsize=15)
    ax.imshow(rules, vmin=0, vmax=0, cmap=cmap)
    for i in range(N):
        for j in range(N):
            ax.text(
                i,
                j,
                str(rules[i][j]),
                ha="center",
                va="center",
                color="white",
                fontsize=15,
            )
    plt.savefig("{}_rules.png".format(name))
    plt.close()
    print("{}_rules.png".format(name))

def convert_to_categories(elements):
    # Convert vector of 0s and 1s to a single binary representation along the first dimension
    binary_rep = np.apply_along_axis(
        lambda x: "".join(map(str, x)), axis=1, arr=elements
    )
    return np.array([int(x, 2) for x in binary_rep])


def entropy(p):
    """Compute entropy given a probability distribution."""
    p = np.clip(p, 1e-15, 1)

    return -np.sum(p * np.log(p)) / np.log(len(p))


def compute_coverage(confusion_matrix):
    """Compute the coverage of a confusion matrix.

    Essentially this metric is
    """

    max_values = np.max(confusion_matrix, axis=0)
    clipped_values = np.clip(max_values, 0, 1)

    # Redefinition of soft coverage
    coverage = np.sum(clipped_values) / len(clipped_values)

    return coverage


def plot_confusion_matrix(
    y_true, y_pred, labels=None, title="Confusion Matrix", save_path=None
):
    """
    Generate and plot a confusion matrix using Matplotlib with normalization.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (array-like, optional): List of class labels (default: None).
        title (str, optional): Title of the plot (default: 'Confusion Matrix').
        save_path (str, optional): Path to save the plot image (default: None).
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)

    plt.tight_layout()
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")

    if save_path is not None:
        print("Saved", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    return cm


def plot_multilabel_confusion_matrix(
    y_true, y_pred, class_names, title, save_path=None
):
    y_true_categories = convert_to_categories(y_true.astype(int))
    y_pred_categories = convert_to_categories(y_pred.astype(int))

    to_rtn_cm = confusion_matrix(y_true_categories, y_pred_categories)

    cm = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    num_rows = (num_classes + 4) // 5  # Calculate the number of rows needed

    plt.figure(figsize=(20, 4 * num_rows))  # Adjust the figure size

    for i in range(num_classes):
        plt.subplot(num_rows, 5, i + 1)  # Set the subplot position
        plt.imshow(cm[i], interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Class: {class_names[i]}")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0", "1"])
        plt.yticks(tick_marks, ["0", "1"])

        fmt = ".0f"
        thresh = cm[i].max() / 2.0
        for j in range(cm[i].shape[0]):
            for k in range(cm[i].shape[1]):
                plt.text(
                    k,
                    j,
                    format(cm[i][j, k], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i][j, k] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.suptitle(title)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

    return to_rtn_cm


def plot_actions_confusion_matrix(c_true, c_pred, title, save_path=None):

    # Define scenarios and corresponding labels
    scenarios = {
        "forward": [slice(0, 3), slice(0, 3)],
        "stop": [slice(3, 9), slice(3, 9)],
        #'forward_stop': [slice(None, 9), slice(None, 9)],
        "left": [slice(9, 15), slice(9, 15)],
        "right": [slice(15, 21), slice(15, 21)],
    }

    to_rtn = {}

    # Plot confusion matrix for each scenario
    for scenario, indices in scenarios.items():

        g_true = convert_to_categories(c_true[:, indices[0]].astype(int))
        c_pred_scenario = convert_to_categories(c_pred[:, indices[1]].astype(int))

        # Compute confusion matrix
        cm = confusion_matrix(g_true, c_pred_scenario)

        # Plot confusion matrix
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"{title} - {scenario}")
        plt.colorbar()

        n_classes = c_true[:, indices[0]].shape[1]

        tick_marks = np.arange(2**n_classes)
        plt.xticks(tick_marks, ["" for _ in range(len(tick_marks))])
        plt.yticks(tick_marks, ["" for _ in range(len(tick_marks))])

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        # Save or show plot
        if save_path:
            plt.savefig(f"{save_path}_{scenario}.png")
        else:
            plt.show()

        to_rtn.update({scenario: cm})

        plt.close()

    return to_rtn


def save_embeddings(dataset: BaseDataset, device, name):
    dataset.return_embeddings = True
    dataset.args.batch_size = 1  # 1 as batch size
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    ood_loader = dataset.ood_loader
    dataset.print_stats()

    encoder, _ = dataset.get_backbone()
    encoder.to(device)

    if not os.path.exists(f"embeddings_{name}"):
        os.makedirs(f"embeddings_{name}")

    for loader, subfolder_name in zip(
        [train_loader, val_loader, test_loader, ood_loader],
        ["train", "val", "test", "ood"],
    ):
        encoder.eval()

        # create the folder
        if not os.path.exists(f"embeddings_{name}/{subfolder_name}"):
            os.makedirs(f"embeddings_{name}/{subfolder_name}")

        for _, data in enumerate(loader):
            images, labels, concepts, names = data
            images, labels, concepts, names = (
                images.to(device),
                labels.to(device),
                concepts.to(device),
                names,
            )

            embeddings = encoder(images)
            embeddings = embeddings.squeeze(dim=0)

            # Save embeddings
            file_name = names[0]  # Remove extension
            save_path = os.path.join(
                f"embeddings_{name}/{subfolder_name}", f"{file_name}.pt"
            )
            torch.save(embeddings, save_path)


def save_predictions_to_csv(model, test_set, csv_name, dataset, args):
    model.eval()

    ys, y_true, cs, cs_true = None, None, None, None

    for data in tqdm(test_set, desc="Saving predictions to CSV..."):
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
        out_dict.update({"LABELS": labels, "CONCEPTS": concepts})

        if ys is None:
            ys = out_dict["YS"].cpu()
            y_true = out_dict["LABELS"].cpu()
            cs = out_dict["pCS"].cpu()
            cs_true = out_dict["CONCEPTS"].cpu()
            if args.model in ["mnistdsl", "clevrdsl"]:
                y_pred = out_dict["PRED"]
        else:
            ys = torch.concatenate((ys, out_dict["YS"].cpu()), dim=0)
            y_true = torch.concatenate((y_true, out_dict["LABELS"].cpu()), dim=0)
            cs = torch.concatenate((cs, out_dict["pCS"].cpu()), dim=0)
            cs_true = torch.concatenate((cs_true, out_dict["CONCEPTS"].cpu()), dim=0)
            if args.model in ["mnistdsl", "clevrdsl"]:
                y_pred = torch.concatenate((y_pred, out_dict["PRED"].cpu()), dim=0)

    if dataset.endswith("mnist"):
        y_true = y_true.unsqueeze(1)
        cs = cs.reshape(cs.shape[0], cs.shape[1] * cs.shape[2])
    elif "kand" in dataset:
        cs = cs.reshape(cs.shape[0], cs.shape[1] * cs.shape[2])
        cs_true = cs_true.reshape(cs_true.shape[0], cs_true.shape[1] * cs_true.shape[2])
    elif "xor" in dataset:
        y_true = y_true.unsqueeze(dim=1)
        cs = cs.reshape(cs.size(0), cs.size(1) * cs.size(2))
    elif "mnmath" in dataset:
        cs = torch.argmax(cs, dim=2)
        cs_true = cs_true.reshape(cs_true.size(0), cs_true.size(1) * cs_true.size(2))
    elif "clevr" in dataset:
        y_true = y_true.unsqueeze(1)

    if args.model in ["mnistdsl", "clevrdsl"]:
        ys = torch.unsqueeze(ys, dim=1)
        concatenated_tensor = (
            torch.concatenate((y_pred, ys, y_true, cs, cs_true), dim=1).cpu().detach().numpy()
        )
    else:
        concatenated_tensor = (
            torch.concatenate((ys, y_true, cs, cs_true), dim=1).cpu().detach().numpy()
        )

    # Save predictions to CSV file
    csv_path = os.path.join(csv_name)
    print("Saving predictions to", csv_path)

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(concatenated_tensor)


def train(model: MnistDPL, dataset: BaseDataset, _loss: ADDMNIST_DPL, args):
    """TRAINING

    Args:
        model (MnistDPL): network
        dataset (BaseDataset): dataset Kandinksy
        _loss (ADDMNIST_DPL): loss function
        args: parsed args

    Returns:
        None: This function does not return a value.
    """

    # name
    # csv_name = f"{args.dataset}-{args.model}-lr-{args.lr}-seed-{args.seed}.csv"

    # best f1
    best_f1 = 0.0

    to_add = ""
    if args.multi_linear and "cbm" in args.model:
        to_add = "_multi_linear"

    dataset_path = f"./best_models_{args.dataset}"
    os.makedirs(
        dataset_path,
        exist_ok=True,
    )
    os.makedirs(
        dataset_path + f"/best_models_{args.dataset}_visual",
        exist_ok=True,
    )

    save_path = dataset_path + f"/best_model_{args.dataset}_{args.task}_{args.model}_{args.task}_{args.seed}{to_add}.pth"

    # Default Setting for Training
    model.to(model.device)

    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    dataset.print_stats()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, args.exp_decay)
    w_scheduler = None

    if args.warmup_steps > 0:
        w_scheduler = GradualWarmupScheduler(model.opt, 1.0, args.warmup_steps)

    if not args.tuning and args.wandb is not None:
        fprint("\n---wandb on\n")
        wandb.init(
            project=args.project,
            entity=args.wandb,
            name=str(args.dataset) + "_" + str(args.model),
            config=args,
        )

    fprint("\n--- Start of Training ---\n")

    # default for warm-up
    model.opt.zero_grad()
    model.opt.step()

    for epoch in range(args.n_epochs):
        model.train()

        ys, y_true, cs, cs_true = None, None, None, None
        
        import time
        # start = time.time()
        for i, data in enumerate(train_loader):
            # end = time.time()
            # print("Osti", end-start)
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            out_dict = model(images)
            if "rec" in args.model or args.contrastive or "senn" in args.model:
                out_dict.update({"INPUTS": images, "LABELS": labels, "CONCEPTS": concepts, "MODEL": model, "EPOCH": epoch})
            elif ("clevr" in args.model and args.k_sup > 0) or ("boia" in args.model and args.k_sup > 0):
                out_dict.update({"LABELS": labels, "CONCEPTS": concepts, "MODEL": model})
            else:
                out_dict.update({"LABELS": labels, "CONCEPTS": concepts})

            model.opt.zero_grad()

            loss, losses = _loss(out_dict, args)
            
            loss.backward()
            model.opt.step()

            if ys is None:
                ys = out_dict["YS"]
                y_true = out_dict["LABELS"]
                cs = out_dict["pCS"]
                cs_true = out_dict["CONCEPTS"]

                if args.model in ["mnistdsl", "clevrdsl"]:
                    y_pred = out_dict["PRED"]
            else:
                ys = torch.concatenate((ys, out_dict["YS"]), dim=0)
                y_true = torch.concatenate((y_true, out_dict["LABELS"]), dim=0)
                cs = torch.concatenate((cs, out_dict["pCS"]), dim=0)
                cs_true = torch.concatenate((cs_true, out_dict["CONCEPTS"]), dim=0)

                if args.model in ["mnistdsl", "clevrdsl"]:
                    y_pred = torch.concatenate((y_pred, out_dict["PRED"]), dim=0)

            if not args.tuning and args.wandb is not None:
                wandb_log_step(i, epoch, loss.item(), losses)

            if i % 10 == 0:
                progress_bar(i, len(train_loader) - 9, epoch, loss.item())

        if args.task == "mnmath":
            y_pred = (ys > 0.5).to(torch.long)
        elif args.model not in ["mnistdsl", "clevrdsl"]:
            y_pred = torch.argmax(ys, dim=-1)

        if args.task == "boia":
            acc, f1 = accuracy_binary(ys, y_true)

            print(
                "\n Train Label acc: ",
                acc,
                "Train Label f1",
                f1,
            )
        else:
            if "patterns" in args.task:
                y_true = y_true[:, -1]  # it is the last one

            if args.task == "mnmath":
                acc = (
                    (y_pred.flatten().detach().cpu() == y_true.flatten().detach().cpu()).sum().item()
                    / len(y_pred.flatten())
                    * 100
                )
            else: 
                acc = (
                    (y_pred.detach().cpu() == y_true.detach().cpu()).sum().item()
                    / len(y_true)
                    * 100
                )

            print(
                "\n Train acc: ",
                acc,
                "%",
                len(y_true),
            )

        model.eval()
        tloss, cacc, yacc, f1 = evaluate_metrics(model, val_loader, args)

        # update at end of the epoch
        if epoch < args.warmup_steps:
            w_scheduler.step()
        else:
            scheduler.step()
            if hasattr(_loss, "grade"):
                _loss.update_grade(epoch)

        if args.tuning:
            wandb.log({"accuracy": yacc})
            wandb.log({"f1": f1})
            wandb.log({"cacc": cacc})

        ### LOGGING ###
        fprint("  ACC C", cacc, "  ACC Y", yacc, "F1 Y", f1)

        if not args.tuning and f1 > best_f1:
            print("Saving...")
            # Update best F1 score
            best_f1 = f1

            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with F1 score: {best_f1}")

        if not args.tuning and args.wandb is not None:
            wandb_log_epoch(
                epoch=epoch,
                acc=yacc,
                cacc=cacc,
                tloss=tloss,
                lr=float(scheduler.get_last_lr()[0]),
            )

        if f1 > 90:
            fprint(f"## Early stopping: F1(Y) = {f1} ##")
            break

    if not args.tuning:
        if args.model in ["mnistdsldpl", "mnistdsl"]:
            _, rules = model.get_rules_matrix(eval=True)
            visualize_rules(rules, f"rm_{args.model}", model.n_facts)

        # Evaluate performances on val or test
        if args.validate:
            y_true, c_true, y_pred, c_pred, p_cs, p_ys, p_cs_all, p_ys_all = (
                evaluate_metrics(model, val_loader, args, last=True)
            )
        else:
            y_true, c_true, y_pred, c_pred, p_cs, p_ys, p_cs_all, p_ys_all = (
                evaluate_metrics(model, test_loader, args, last=True)
            )

        yf1 = None

        if "patterns" not in args.task and "clevr" not in args.task:
            yac, yf1 = evaluate_mix(y_true, y_pred)
            cac, cf1 = evaluate_mix(c_true, c_pred)
            h_c = mean_entropy(p_cs_all, model.n_facts)

            fprint(f"Concepts:\n    ACC: {cac}, F1: {cf1}")
            fprint(f"Labels:\n      ACC: {yac}, F1: {yf1}")
            fprint(f"Entropy:\n     H(C): {h_c}")

        if args.task == "boia":
            y_labels = ["stop", "forward", "left", "right"]

            concept_labels = [
                "green_light",
                "follow",
                "road_clear",
                "red_light",
                "traffic_sign",
                "car",
                "person",
                "rider",
                "other_obstacle",
                "left_lane",
                "left_green_light",
                "left_follow",
                "no_left_lane",
                "left_obstacle",
                "letf_solid_line",
                "right_lane",
                "right_green_light",
                "right_follow",
                "no_right_lane",
                "right_obstacle",
                "right_solid_line",
            ]

            plot_multilabel_confusion_matrix(
                y_true, y_pred, y_labels, "Labels", save_path=dataset_path + f"/best_models_{args.dataset}_visual/labels_{args.model}_{args.task}_{args.seed}{to_add}.png"
            )
            cfs = plot_actions_confusion_matrix(
                c_true, c_pred, "Concepts", save_path=dataset_path + f"/best_models_{args.dataset}_visual/total_concepts_{args.model}_{args.task}_{args.seed}{to_add}_"
            )
            cf = plot_multilabel_confusion_matrix(
                c_true, c_pred, concept_labels, "Concepts", save_path=dataset_path + f"/best_models_{args.dataset}_visual/total_concepts_{args.model}_{args.task}_{args.seed}{to_add}_"
            )

            print("Concept collapse", 1 - compute_coverage(cf))

            for key, value in cfs.items():
                print("Concept collapse", key, 1 - compute_coverage(value))
        
        elif args.task == "mnmath":
            y_labels = ["first", "second"]
            concept_labels = [
                ["{i}" for i in range(10) for _ in range(4)] 
            ]
            plot_multilabel_confusion_matrix(
                y_true, y_pred, y_labels, "Labels", save_path=dataset_path + f"/best_models_{args.dataset}_visual/labels_{args.model}_{args.task}_{args.seed}{to_add}.png"
            )
            cf = plot_confusion_matrix(
                c_true,
                c_pred,
                labels=dataset.get_concept_labels(),
                title="Concepts",
                save_path=dataset_path + f"/best_models_{args.dataset}_visual/concepts_{args.dataset}_{args.model}_{args.task}_lr_{args.lr}_{args.seed}{to_add}.png",
            )

            print("Concept collapse", 1 - compute_coverage(cf))
        else:

            if args.task in ["patterns", "mini_patterns"]:
                # the last one is the groundtruth on the final prediction
                y_true = y_true[:, -1]

            plot_confusion_matrix(
                y_true,
                y_pred,
                labels=dataset.get_labels(),
                title="Labels",
                save_path=dataset_path + f"/best_models_{args.dataset}_visual/labels_{args.dataset}_{args.model}_{args.task}_lr_{args.lr}_{args.seed}{to_add}.png",
            )

            if args.task in ["patterns", "mini_patterns"]:
                t_shapes = c_true[:, :3].reshape(-1)
                p_shapes = c_pred[:, :3].reshape(-1)
                t_colors = c_true[:, 3:6].reshape(-1)
                p_colors = c_pred[:, 3:6].reshape(-1)

                shapes_concepts, colors_concepts = dataset.get_concept_labels()

                cf_shapes = plot_confusion_matrix(
                    t_shapes,
                    p_shapes,
                    labels=shapes_concepts,
                    title="Concepts",
                    save_path=dataset_path + f"/best_models_{args.dataset}_visual/concepts_{args.dataset}_{args.model}_{args.task}_lr_{args.lr}_{args.seed}{to_add}-shapes.png",
                )
                print("Concept collapse shapes", 1 - compute_coverage(cf_shapes))

                cf_colors = plot_confusion_matrix(
                    t_colors,
                    p_colors,
                    labels=colors_concepts,
                    title="Concepts",
                    save_path=dataset_path + f"/best_models_{args.dataset}_visual/concepts_{args.dataset}_{args.model}_{args.task}_lr_{args.lr}_{args.seed}{to_add}-colors.png",
                )
                print("Concept collapse colors", 1 - compute_coverage(cf_colors))

            else:
                if args.dataset not in ["clevr"]:
                    cf = plot_confusion_matrix(
                        c_true,
                        c_pred,
                        labels=dataset.get_concept_labels(),
                        title="Concepts",
                        save_path=dataset_path + f"/best_models_{args.dataset}_visual/concepts_{args.dataset}_{args.model}_{args.task}_lr_{args.lr}_{args.seed}{to_add}.png",
                    )

                    print("Concept collapse", 1 - compute_coverage(cf))

        # load best
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))

        print("Best model is", save_path)

        if not args.tuning and args.wandb is not None:
            K = max(max(y_pred), max(y_true))

            wandb.log({"test-y-acc": yac * 100, "test-y-f1": yf1 * 100})
            wandb.log({"test-c-acc": cac * 100, "test-c-f1": cf1 * 100})

            wandb.log(
                {
                    "cf-labels": wandb.plot.confusion_matrix(
                        None, y_true, y_pred, class_names=[str(i) for i in range(K + 1)]
                    ),
                }
            )
            K = max(np.max(c_pred), np.max(c_true))
            wandb.log(
                {
                    "cf-concepts": wandb.plot.confusion_matrix(
                        None, c_true, c_pred, class_names=[str(i) for i in range(K + 1)]
                    ),
                }
            )

            if hasattr(model, "decoder"):
                list_images = make_grid(
                    conditional_gen(model),
                    nrow=8,
                )
                images = wandb.Image(list_images, caption="Generated samples")
                wandb.log({"Conditional Gen": images})

                list_images = make_grid(recon_visaulization(out_dict), nrow=8)
                images = wandb.Image(list_images, caption="Reconstructed samples")
                wandb.log({"Reconstruction": images})

            wandb.finish()

        # LOADER AND EVALUATE
        ood_loader = getattr(dataset, "ood_loader", None)
        loaders = [test_loader, ood_loader] if ood_loader else [test_loader]

        for n_eval, loader in enumerate(loaders):
            y_true, c_true, y_pred, c_pred, p_cs, p_ys, p_cs_all, p_ys_all = (
                evaluate_metrics(model, loader, args, last=True)
            )

            strategy = ""
            
            if args.c_sup > 0:
                strategy += f"c_sup_{args.w_c}"
            
            if args.k_sup > 0:
                strategy += f"_k_sup_{args.perc_k}"

            if args.entropy > 0:
                strategy += f"_entropy_{args.w_h}"

            if "rec" in args.model:
                strategy += f"_rec_{args.w_rec}"

            if args.contrastive > 0:
                strategy += f"_contrastive_{args.w_con}"

            yac, yf1 = evaluate_mix(y_true, y_pred)

            if yf1 > 0.9:
                to_add += "_optimal"

            base_filename = f"best_models_{args.dataset}/{args.dataset}_{args.model}_{args.task}_{args.seed}_which_c_{args.which_c}_{strategy}{to_add}"
            csv_filename = f"{base_filename}.csv"
            model_filename = f"{base_filename}.pth"

            torch.save(model.state_dict(), model_filename)

            pi = get_hungarian_permutation(c_pred, c_true, args.dataset)

            concept_targets, concept_preds = retrive_concepts_and_labels_hungarian(c_pred, c_true, pi, args.dataset)

            if args.model in ["mnistdpl", "clevrdpl", "boiadpl"]: 
                bf1, bacc = 0.0, 0.0
            else:
                if "clevr" in args.model:
                    bacc, bf1 = evaluate_knowledge_clevr(model, pi)
                elif "boia" in args.model:
                    bacc, bf1 = evaluate_knowledge_boia(model, pi, args.model)
                else:
                    bf1, bacc = low_taper_fade(model, pi, is_cbm = True if "cbm" in args.model else False, args=args, dataset=dataset, index=n_eval)

            if "clevr" in args.model:
                concept_accuracy, concept_f1 = compute_clevr_accuracy(concept_targets, concept_preds)
            elif "boia" in args.model:
                concept_accuracy = (concept_targets == concept_preds).all(axis=1).astype(float).mean().item()
                concept_f1 = f1_score(concept_targets.flatten(), concept_preds.flatten(), average="macro")
            else:
                concept_accuracy = accuracy_score(concept_targets, concept_preds)
                concept_f1 = f1_score(concept_targets, concept_preds, average="macro")

            if "boia" in args.model:
                n_ll = torch.nn.BCEWithLogitsLoss(reduction="mean")(torch.tensor(p_ys), torch.tensor(y_true)).item()
            else:
                n_ll = compute_nll(p_ys, y_true)

            if "clevr" in args.model:
                collapse = compute_clevr_collapse(predicted_concepts=concept_preds, true_concepts=concept_targets)
            elif "boia" in args.model:
                cf = plot_multilabel_confusion_matrix(
                    concept_targets, concept_preds, concept_labels, "Concepts", save_path=dataset_path + f"/best_models_{args.dataset}_visual/total_concepts_{args.model}_{args.task}_{args.seed}{to_add}_"
                )

                total_collapse = 0.0
                num_concepts = len(cfs)

                for key, value in cfs.items():
                    curr_collapse = 1 - compute_coverage(value)  # Assuming compute_coverage is defined elsewhere
                    total_collapse += curr_collapse

                collapse = total_collapse / num_concepts if num_concepts > 0 else 0.0
            else:
                collapse = compute_concept_collapse(concept_targets, concept_preds)

            # Inputs
            metrics = {
                "yacc":yac,               # Example values for metrics
                "yf1": yf1,
                "cacc": concept_accuracy,
                "cf1": concept_f1,
                "collapse": collapse,
                "betaf1": bf1,
                "betaacc": bacc,
                "nll": n_ll,
            }

            csv_header = ["yacc", "yf1", "cacc", "cf1", "collapse", "betaf1", "betaacc", "nll"]

            if "boia" in args.model:
                metrics.update({"collapser": cfs['right']}) 
                csv_header.append('collapser')
                metrics.update({"collapsefor": cfs['forward']}) 
                csv_header.append('collapsefor')
                metrics.update({"collapsestop": cfs['stop']}) 
                csv_header.append('collapsestop')
                metrics.update({"collapsel": cfs['left']}) 
                csv_header.append('collapsel')

            csv_row = [metrics[key] for key in csv_header]

            if n_eval == 0:
                with open(csv_filename, mode="w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_header)
                    writer.writerow(csv_row)
                    print(csv_row)

                print(f"CSV file created: {csv_filename}")

                torch.save(model.state_dict(), model_filename)
                print(f"Model file saved: {model_filename}")
            else:
                with open(csv_filename, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row)
                    print(csv_row)

                print("CSV file updated", csv_filename)

            