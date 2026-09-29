# Args module

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """Adds the arguments used by all the models.

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # dataset
    parser.add_argument(
        "--dataset",
        default="addmnist",
        type=str,
        choices=DATASET_NAMES,
        help="Which dataset to perform experiments on.",
    )
    parser.add_argument(
        "--task",
        default="addition",
        type=str,
        choices=[
            "addition",
            "sumparity",
            "sumparityrigged",
            "clevr",
            "boia",
        ],
        help="Which task to choose: 'addition' (MNIST-Add), 'sumparity'/'sumparityrigged' "
        "(MNIST-SumParity, in- and out-of-distribution variants), 'clevr' (Clevr) or 'boia' (BDD-OIA).",
    )
    # model settings
    parser.add_argument(
        "--model",
        type=str,
        default="mnistdpl",
        help="Model name.",
        choices=get_all_models(),
    )
    parser.add_argument(
        "--c_sup",
        type=float,
        default=0,
        help="Fraction of concept supervision on concepts",
    )
    parser.add_argument(
        "--k_sup",
        type=float,
        default=0,
        help="Weight of the knowledge supervision",
    )
    parser.add_argument(
        "--which_c",
        type=int,
        nargs="+",
        default=[-1],
        help="Which concepts explicitly supervise (-1 means all)",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        default=False,
        help="Process the image as a whole.",
    )
    parser.add_argument(
        "--splitted",
        action="store_true",
        default=False,
        help="Create different encoders.",
    )
    parser.add_argument(
        "--entropy",
        action="store_true",
        default=False,
        help="Activate entropy on batch.",
    )
    parser.add_argument(
        "--contrastive",
        action="store_true",
        default=False,
        help="Activate contrastive learning.",
    )
    parser.add_argument(
        "--w_con",
        type=float,
        default=1,
        help="Weight of contrastive learning",
    )
    # weight of mitigation
    parser.add_argument("--gamma", type=float, default=1, help="Weight of mitigation")
    # additional hyperparams
    parser.add_argument(
        "--w_rec", type=float, default=1, help="Weight of Reconstruction"
    )
    parser.add_argument("--beta", type=float, default=2, help="Multiplier of KL")
    parser.add_argument("--w_h", type=float, default=1, help="Weight of entropy")
    parser.add_argument("--w_c", type=float, default=1, help="Weight of concept sup")

    # optimization params
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)."
    )
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup epochs.")
    parser.add_argument(
        "--exp_decay", type=float, default=1.0, help="Exp decay of learning rate."
    )

    # learning hyperams
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="Number of epochs per task."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")

    # deep ensembles
    parser.add_argument(
        "--boia-model",
        default="ce",
        choices=["ce", "bce"],
        type=str,
        help="Training using CE or BCE",
    )

    parser.add_argument(
        "--backbone",
        default="conceptizer",
        choices=["conceptizer", "neural"],
        type=str,
        help="Which backbone to use",
    )
    # parameters for tuning
    parser.add_argument(
        "--count",
        default=30,
        type=int,
        help="Number of hyper-params configurations that has to be tried during tuning.",
    )
    parser.add_argument(
        "--val_metric",
        default="f1",
        choices=["accuracy", "f1"],
        type=str,
        help="Validation metric that has to be minimized during hyper-parameter tuning.",
    )
    parser.add_argument(
        "--tuning",
        default=False,
        action="store_true",
        help="Whether to perform tuning of the specified model.",
    )
    parser.add_argument(
        "--proj_name",
        default="dsltune",
        type=str,
        help="Weights and Biases project name where the runs have to be logged.",
    )
    parser.add_argument(
        "--entity",
        default="user",
        type=str,
        help="Weights and Biases project entity (username).",
    )
    parser.add_argument(
        "--boia-ood-knowledge",
        default=False,
        action="store_true",
        help="Whether to employ BOIA OOD-knowledge (Ambulance) only for DPL",
    )
    parser.add_argument(
        "--device",
        type=str, 
        default=None,
        help="Device to use"
    )
    parser.add_argument(
        "--jrs",
        action="store_true",
        default=False,
        help="Use jrs eval"
    )
    parser.add_argument(
        "--perc_k",
        default=0.0,
        type=float,
        help="Percentage of knowledge supervision.",
    )
    parser.add_argument(
        "--w_senn",
        default=1.0,
        type=float,
        help="Weight of the jacobian regularization loss.",
    )
    parser.add_argument("-es", "--eps_sym", default=0.2807344052335263, type=float)
    parser.add_argument("-er", "--eps_rul", default=0.1077119516324264, type=float)

    parser.add_argument(
        "--multi_linear",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use a multilinear cbm.",
    )

def add_management_args(parser: ArgumentParser) -> None:
    """Adds the arguments used in management

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # random seed
    parser.add_argument("--seed", type=int, default=None, help="The random seed.")
    # verbosity
    parser.add_argument("--notes", type=str, default=None, help="Notes for this run.")
    parser.add_argument("--non_verbose", action="store_true")
    # logging
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="Enable wandb logging -- set name of project",
    )
    # checkpoints
    parser.add_argument(
        "--checkin",
        type=str,
        default=None,
        help="location and path FROM where to load ckpt.",
    )
    parser.add_argument(
        "--checkout",
        action="store_true",
        default=False,
        help="save the model to data/ckpts.",
    )
    # post-hoc evaluation
    parser.add_argument(
        "--posthoc",
        action="store_true",
        default=False,
        help="Used to evaluate only the loaded model",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Used to evaluate on the validation set for hyperparameters search",
    )


def add_test_args(parser: ArgumentParser) -> None:
    """Arguments for the Test part of the code

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # random seed
    parser.add_argument(
        "--use_ood", action="store_true", help="Use Out of Distribution test samples."
    )
