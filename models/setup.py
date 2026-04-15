"""
RegistrationSetup: object construction, default arguments, network/optimizer init.

Contains:
  __init__           — full construction of ImplicitRegistratorSequence
  set_default_arguments — populate self.args with sensible defaults
  _registration_init    — build network, optimizer, scheduler, loss criterion
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from objectives import ncc
from models.coords import KEY_SAX_VIEW, KEY_4CH_VIEW


class RegistrationSetup:
    """
    Owns __init__ and the two setup helpers.

    Calling self._init_sequence(), self._init_coords(), self._init_temporal_coords()
    from __init__ is safe because Python resolves those on the fully-composed
    ImplicitRegistratorSequence class via MRO.
    """

    def __init__(
        self,
        sequence,
        spacing_xyz,
        cardiac_views=(KEY_SAX_VIEW, KEY_4CH_VIEW),
        **kwargs,
    ):
        # ── Core registration attributes ─────────────────────────────────────
        self.early_stopping = kwargs.get("early_stopping", False)
        self.stopped_at_epoch = 0
        self.device = "cuda"

        if isinstance(cardiac_views, tuple):
            cardiac_views = list(cardiac_views)
        self.cardiac_views = cardiac_views
        kwargs["cardiac_views"] = "_".join(cardiac_views)

        self.reg_with_mask = kwargs.get("reg_with_mask", True)
        self.use_mask_loss = kwargs.get("use_mask_loss", False)
        self.temporal_registration = True
        kwargs["temporal_registration"] = True

        self.exper_dir = Path(kwargs.get("exper_dir", "/"))
        self.model_dir = self.exper_dir / "models"
        kwargs["model_dir"] = str(self.model_dir)

        # Build defaults dict, then network / optimizer / loss criterion.
        self.set_default_arguments()
        self._registration_init(**kwargs)

        self.xyz_sequence = kwargs["xyz_sequence"]
        self.normalize_coords = False
        self.multiview = len(cardiac_views) > 1
        self.coords_scale_factor = None
        self.min_coord_offset = None

        if not self.exper_dir.is_dir():
            self.exper_dir.mkdir(parents=True)

        # ── Sequence-specific attributes ─────────────────────────────────────
        self.sequence = sequence
        self.spacing_xyz = spacing_xyz
        self.T = len(sequence)

        self.compute_physical_dvf = kwargs.get("compute_physical_dvf", False)
        self.convert_to_engineering = kwargs.get("convert_to_engineering", False)
        self.strain_sigma = float(kwargs.get("strain_gaussian_sigma", 1.0))

        # Coordinate initialisation (methods defined in TemporalCoordinates).
        self._init_sequence()
        self.cimage_fixed = self.reference_image
        self.cimage_moving = None  # no single moving frame in the temporal case
        self._init_coords()
        self._init_temporal_coords()

        # ── Temporal loss weights ─────────────────────────────────────────────
        self.temporal_delta = int(kwargs.get("temporal_delta", 1))

    def set_default_arguments(self):
        """Return the full set of default registration hyper-parameters."""
        self.args = {
            "exper_dir": "/home/jorg/expers/cmri_motion",
            "normalize_coords": True,
            "xyz_sequence": False,
            "mask": None,
            "mask_2": None,
            "wandb": False,
            "voxel_size": (1, 1, 1),
            "method": 1,
            "lr": 0.00001,
            "batch_size": 10000,
            "layers": [3, 256, 256, 256, 3],
            "raw_jacobian_regularization": False,
            "jacobian_regularization": False,
            "alpha_jacobian": 0.05,
            "background_weight": 0.001,
            "bendreg_paperversion": False,
            "alpha_bending": 10.0,
            "image_shape": (200, 200),
            "network": None,
            "epochs": 2500,
            "log_interval": 625,  # epochs // 4
            "verbose": True,
            "save_folder": "output",
            "network_type": "MLP",
            "gpu": torch.cuda.is_available(),
            "optimizer": "Adam",
            "loss_function": "ncc",
            "momentum": 0.5,
            "positional_encoding": False,
            "weight_init": True,
            "omega": 32,
            "seed": 1,
        }

    def _registration_init(self, **kwargs):
        """
        Initialise training state from kwargs (or defaults).

        Builds the SIREN network(s), optimizer, LR scheduler, and loss criterion.
        Reads all regularization flags and stores them as instance attributes.
        """
        # ── Training schedule ──────────────────────────────────────────────
        self.epochs = kwargs.get("epochs", self.args["epochs"])
        self.scheduler_arg = kwargs.get("scheduler", False)
        self.seed = kwargs.get("seed", self.args["seed"])
        self.log_interval = kwargs.get("log_interval", self.args["log_interval"])
        self.gpu = kwargs.get("gpu", self.args["gpu"])
        self.lr = kwargs.get("lr", self.args["lr"])
        self.momentum = kwargs.get("momentum", self.args["momentum"])
        self.optimizer_arg = kwargs.get("optimizer", self.args["optimizer"])
        self.loss_function_arg = kwargs.get("loss_function", self.args["loss_function"])
        self.verbose = kwargs.get("verbose", self.args["verbose"])
        self.save_folder = kwargs.get("save_folder", self.args["save_folder"])

        # ── Network architecture ────────────────────────────────────────────
        self.layers = kwargs.get("layers", self.args["layers"])
        self.weight_init = kwargs.get("weight_init", self.args["weight_init"])
        self.omega = kwargs.get("omega", self.args["omega"])

        # Input dimension: 3 spatial + 1 time
        self.input_dim = 4 if self.temporal_registration else 3
        self.layers[0] = self.input_dim

        # ── Output directory ────────────────────────────────────────────────
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.save_folder += "/"

        # ── Loss tracking buffers ───────────────────────────────────────────
        self.loss_list = [0] * self.epochs
        self.data_loss_list = [0] * self.epochs
        self.cycle_loss_list = [0] * self.epochs

        torch.manual_seed(self.seed)

        # ── Network construction ────────────────────────────────────────────
        self.network_from_file = (
            self.model_dir / kwargs["network"]
            if "network" in kwargs
            else self.args["network"]
        )
        self.network_type = kwargs.get("network_type", self.args["network_type"])

        from networks.siren import Siren

        self.network = Siren(self.layers, self.weight_init, self.omega)

        if self.network_from_file is not None:
            self.network.load_state_dict(torch.load(self.network_from_file))
            self.network.eval()
            if self.gpu:
                self.network.cuda()

        # ── Optimizer ──────────────────────────────────────────────────────
        m_params = list(self.network.parameters())
        opt_name = self.optimizer_arg.lower()
        optimizers = {
            "sgd": lambda: optim.SGD(m_params, lr=self.lr, momentum=self.momentum),
            "adamw": lambda: optim.AdamW(m_params, lr=self.lr),
            "adam": lambda: optim.Adam(m_params, lr=self.lr),
            "adadelta": lambda: optim.Adadelta(m_params, lr=self.lr),
        }
        if opt_name in optimizers:
            self.optimizer = optimizers[opt_name]()
        else:
            print(
                f"WARNING: {self.optimizer_arg!r} not recognised, falling back to SGD"
            )
            self.optimizer = optim.SGD(m_params, lr=self.lr, momentum=self.momentum)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            verbose=True,
            min_lr=1e10,
        )

        # ── Similarity loss ─────────────────────────────────────────────────
        loss_name = self.loss_function_arg.lower()
        loss_map = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "ncc": ncc.NCC(),
            "smoothl1": nn.SmoothL1Loss(beta=0.2),
            "huber": nn.HuberLoss(),
        }
        if loss_name in loss_map:
            self.criterion = loss_map[loss_name]
        else:
            print(
                f"WARNING: {self.loss_function_arg!r} not recognised, falling back to MSE"
            )
            self.criterion = nn.MSELoss()

        if self.gpu:
            self.network.cuda()

        # ── Misc training flags ─────────────────────────────────────────────
        self.mask = kwargs.get("mask", self.args["mask"])
        self.mask_2 = kwargs.get("mask_2", self.args["mask_2"])

        # ── Regularization flags ────────────────────────────────────────────
        self.raw_jacobian_regularization = kwargs.get(
            "raw_jacobian_regularization", self.args["raw_jacobian_regularization"]
        )
        self.jacobian_regularization = kwargs.get(
            "jacobian_regularization", self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = kwargs.get("alpha_jacobian", self.args["alpha_jacobian"])
        self.background_weight = kwargs.get(
            "background_weight", self.args["background_weight"]
        )
        self.bendreg_paperversion = kwargs.get(
            "bendreg_paperversion", self.args["bendreg_paperversion"]
        )
        self.alpha_bending = kwargs.get("alpha_bending", self.args["alpha_bending"])

        # ── Misc ────────────────────────────────────────────────────────────
        torch.manual_seed(self.seed)
        self.wandb = kwargs.get("wandb", self.args["wandb"])
        self.image_shape = kwargs.get("image_shape", self.args["image_shape"])
        self.batch_size = kwargs.get("batch_size", self.args["batch_size"])
