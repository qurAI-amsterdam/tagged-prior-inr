import os

# set cuda visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
from pathlib import Path
import torch
import subprocess
from utils.coords import KEY_SAX_VIEW

from models import ImplicitRegistratorSequence
from utils.io import get_images_with_segmentations, get_experiments_folder
from canonical.alignment import get_canonical_sequence_aligned
from postprocessing.pipeline import post_process_sequence_completed
import sys
import multiprocessing
from functools import partial
import configparser
import logging
import ast

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("registration.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


def load_all_kwargs(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    out = {}
    for section in cfg.sections():
        for key, raw in cfg.items(section):
            out[key] = parse_value(raw)
    return out, cfg


def parse_value(val):
    """
    Try to coerce "True"/"1"/"1.0"/"[...]" into Python types;
    fall back to string if it won't literal_eval.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def check_cuda_enabled():
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        if "NVIDIA-SMI" in output and "CUDA" in output:
            if torch.cuda.is_available():
                logging.info("CUDA is available.")
                return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")

    logging.info("CUDA is not available or nvidia-smi is not found.")
    return False


def run_registration(
    data_dict,
    patid,
    path_to_pat_experiments,
    save_net,
    cardiac_views,
    kwargs,
):

    spacing = data_dict["spacing"]
    sequence = data_dict["sequence"]

    ImpReg = ImplicitRegistratorSequence(
        sequence=sequence,
        spacing_xyz=spacing,
        cardiac_views=cardiac_views,
        **kwargs,
    )

    ImpReg.exper_dir = Path(path_to_pat_experiments)
    ImpReg.model_dir = ImpReg.exper_dir / "models"
    ImpReg.save_folder = ImpReg.exper_dir

    if kwargs.get("use_prior"):
        prior_path = Path(kwargs["path_to_prior"])
        logging.info(f"Loading prior weights from {prior_path}")
        ImpReg.network.load_state_dict(torch.load(prior_path, map_location="cpu"), strict=False)

    registration_start = time.time()
    ImpReg.fit_sequence()
    print(f"Registration time: {time.time() - registration_start}")

    post_process_start = time.time()

    if save_net:
        if not os.path.exists(ImpReg.model_dir):
            os.makedirs(ImpReg.model_dir, exist_ok=True)
        torch.save(ImpReg.network.state_dict(), ImpReg.model_dir / "final_model.pth")

    result_dict_T = post_process_sequence_completed(
        ImpReg,
        spacing_xyz=spacing,
        save_dir=ImpReg.exper_dir,
        save_npz=True,
        compute_masks=True,
        compute_physical_dvf=False,
        compute_dice=True,
        kwargs=kwargs,
    )
    print(f"Post-processing time: {time.time() - post_process_start}")

    # Always store per-component loss history in the result dict
    result_dict_T["loss_components"] = getattr(ImpReg, "loss_components", None)
    result_dict_T["loss_list"] = ImpReg.loss_list

    print(f"Post-processing time: {time.time() - post_process_start}")


def process_patients(
    patid,
    path_to_experiments,
    path_to_images_sax,
    path_to_segmentations_sax,
    cardiac_views,
    kwargs,
):
    start = time.time()
    cardiac_views = [KEY_SAX_VIEW]
    print(f"Running registration for patient {patid}")

    patid_basename = patid.split(".")[0]

    path_to_pat_experiments = Path(os.path.join(path_to_experiments, patid_basename))

    if not os.path.exists(path_to_pat_experiments):
        try:
            os.makedirs(path_to_pat_experiments, exist_ok=True)
            path_to_pat_sax = os.path.join(path_to_images_sax, patid)
            path_to_segmentations_sax = os.path.join(path_to_segmentations_sax, patid)

            img4d_sax, seg4d_sax = get_images_with_segmentations(
                path_to_pat_sax,
                path_to_segmentations_sax,
                **kwargs,
            )

            data_dict = get_canonical_sequence_aligned(
                img4d_sax,
                seg4d_sax,
                crop_ROI=kwargs["crop"],
            )

            print(f"Loading time: {time.time() - start}")

            run_registration(
                data_dict,
                patid,
                path_to_pat_experiments,
                kwargs["save_net"],
                cardiac_views,
                kwargs,
            )
        except Exception as e:
            logging.error(f"Error processing patient {patid}: {e}")
            print(f"Error processing patient {patid}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cardiac motion registration")
    parser.add_argument(
        "--config", default="registration_config.cfg",
        help="Path to config file (default: registration_config.cfg)"
    )
    parser.add_argument(
        "--gpu", default=None, type=str,
        help="GPU id(s) to use, e.g. '0' or '0,1'.  Sets CUDA_VISIBLE_DEVICES."
             " If omitted, uses whatever is already set in the environment."
    )
    parser.add_argument(
        "--worker-id", default=0, type=int,
        help="Index of this worker (0-based).  This worker processes patients"
             " at positions [worker-id, worker-id + n-workers, ...]."
             " Default: 0 (run all patients)."
    )
    parser.add_argument(
        "--n-workers", default=1, type=int,
        help="Total number of parallel workers (one per GPU).  Default: 1."
    )
    args = parser.parse_args()

    # Pin this process to a specific GPU before any CUDA initialisation.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logging.info(f"Worker {args.worker_id}/{args.n_workers}: CUDA_VISIBLE_DEVICES={args.gpu}")

    check_cuda_enabled()

    # LOAD CONFIG FILE
    cfg_name = args.config
    kwargs, config = load_all_kwargs(cfg_name)

    print(f"Loaded kwargs: {kwargs}")

    # DEFINE PATHS TO DATA AND SEGMENTATIONS
    root = Path(kwargs["root"])
    data_folder = Path(kwargs["path_to_data"])
    seg_folder = Path(kwargs["path_to_segmentation"])
    path_to_images_sax = root / data_folder
    path_to_segmentations_sax = root / seg_folder

    # DEFINE PATHS TO EXPERIMENTS
    experiment_folder_name = kwargs["experiment_folder_name"]
    path_to_experiments = get_experiments_folder(
        root, experiment_folder_name, addition=""
    )

    patid_list = sorted([
        file for file in os.listdir(path_to_images_sax) if file.endswith(".nii.gz")
    ])

    # Distribute work across workers: take every n_workers-th patient.
    # Worker 0 → [0, N, 2N, ...], Worker 1 → [1, N+1, 2N+1, ...], etc.
    if args.n_workers > 1:
        patid_list = patid_list[args.worker_id::args.n_workers]
        logging.info(
            f"Worker {args.worker_id}/{args.n_workers}: assigned {len(patid_list)} patients."
        )

    if kwargs["debug"]:
        import random

        patid_list = [pid for pid in patid_list if "51" in pid] # or "58" in pid
        kwargs["multiprocessing"] = False
        print(
            f"Debug mode: only running on patient {patid_list} with multiprocessing off."
        )

    cardiac_views = [KEY_SAX_VIEW]

    logging.info(
        f"Running registration for {len(patid_list)} from {kwargs['dataset']} patients: {patid_list}"
    )

    if kwargs["multiprocessing"]:
        logging.info("Using multiprocessing for registration.")
        multiprocessing.set_start_method("spawn")

        with multiprocessing.Pool(processes=4) as pool:
            partial_func = partial(
                process_patients,
                path_to_experiments=path_to_experiments,
                path_to_images_sax=path_to_images_sax,
                path_to_segmentations_sax=path_to_segmentations_sax,
                cardiac_views=cardiac_views,
                kwargs=kwargs,
            )
            pool.map(partial_func, patid_list)
    else:
        for patid in patid_list:
            process_patients(
                patid,
                path_to_experiments,
                path_to_images_sax,
                path_to_segmentations_sax,
                cardiac_views=cardiac_views,
                kwargs=kwargs,
            )

    print("finished")
