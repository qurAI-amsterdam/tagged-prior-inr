"""
Experiment configuration persistence utilities.

Provides saveExperimentSettings (previously in kwatsch/common.py).
"""

import yaml


def saveExperimentSettings(args, fname):
    """
    Serialise experiment settings (dict or argparse Namespace) to a YAML file.

    Parameters
    ----------
    args  : dict or argparse.Namespace
    fname : str  path to output YAML file
    """
    if isinstance(args, dict):
        with open(fname, "w") as fp:
            yaml.dump(args, fp)
    else:
        with open(fname, "w") as fp:
            yaml.dump(vars(args), fp)
