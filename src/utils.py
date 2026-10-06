
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fixe toutes les sources de hasard pour rendre les runs reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Renvoie le GPU si disponible, sinon le CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
