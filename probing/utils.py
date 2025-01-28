import datetime
import os
from pathlib import Path
import json
import shutil
from tqdm import tqdm

def timestamp():
    return datetime.datetime.now().strtime("%Y:%m:%d %H:%M:%S")


def copy_dataset(logdir, targetdir):
    """
    Sets up dataset from an existing dreamerv3 logdir
    
    Args:
        path (str): Path to the logdir
    """

    # retrieve dataset
    path = Path(logdir)
    if not path.is_dir():
        raise ValueError(f"Provided path {path} is not a directory")
    env_path = path / "env0"

    target = Path(targetdir)
    os.makedirs(target, exist_ok=True)

    for i, file in enumerate(tqdm(os.listdir(env_path))):
        if "stats" in file and file.endswith(".jsonl"):
            shutil.copy(env_path / file, target / file)
    

if __name__== "__main__":
    logdir = "/workspace/assets/logdir/crafter/20250111T122207"
    path_to_save = "/workspace/assets/probing/dataset/crafter"
    copy_dataset(logdir, path_to_save)