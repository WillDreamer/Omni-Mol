import time
import os
from huggingface_hub import snapshot_download

cache_dir = '/root/.cache/huggingface'


repo_id = "MillenniumHL/omnimol"
if not os.path.exists("./data"):
    os.makedirs("./data")
local_dir = "./data/"

if os.listdir(local_dir) == []:
    while True:
        try:
            snapshot_download(cache_dir=cache_dir,
            local_dir=local_dir,
            repo_id=repo_id,
            local_dir_use_symlinks=False,
            resume_download=True,
            repo_type="dataset",
            )
        except Exception as e :
            print(e)
            # time.sleep(5)
        else:
            print('Data downloaded')
            break

repo_id = "MillenniumHL/omnimol-ckpts"
if not os.path.exists("./assets"):
    os.makedirs("./assets")
local_dir = "./assets"

if os.listdir(local_dir) == []:
    while True:
        try:
            snapshot_download(cache_dir=cache_dir,
            local_dir=local_dir,
            repo_id=repo_id,
            local_dir_use_symlinks=False,
            resume_download=True,
            )
        except Exception as e :
            print(e)
            # time.sleep(5)
        else:
            print('Model downloaded')
            break