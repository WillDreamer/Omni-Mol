import time
from huggingface_hub import snapshot_download
#huggingface上的模型名称
repo_id = "meta-llama/Llama-3.2-1B-Instruct"
#本地存储地址
local_dir = "/root/OmniMol/checkpoints"
cache_dir = '/root/.cache/huggingface'
while True:
    try:
        snapshot_download(cache_dir=cache_dir,
        local_dir=local_dir,
        repo_id=repo_id,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.model", "*.json", "*.bin",
        "*.py", "*.md", "*.txt","*.safetensors",],
        ignore_patterns=[ "*.msgpack",
        "*.h5", "*.ot",],
        )
    except Exception as e :
        print(e)
        # time.sleep(5)
    else:
        print('下载完成')
        break
