from huggingface_hub import snapshot_download
import os
model_path = os.path.join("/scratch-shared/", os.getenv("USER"), "LTX-Video")


snapshot_download("Lightricks/LTX-Video", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')
