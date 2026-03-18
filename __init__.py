import os

_TAG = "sdxl-equirect"

# All model dirs are symlinked to GCS FUSE in Cloud Run.
# Save to /tmp to avoid slow/unreliable writes through FUSE.
_CKPT_LOCAL_DIR = "/tmp/sdxl_equirect_checkpoints"

_MODELS = [
    {
        "label": "JuggernautXL v9 RunDiffusionPhoto v2",
        "repo_id": "RunDiffusion/Juggernaut-XL-v9",
        "hf_path": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "subdir": "checkpoints",
        "filename": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "local_dir": _CKPT_LOCAL_DIR,
    },
]


def _log(msg):
    print(f"[{_TAG}] {msg}", flush=True)


def _download_models():
    try:
        import requests
    except ImportError:
        _log("ERROR: requests not available, cannot download models")
        return

    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for model in _MODELS:
        local_dir = model["local_dir"]
        local_path = os.path.join(local_dir, model["filename"])

        if os.path.exists(local_path):
            _log(f"Already exists: {model['filename']}")
            continue

        os.makedirs(local_dir, exist_ok=True)
        url = f"https://huggingface.co/{model['repo_id']}/resolve/main/{model['hf_path']}"
        _log(f"Downloading {model['label']} ...")

        try:
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
            size_gb = os.path.getsize(local_path) / 1e9
            _log(f"Saved: {local_path} ({size_gb:.1f} GB)")
        except Exception as e:
            _log(f"ERROR downloading {model['label']}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)

    _log("All models ready.")


def _register_extra_paths():
    try:
        import folder_paths
        folder_paths.add_model_folder_path("checkpoints", _CKPT_LOCAL_DIR)
    except Exception as e:
        _log(f"WARNING: could not register checkpoint path: {e}")


_download_models()
_register_extra_paths()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
