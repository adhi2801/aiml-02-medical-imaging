"""Run notebooks/kaggle_train.ipynb on a Kaggle GPU without a public GitHub repo.

Packs the committed source (git archive HEAD) into a private Kaggle dataset,
then pushes the notebook as a GPU kernel with that dataset and the RSNA
competition attached. Re-running uploads a new dataset version and a new
kernel version.

    python scripts/kaggle_push.py            # upload source + start the run
    python scripts/kaggle_push.py --status   # check the run
    python scripts/kaggle_push.py --fetch    # download outputs to kaggle_out/

Prerequisite: accept the rules of rsna-pneumonia-detection-challenge on kaggle.com.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
SRC_SLUG = "aiml02-src"
KERNEL_SLUG = "aiml02-train"
COMPETITION = "rsna-pneumonia-detection-challenge"


def push_source(api, user, stage):
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(stage / "repo.tar.gz"), "HEAD"],
                   cwd=ROOT, check=True)
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "aiml02 source", "id": f"{user}/{SRC_SLUG}", "licenses": [{"name": "other"}]}))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    try:
        api.dataset_status(f"{user}/{SRC_SLUG}")
        exists = True
    except Exception:
        exists = False
    if exists:
        api.dataset_create_version(str(stage), version_notes=f"source @ {sha}", quiet=True)
    else:
        api.dataset_create_new(str(stage), public=False, quiet=True)
    print(f"source @ {sha} -> {user}/{SRC_SLUG}")


def push_kernel(api, user, stage):
    nb = stage / "kaggle_train.ipynb"
    nb.write_bytes((ROOT / "notebooks" / "kaggle_train.ipynb").read_bytes())
    (stage / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": nb.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{SRC_SLUG}"], "competition_sources": [COMPETITION],
    }))
    api.kernels_push(str(stage))
    print(f"kernel started: https://www.kaggle.com/code/{user}/{KERNEL_SLUG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    kernel = f"{user}/{KERNEL_SLUG}"

    if args.status:
        print(api.kernels_status(kernel))
    elif args.fetch:
        out = ROOT / "kaggle_out"
        api.kernels_output(kernel, path=str(out))
        print(f"outputs in {out}")
    else:
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as ker:
            push_source(api, user, Path(src))
            push_kernel(api, user, Path(ker))


if __name__ == "__main__":
    main()
