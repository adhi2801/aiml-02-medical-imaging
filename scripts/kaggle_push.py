"""Run notebooks/kaggle_train.ipynb on a Kaggle GPU without a public GitHub repo.

Packs the committed source (git archive HEAD) into a private Kaggle dataset,
then pushes the notebook as a GPU kernel with that dataset and the RSNA
competition attached. Re-running uploads a new dataset version and a new
kernel version.

    python scripts/kaggle_push.py            # upload source + start the run
    python scripts/kaggle_push.py --status   # check the run
    python scripts/kaggle_push.py --eval-only  # re-evaluate models/classifier.pt, no training
    python scripts/kaggle_push.py --log      # print the run log
    python scripts/kaggle_push.py --fetch    # download outputs to kaggle_out/

Prerequisite: accept the rules of rsna-pneumonia-detection-challenge on kaggle.com.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
SRC_SLUG = "aiml02-src"
KERNEL_SLUG = "aiml02-train"
MODELS_SLUG = "aiml02-models"
COMPETITION = "rsna-pneumonia-detection-challenge"


def push_dataset(api, user, slug, title, stage, notes):
    """Create or version a private dataset from `stage`, then wait until Kaggle has processed it."""
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}))
    try:
        api.dataset_status(f"{user}/{slug}")
        exists = True
    except Exception:
        exists = False
    if exists:
        api.dataset_create_version(str(stage), version_notes=notes, quiet=True)
    else:
        api.dataset_create_new(str(stage), public=False, quiet=True)
    # A kernel started while the new version is still processing would run the previous files
    for _ in range(60):
        if str(api.dataset_status(f"{user}/{slug}")).lower() == "ready":
            break
        time.sleep(5)
    else:
        raise SystemExit(f"{slug}: dataset version still not ready after 5 min; re-run later")
    print(f"{notes} -> {user}/{slug}")


def push_source(api, user, stage):
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(stage / "repo.tar.gz"), "HEAD"],
                   cwd=ROOT, check=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    push_dataset(api, user, SRC_SLUG, "aiml02 source", stage, f"source @ {sha}")


def push_checkpoint(api, user, stage):
    """The trained classifier; when attached, the notebook evaluates it instead of training a new one."""
    ckpt = ROOT / "models" / "classifier.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found: run training first or fetch it from a previous Kaggle run")
    shutil.copy2(ckpt, stage / ckpt.name)
    push_dataset(api, user, MODELS_SLUG, "aiml02 released checkpoint", stage, "classifier.pt")


def push_kernel(api, user, stage, eval_only=False):
    nb = stage / "kaggle_train.ipynb"
    nb.write_bytes((ROOT / "notebooks" / "kaggle_train.ipynb").read_bytes())
    (stage / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": nb.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{SRC_SLUG}"] + ([f"{user}/{MODELS_SLUG}"] if eval_only else []),
        "competition_sources": [COMPETITION],
    }))
    resp = api.kernels_push(str(stage))
    err = getattr(resp, "error", None)
    if err:  # kernels_push reports failures in the response instead of raising
        raise SystemExit(f"kernel push failed: {err}")
    print(f"kernel started: {getattr(resp, 'url', None) or f'https://www.kaggle.com/code/{user}/{KERNEL_SLUG}'}")


def fetch_log(api, user):
    """Only the run log. kernels_output pages through every output file before writing the log."""
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    with api.build_kaggle_client() as kaggle:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug, request.page_size = user, KERNEL_SLUG, 1
        log = kaggle.kernels.kernels_api_client.list_kernel_session_output(request).log or "[]"
    lines = [e.get("data", "") for e in json.loads(log)]
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--log", action="store_true", help="print the run log")
    ap.add_argument("--eval-only", action="store_true",
                    help="upload models/classifier.pt and only re-run data prep + evaluation (no training)")
    args = ap.parse_args()

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    kernel = f"{user}/{KERNEL_SLUG}"

    if args.status:
        print(api.kernels_status(kernel))
    elif args.log:
        print(fetch_log(api, user))
    elif args.fetch:
        out = ROOT / "kaggle_out"
        api.kernels_output(kernel, path=str(out))
        print(f"outputs in {out}")
    else:
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as ker,                 tempfile.TemporaryDirectory() as mdl:
            push_source(api, user, Path(src))
            if args.eval_only:
                push_checkpoint(api, user, Path(mdl))
            push_kernel(api, user, Path(ker), args.eval_only)


if __name__ == "__main__":
    main()
