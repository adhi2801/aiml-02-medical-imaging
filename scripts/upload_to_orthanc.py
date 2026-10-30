"""Push demo DICOMs into Orthanc, the same way a modality would (REST STOW of raw .dcm files).

  python scripts/upload_to_orthanc.py data/demo_dicoms            # every .dcm in the folder
  python scripts/upload_to_orthanc.py data/raw/rsna/stage_2_train_images --from-test-split 40

--from-test-split N picks N held-out TEST images (balanced positive/negative) so the demo never
shows the model an image it was trained on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--url", default="http://localhost:8042")
    ap.add_argument("--user", default="orthanc")
    ap.add_argument("--password", default="orthanc")
    ap.add_argument("--from-test-split", type=int, metavar="N")
    args = ap.parse_args()

    if args.from_test_split:
        meta = pd.read_csv(ROOT / "data/processed/rsna/metadata.csv")
        test = meta[meta.split == "test"]
        n = args.from_test_split // 2
        picks = pd.concat([test[test.label == 1].sample(n, random_state=0),
                           test[test.label == 0].sample(args.from_test_split - n, random_state=0)])
        files = [args.folder / f"{pid}.dcm" for pid in picks.patient_id]
    else:
        files = sorted(args.folder.glob("*.dcm"))
    if not files:
        sys.exit(f"no .dcm files found in {args.folder}")

    ok = 0
    with httpx.Client(base_url=args.url, auth=(args.user, args.password), timeout=60) as c:
        for f in files:
            r = c.post("/instances", content=f.read_bytes(), headers={"Content-Type": "application/dicom"})
            status = r.json().get("Status", r.status_code) if r.is_success else r.status_code
            ok += r.is_success
            print(f"{f.name}: {status}")
    print(f"uploaded {ok}/{len(files)}")


if __name__ == "__main__":
    main()
