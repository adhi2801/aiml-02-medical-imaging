"""Fast CPU tests: run in CI on every push, no dataset or GPU needed."""
import numpy as np
import pydicom
import pytest
import torch
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from medimg.data.dataset import preprocess
from medimg.data.dicom import _parse_age, read_dicom
from medimg.eval.stats import bootstrap_auc, confusion_stats, ece, fit_temperature, sigmoid
from medimg.explain.gradcam import boxes_to_mask, cam_iou, cam_to_rgba, gradcam, pointing_hit
from medimg.models.classifier import CXRClassifier, load_checkpoint, save_checkpoint


def make_dicom(path, pixels: np.ndarray, photometric="MONOCHROME2", sex="F", age="058Y", view="AP"):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.Dataset()
    ds.file_meta = meta
    ds.PatientID, ds.PatientSex, ds.PatientAge, ds.ViewPosition, ds.Modality = "p1", sex, age, view, "CR"
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = generate_uid(), generate_uid(), meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


@pytest.fixture
def gradient():
    return np.tile(np.linspace(0, 4095, 64, dtype=np.uint16), (64, 1))


def test_read_dicom_metadata_and_range(tmp_path, gradient):
    f = tmp_path / "a.dcm"
    make_dicom(f, gradient)
    img, meta, _ = read_dicom(str(f))
    assert img.dtype == np.uint8 and img.shape == (64, 64)
    assert img[:, 0].mean() < 10 and img[:, -1].mean() > 245
    assert (meta.sex, meta.age, meta.view) == ("F", 58, "AP")


def test_monochrome1_is_inverted(tmp_path, gradient):
    f = tmp_path / "b.dcm"
    make_dicom(f, gradient, photometric="MONOCHROME1")
    img, _, _ = read_dicom(str(f))
    assert img[:, 0].mean() > 245 and img[:, -1].mean() < 10


@pytest.mark.parametrize("raw,expected", [("058Y", 58), ("58", 58), ("006M", 0), ("", None), ("abc", None)])
def test_parse_age(raw, expected):
    assert _parse_age(raw) == expected


@pytest.mark.parametrize("arch", ["densenet121", "efficientnet_b0"])
def test_gradcam_shape_and_range(arch):
    model = CXRClassifier(arch, pretrained=False)
    x = torch.stack([preprocess(np.random.randint(0, 255, (100, 100), np.uint8), 64) for _ in range(2)])
    logits, cams = gradcam(model, x)
    assert logits.shape == (2,) and cams.shape == (2, 64, 64)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6


def test_localization_metrics():
    gt = boxes_to_mask([[50, 50, 50, 50]], orig_hw=(200, 200), out_hw=(100, 100))  # box covers [25:50, 25:50]
    assert gt.sum() == 25 * 25
    cam = np.zeros((100, 100), np.float32)
    cam[30:45, 30:45] = 1.0
    assert pointing_hit(cam, gt)
    assert cam_iou(cam, gt) == pytest.approx(15 * 15 / (25 * 25))
    assert cam_to_rgba(cam, (40, 40)).shape == (40, 40, 4)


def test_stats():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    perfect = np.where(y == 1, 0.9, 0.1)
    auc, lo, hi = bootstrap_auc(y, perfect, iters=100)
    assert auc == 1.0 and lo == 1.0
    p = rng.uniform(size=5000)
    calibrated_y = (rng.uniform(size=5000) < p).astype(int)
    assert ece(calibrated_y, p) < 0.03
    # Over-confident logits (true scale x3) should be fixed by T ~ 3.
    z = rng.normal(0, 2, 20000)
    yz = (rng.uniform(size=20000) < sigmoid(z)).astype(int)
    assert fit_temperature(3 * z, yz) == pytest.approx(3.0, rel=0.1)
    cs = confusion_stats(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (cs["tp"], cs["fn"], cs["fp"], cs["tn"]) == (1, 1, 1, 1)


def test_params_overlay(tmp_path, monkeypatch):
    from medimg import ROOT, load_params
    base = load_params()
    overlay = tmp_path / "small.yaml"
    overlay.write_text("segmentation:\n  cache_rate: 0.1\n")
    monkeypatch.setenv("MEDIMG_PARAMS", str(overlay))
    p = load_params(ROOT / "params.yaml")
    assert p["segmentation"]["cache_rate"] == 0.1
    assert p["segmentation"]["roi_size"] == base["segmentation"]["roi_size"]  # untouched keys survive
    assert p["classifier"] == base["classifier"]


def test_keep_largest_component_drops_far_blobs():
    from medimg.seg.figures_seg import keep_largest
    m = np.zeros((20, 20, 20), np.uint8)
    m[2:8, 2:8, 2:8] = 1          # organ
    m[15:17, 15:17, 15:17] = 1    # far-away false positive
    m[8, 8, 8] = 1                # touches the organ diagonally: 26-connected, so it stays
    out = keep_largest(m)
    assert out[15, 15, 15] == 0 and out[8, 8, 8] == 1 and out.sum() == 6 ** 3 + 1
    assert keep_largest(np.zeros_like(m)).sum() == 0


def test_shortcut_baseline_detects_view_leak():
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from medimg.eval.stats import shortcut_baseline
    rng = np.random.default_rng(0)
    n = 2000
    view = rng.choice(["AP", "PA"], n)
    label = (rng.random(n) < np.where(view == "AP", 0.5, 0.1)).astype(int)  # AP films are 5x more often positive
    df = pd.DataFrame({"view": view, "sex": rng.choice(["M", "F"], n), "age": rng.integers(20, 90, n).astype(float),
                       "label": label})
    df.loc[::50, "age"] = np.nan
    train, test = df.iloc[:1500], df.iloc[1500:]
    view_auc = roc_auc_score(test.label, shortcut_baseline(train, test, ["view"]))
    assert view_auc > 0.65                                   # the leak is found from metadata alone
    p_all = shortcut_baseline(train, test, ["view", "sex", "age"])
    assert p_all.shape == (len(test),) and np.isfinite(p_all).all()


def test_checkpoint_roundtrip_with_numpy_metric(tmp_path):
    # sklearn returns np.float64; it must survive torch.load(weights_only=True)
    model = CXRClassifier("efficientnet_b0", pretrained=False)
    save_checkpoint(model, 64, tmp_path / "c.pt", val_auroc=np.float64(0.89), epoch=np.int64(7))
    _, ckpt = load_checkpoint(tmp_path / "c.pt")
    assert ckpt["val_auroc"] == pytest.approx(0.89) and type(ckpt["epoch"]) is int


def test_cv_folds_partition_and_bn_recompute():
    from medimg.seg.cv_seg import bootstrap_ci, folds, recompute_bn

    parts = folds(41, 5, seed=0)
    assert sorted(i for p in parts for i in p.tolist()) == list(range(41)) and len(parts) == 5

    # Averaged weights get BatchNorm statistics recomputed from (dict) training batches.
    net = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 1), torch.nn.BatchNorm3d(2))
    loader = [{"image": torch.full((2, 1, 4, 4, 4), 3.0) + torch.randn(2, 1, 4, 4, 4) * 0.1} for _ in range(5)]
    recompute_bn(net, loader, "cpu", batches=5)
    bn = net[1]
    assert not net.training and bn.momentum is None and int(bn.num_batches_tracked) == 5

    lo, hi = bootstrap_ci(np.array([0.9, 0.92, 0.95, 0.5, 0.94]))
    assert 0.5 <= lo < hi <= 0.95


def test_seg_batches_mix_fresh_and_cached_items(tmp_path):
    """Cross-validation trains on CTs whose preprocessing is cached and CTs processed for the first time in the same
    batch; both must come out as tensors or collation fails (it did on Kaggle in fold 1)."""
    import nibabel as nib
    from monai.data import DataLoader, PersistentDataset
    from monai.data.utils import json_hashing

    from medimg.seg.train_seg import transforms

    rng = np.random.default_rng(0)
    items = []
    for i in range(4):
        img = rng.normal(50, 30, (24, 24, 24)).astype(np.float32)
        lab = np.zeros((24, 24, 24), np.uint8)
        lab[8:16, 8:16, 8:16] = 1
        for name, arr in (("img", img), ("lab", lab)):
            nib.save(nib.Nifti1Image(arr, np.eye(4)), tmp_path / f"{name}{i}.nii.gz")
        items.append({"image": str(tmp_path / f"img{i}.nii.gz"), "label": str(tmp_path / f"lab{i}.nii.gz")})
    sp = {"pixdim": [1.0, 1.0, 1.0], "roi_size": [16, 16, 16], "samples_per_volume": 2}
    cache = tmp_path / "cache"
    PersistentDataset(items[:2], transforms(sp, True), cache_dir=cache, hash_transform=json_hashing)[0]  # warm 2
    PersistentDataset(items[:2], transforms(sp, True), cache_dir=cache, hash_transform=json_hashing)[1]
    ds = PersistentDataset(items, transforms(sp, True), cache_dir=cache, hash_transform=json_hashing)
    for batch in DataLoader(ds, batch_size=4, num_workers=0):  # 2 cached + 2 fresh in one batch
        assert isinstance(batch["image"], torch.Tensor) and batch["image"].dtype == torch.float32
