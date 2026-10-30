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
from medimg.models.classifier import CXRClassifier


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
