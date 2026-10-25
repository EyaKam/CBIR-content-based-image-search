"""VGG16 pipeline checks that do not need the ImageNet weights (an untrained network is used)."""
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def fe(monkeypatch_module=None):
    import os
    os.environ["CBIR_VGG_WEIGHTS"] = "none"
    from cbir.feature_extractor import FeatureExtractor
    return FeatureExtractor()


def images():
    rng = np.random.default_rng(0)
    return [Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8"))
            for h, w in ((100, 80), (300, 200), (50, 50))]


def test_feature_is_4096_dims_unit_norm_and_nonnegative(fe):
    f = fe.extract(images()[0])
    assert f.shape == (4096,) and np.linalg.norm(f) == pytest.approx(1.0, abs=1e-5)
    assert (f >= 0).all()                       # fc1 is followed by a ReLU


def test_batch_equals_single_and_array_equals_pil(fe):
    ims = images()
    batch = fe.extract_batch(ims, batch_size=2)
    assert batch.shape == (3, 4096)
    for im, row in zip(ims, batch):
        assert np.allclose(fe.extract(im), row, atol=1e-4)
        assert np.allclose(fe.extract(np.asarray(im)), row, atol=1e-4)


def test_grayscale_and_rgba_inputs_are_accepted(fe):
    g = Image.new("L", (64, 64), 128)
    a = Image.new("RGBA", (64, 64), (10, 20, 30, 40))
    assert fe.extract(g).shape == (4096,) and fe.extract(a).shape == (4096,)
