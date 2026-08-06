from types import SimpleNamespace

import torch
from PIL import Image

from flux_native_in_memory import extract_flux_native_entry, strip_flux_replay_entry


class _FakeFeaturizer:
    def __init__(self):
        self.calls = []

    def forward(self, args, image, **kwargs):
        self.calls.append((args, image.clone(), kwargs))
        feature = torch.ones((1, 4, image.shape[-2] // 16, image.shape[-1] // 16))
        ada = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        return feature, ada


def test_extract_flux_native_entry_is_memory_only_and_uses_official_resize(tmp_path):
    image_dir = tmp_path / "JPEGImages" / "cat"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (80, 40), color=(20, 40, 60)).save(image_dir / "pair.jpg")
    args = SimpleNamespace(img_size=[64, 64], t=260, k=[28], ensemble_size=8)
    featurizer = _FakeFeaturizer()

    entry = extract_flux_native_entry(
        featurizer,
        args,
        dataset_path=str(tmp_path),
        category="cat",
        image_name="pair.jpg",
        caption="a cat",
    )

    assert tuple(entry["feature"].shape) == (1, 4, 2, 4)
    assert entry["feature"].device.type == "cpu"
    assert entry["ada"].device.type == "cpu"
    assert len(featurizer.calls) == 1
    _, image_tensor, kwargs = featurizer.calls[0]
    assert tuple(image_tensor.shape) == (3, 32, 64)
    assert kwargs == {
        "caption": "a cat",
        "category": "cat",
        "timestep": 260,
        "block_idx": [28],
        "ensemble_size": 8,
    }
    assert list(tmp_path.rglob("*.pth")) == []


def test_strip_flux_replay_entry_keeps_only_reusable_native_tensors():
    feature = torch.randn(1, 4, 2, 3)
    ada = [torch.randn(1, 2, 4)]
    replay = {
        "feature": feature,
        "ada": ada,
        "replay_state": {"large_hidden_state": torch.randn(8, 16)},
        "metadata": {"cache_version": 4},
    }

    native = strip_flux_replay_entry(replay)

    assert set(native) == {"feature", "ada"}
    assert native["feature"] is feature
    assert native["ada"] is ada
    assert "replay_state" not in native


def test_fjsar_entry_horizontal_flip_is_opt_in_and_in_memory(tmp_path):
    image_dir = tmp_path / "JPEGImages" / "cat"
    image_dir.mkdir(parents=True)
    image = Image.new("RGB", (32, 16), color=(0, 0, 0))
    for x in range(16):
        for y in range(16):
            image.putpixel((x, y), (255, 0, 0))
    image.save(image_dir / "pair.png")
    args = SimpleNamespace(
        img_size=[32, 32],
        t=260,
        k=[28],
        ensemble_size=8,
    )
    featurizer = _FakeFeaturizer()

    extract_flux_native_entry(
        featurizer,
        args,
        dataset_path=str(tmp_path),
        category="cat",
        image_name="pair.png",
        caption="a cat",
    )
    extract_flux_native_entry(
        featurizer,
        args,
        dataset_path=str(tmp_path),
        category="cat",
        image_name="pair.png",
        caption="a cat",
        horizontal_flip=True,
    )

    original = featurizer.calls[0][1]
    flipped = featurizer.calls[1][1]
    assert torch.equal(flipped, original.flip(-1))
    assert list(tmp_path.rglob("*.pth")) == []
