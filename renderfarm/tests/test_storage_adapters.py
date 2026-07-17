"""Storage couriers: factory errors, fail-loud env checks, S3 upload flow
(with an injected fake client — no SDK needed), courier scan/rewrite."""

import pytest

from renderfarm.courier import find_local_media, prepare_prompt
from renderfarm.storage import get_storage
from renderfarm.storage.base_storage import require_env
from renderfarm.storage.s3_compatible_storage import S3CompatibleStorage


# ── factory + fail-loud env ──────────────────────────────────────────
def test_factory_unset_provider_fails_loudly(monkeypatch):
    monkeypatch.delenv("RIB_STORAGE_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="RIB_STORAGE_PROVIDER is not set"):
        get_storage()


def test_factory_unknown_provider(monkeypatch):
    monkeypatch.setenv("RIB_STORAGE_PROVIDER", "carrier-pigeon")
    with pytest.raises(RuntimeError, match="unknown RIB_STORAGE_PROVIDER"):
        get_storage()


def test_require_env_names_missing_vars(monkeypatch):
    monkeypatch.delenv("RIB_TEST_MISSING_VAR", raising=False)
    with pytest.raises(RuntimeError, match="RIB_TEST_MISSING_VAR"):
        require_env("RIB_TEST_MISSING_VAR")


def test_s3_missing_bucket_fails_loudly(monkeypatch):
    monkeypatch.delenv("RIB_S3_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="RIB_S3_BUCKET"):
        S3CompatibleStorage(client=object())


# ── s3 adapter with fake client ──────────────────────────────────────
class FakeS3Client:
    def __init__(self):
        self.uploaded = []

    def upload_file(self, path, bucket, key):
        self.uploaded.append((path, bucket, key))

    def generate_presigned_url(self, op, Params=None, ExpiresIn=None):
        return f"https://fake.s3/{Params['Bucket']}/{Params['Key']}?sig=x&ttl={ExpiresIn}"


def test_s3_upload_returns_presigned_url(tmp_path, monkeypatch):
    monkeypatch.setenv("RIB_S3_BUCKET", "farm-bucket")
    monkeypatch.setenv("RIB_S3_URL_TTL", "3600")
    f = tmp_path / "plate.exr"
    f.write_bytes(b"exr")
    fake = FakeS3Client()
    url = S3CompatibleStorage(client=fake).upload(str(f))
    assert url.startswith("https://fake.s3/farm-bucket/rib/")
    assert url.endswith("ttl=3600") and "plate.exr" in url
    assert fake.uploaded[0][1] == "farm-bucket"


# ── courier scan + rewrite ───────────────────────────────────────────
class FakeStorage:
    def __init__(self):
        self.uploads = []

    def upload(self, path):
        self.uploads.append(path)
        return f"https://cdn.example/{len(self.uploads)}"


def _prompt(tmp_path):
    img = tmp_path / "face.png"
    img.write_bytes(b"png")
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": str(img)}},
        "2": {"class_type": "KSampler", "inputs": {"steps": 20, "positive": ["1", 0]}},
        "3": {"class_type": "LoadImage", "inputs": {"image": str(img)}},  # duplicate ref
    }, str(img)


def test_find_local_media_only_real_files(tmp_path):
    prompt, img = _prompt(tmp_path)
    refs = find_local_media(prompt)
    assert {r["node_id"] for r in refs} == {"1", "3"}
    assert all(r["path"] == img for r in refs)
    # non-file strings and non-media never match
    assert find_local_media({"9": {"class_type": "X", "inputs": {"text": "hello.png"}}}) == []


def test_prepare_prompt_uploads_once_and_rewrites(tmp_path):
    prompt, img = _prompt(tmp_path)
    st = FakeStorage()
    new_prompt, uploads = prepare_prompt(prompt, backend_cfg={}, storage=st)
    assert st.uploads == [img]  # deduped: one upload for two references
    assert new_prompt["1"]["inputs"]["image"] == "https://cdn.example/1"
    assert new_prompt["3"]["inputs"]["image"] == "https://cdn.example/1"
    assert prompt["1"]["inputs"]["image"] == img  # original untouched
    assert uploads == {img: "https://cdn.example/1"}


def test_prepare_prompt_url_loader_map(tmp_path):
    prompt, img = _prompt(tmp_path)
    cfg = {"url_loader_map": {"LoadImage": {"class_type": "LoadImageFromUrl", "input": "url"}}}
    new_prompt, _ = prepare_prompt(prompt, backend_cfg=cfg, storage=FakeStorage())
    assert new_prompt["1"]["class_type"] == "LoadImageFromUrl"
    assert new_prompt["1"]["inputs"] == {"url": "https://cdn.example/1"}
    assert new_prompt["2"]["class_type"] == "KSampler"  # untouched


def test_prepare_prompt_no_media_never_needs_storage():
    prompt = {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512}}}
    new_prompt, uploads = prepare_prompt(prompt, backend_cfg={}, storage=None)
    assert new_prompt is prompt and uploads == {}
