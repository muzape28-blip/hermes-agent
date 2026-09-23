import pytest


def _force_android_arm32(monkeypatch, lazy_deps):
    monkeypatch.setattr(lazy_deps.sys, "platform", "android")
    monkeypatch.setattr(lazy_deps.platform, "machine", lambda: "armv8l")
    monkeypatch.setattr(lazy_deps.sysconfig, "get_platform", lambda: "android-24-armeabi_v7a")
    original_get_config_var = lazy_deps.sysconfig.get_config_var
    monkeypatch.setattr(
        lazy_deps.sysconfig,
        "get_config_var",
        lambda name: "arm-linux-androideabi" if name == "MULTIARCH" else original_get_config_var(name),
    )


def test_android_arm32_detector_matches_termux_armv8l(monkeypatch):
    from tools import lazy_deps

    _force_android_arm32(monkeypatch, lazy_deps)

    assert lazy_deps._is_android_arm32() is True


def test_android_arm32_detector_handles_legacy_termux_linux_platform(monkeypatch):
    from tools import lazy_deps

    monkeypatch.setattr(lazy_deps.sys, "platform", "linux")
    monkeypatch.setattr(lazy_deps.platform, "machine", lambda: "armv8l")
    monkeypatch.setattr(lazy_deps.platform, "release", lambda: "6.1.0-android14")
    monkeypatch.setattr(lazy_deps.sysconfig, "get_platform", lambda: "linux-armv7l")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")

    assert lazy_deps._is_android_arm32() is True


@pytest.mark.parametrize(
    "feature",
    [
        "tool.doc_extract",
        "tool.vision",
        "tool.dashboard",
        "platform.telegram",
        "stt.faster_whisper",
        "wake.openwakeword",
        "wake.sherpa",
        "wake.porcupine",
    ],
)
def test_android_arm32_lazy_native_blockers_fail_before_pip(monkeypatch, feature):
    from tools import lazy_deps

    _force_android_arm32(monkeypatch, lazy_deps)
    monkeypatch.setattr(lazy_deps, "feature_missing", lambda name: lazy_deps.LAZY_DEPS[name])

    def _no_pip(_specs):  # pragma: no cover - called only on regression
        raise AssertionError("lazy pip install should not run on Termux Android ARM32")

    monkeypatch.setattr(lazy_deps, "_venv_pip_install", _no_pip)

    with pytest.raises(lazy_deps.FeatureUnavailable) as excinfo:
        lazy_deps.ensure(feature, prompt=False)

    message = str(excinfo.value)
    assert "unsupported on Termux/Android ARM32" in message
    assert excinfo.value.missing == lazy_deps.LAZY_DEPS[feature]
