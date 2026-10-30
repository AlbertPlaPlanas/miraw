"""Tests for scripts/check_secrets.py.

Fake credentials are assembled at runtime so this file never contains a literal that the
secret scanners (including this one and gitleaks) would flag.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_secrets.py"
_spec = importlib.util.spec_from_file_location("check_secrets", SCRIPT)
assert _spec and _spec.loader
cs = importlib.util.module_from_spec(_spec)
sys.modules["check_secrets"] = cs  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(cs)

FAKE_AWS_KEY_ID = "AKIA" + "Q" * 16
FAKE_AWS_SECRET = "Zx9" + "a1B2c3D4e5" * 3 + "F6g7H8i"  # 40 chars
FAKE_GITHUB_TOKEN = "ghp_" + "A1b2C3d4E5" * 4
PRIVATE_KEY_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"


def rules(findings):
    return [f.rule for f in findings]


@pytest.mark.parametrize(
    "path",
    [".secrets.md", "sub/dir/.secrets.md", ".env", ".env.local", "deploy.pem", "id_rsa", "credentials"],
)
def test_forbidden_files_are_detected(path):
    assert cs.is_forbidden_file(path)


@pytest.mark.parametrize("path", [".env.example", "README.md", "src/miraw/env.py", "docs/secrets.md"])
def test_regular_files_are_allowed(path):
    assert not cs.is_forbidden_file(path)


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        (f"key = {FAKE_AWS_KEY_ID}", "AWS access key id"),
        (f"AWS_SECRET_ACCESS_KEY={FAKE_AWS_SECRET}", "AWS secret access key"),
        (f"aws_secret_key: '{FAKE_AWS_SECRET}'", "AWS secret access key"),
        (PRIVATE_KEY_HEADER, "private key block"),
        (f"token {FAKE_GITHUB_TOKEN}", "GitHub token"),
        ('password = "hunter2hunter2hunter2"', "quoted credential assignment"),  # pragma: allowlist secret
    ],
)
def test_patterns_flag_secrets(line, rule):
    assert rules(cs.scan_text("f.py", line, [])) == [rule]


@pytest.mark.parametrize(
    "line",
    [
        "s3://digitalrnd-drf-projects-raw-ireland/spatial_omics/dev/pyMIRAW/",
        "sequence = 'ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU'",
        "password = os.environ['DB_PASSWORD']",
        "AKIA is the prefix of AWS access key ids",
    ],
)
def test_benign_lines_pass(line):
    assert cs.scan_text("f.py", line, []) == []


def test_allowlist_marker_silences_a_line():
    line = f"example = '{FAKE_AWS_KEY_ID}'  # pragma: allowlist secret"
    assert cs.scan_text("f.py", line, []) == []


def test_known_secret_values_are_found_in_any_format(tmp_path):
    value = "Qw" + "e7Rt9Yu1Io3Pa5" * 2
    (tmp_path / ".secrets.md").write_text(f"Some notes\n\nMY_TOKEN = {value}\nshort = abc\n")
    known = cs.load_known_secrets(tmp_path)
    assert known == [value]
    findings = cs.scan_text("notes.txt", f"see {value[::-1][::-1]} here", known)
    assert rules(findings) == ["value from local .secrets.md/.env"]


def test_known_secret_detected_in_binary_file():
    value = "Bin" + "aryS3cretValue" * 2
    findings = cs.scan_bytes("blob.bin", b"\0\0header" + value.encode() + b"\0", [value])
    assert len(findings) == 1


def test_findings_never_print_the_secret(capsys, monkeypatch):
    monkeypatch.setattr(cs, "staged_content", lambda path: f"k={FAKE_AWS_KEY_ID}".encode())
    monkeypatch.setattr(cs, "load_known_secrets", lambda root: [])
    assert cs.main(["config.py"]) == 1
    err = capsys.readouterr().err
    assert "config.py:1" in err
    assert FAKE_AWS_KEY_ID not in err


# --- functional: real git repository, staged content -------------------------------------------


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    return tmp_path


def _run_guard(repo, *paths):
    return subprocess.run([sys.executable, str(SCRIPT), *paths], cwd=repo, capture_output=True, text=True, check=False)


def test_guard_blocks_staged_value_from_local_secrets_file(repo):
    value = "Lk" + "9Jh7Gf5Ds3Az" * 2
    (repo / ".secrets.md").write_text(f"AWS_SECRET_KEY = {value}\n")
    (repo / "settings.yaml").write_text(f"s3:\n  key: {value}\n")
    _git(repo, "add", "settings.yaml")
    result = _run_guard(repo, "settings.yaml")
    assert result.returncode == 1
    assert "settings.yaml:2" in result.stderr
    assert value not in result.stderr


def test_guard_checks_staged_content_not_working_tree(repo):
    (repo / "app.py").write_text(f"KEY = '{FAKE_AWS_KEY_ID}'\n")
    _git(repo, "add", "app.py")
    (repo / "app.py").write_text("KEY = os.environ['KEY']\n")  # cleaned, but not re-staged
    assert _run_guard(repo, "app.py").returncode == 1


def test_guard_blocks_force_added_secrets_file(repo):
    (repo / ".secrets.md").write_text("nothing sensitive yet\n")
    _git(repo, "add", "-f", ".secrets.md")
    result = _run_guard(repo, ".secrets.md")
    assert result.returncode == 1
    assert "credential file must never be committed" in result.stderr


def test_guard_passes_clean_files(repo):
    (repo / "README.md").write_text("# miRAW\n")
    _git(repo, "add", "README.md")
    assert _run_guard(repo, "README.md").returncode == 0
