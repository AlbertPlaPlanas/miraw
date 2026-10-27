#!/usr/bin/env python3
"""Block commits that would publish secrets.

Run by pre-commit on the staged files (see .pre-commit-config.yaml), and by CI on all tracked
files. Standard library only, so it works before the project environment exists.

Three checks, applied to the *staged* content of each file (what would actually be committed):

1. Forbidden files: credential files that must never be tracked (.secrets.md, .env, keys, ...).
2. Known local secrets: every value defined in the gitignored .secrets.md / .env files of this
   checkout is searched for literally, so the real credentials are caught in any format.
3. Patterns: AWS keys, private keys, GitHub/Slack tokens, quoted password/token assignments.

A false positive on a line can be silenced with an inline ``pragma: allowlist secret`` comment.
Findings never print the secret itself.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ALLOWLIST_MARKER = "pragma: allowlist secret"

# Files that hold credentials locally and must never be committed (matched on the basename).
FORBIDDEN_FILE_GLOBS = (
    ".secrets.md",
    ".secrets*",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    ".netrc",
    ".pgpass",
    "credentials",
)
# Template files that are safe to track despite matching the globs above.
ALLOWED_FILE_NAMES = (".env.example", ".env.template", ".env.sample")

# Local files whose values are treated as known secrets.
SECRET_SOURCE_FILES = (".secrets.md", ".env")
MIN_SECRET_VALUE_LEN = 12

PATTERNS: dict[str, re.Pattern[str]] = {
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
    "AWS secret access key": re.compile(
        r"(?i)aws.{0,30}(?:secret|sk).{0,30}[=:]\s*['\"]?[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+])"
    ),
    "private key block": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY(?: BLOCK)?-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "Slack token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    "quoted credential assignment": re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b"
        r"\s*[=:]\s*['\"][^'\"\s]{12,}['\"]"
    ),
}

_KEY_VALUE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(.+?)\s*$")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"  {where}: {self.rule}"


def is_forbidden_file(path: str) -> bool:
    name = Path(path).name
    if name in ALLOWED_FILE_NAMES:
        return False
    return any(fnmatch.fnmatch(name, glob) for glob in FORBIDDEN_FILE_GLOBS)


def load_known_secrets(repo_root: Path) -> list[str]:
    """Values from the local, gitignored credential files (``KEY = VALUE`` lines)."""
    values: list[str] = []
    for name in SECRET_SOURCE_FILES:
        source = repo_root / name
        if not source.is_file():
            continue
        for line in source.read_text(errors="replace").splitlines():
            match = _KEY_VALUE.match(line)
            if match:
                value = match.group(2).strip().strip("'\"")
                if len(value) >= MIN_SECRET_VALUE_LEN and " " not in value:
                    values.append(value)
    return values


def scan_text(path: str, text: str, known_secrets: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOWLIST_MARKER in line:
            continue
        if any(secret in line for secret in known_secrets):
            findings.append(Finding(path, lineno, "value from local .secrets.md/.env"))
            continue
        for rule, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append(Finding(path, lineno, rule))
                break
    return findings


def scan_bytes(path: str, data: bytes, known_secrets: list[str]) -> list[Finding]:
    if b"\0" in data[:8192]:  # binary: only the literal known-secret check is meaningful
        hit = any(secret.encode() in data for secret in known_secrets)
        return [Finding(path, 0, "value from local .secrets.md/.env (binary file)")] if hit else []
    return scan_text(path, data.decode("utf-8", errors="replace"), known_secrets)


def staged_content(path: str) -> bytes | None:
    """Content of ``path`` in the git index, falling back to the working tree."""
    result = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=False)
    if result.returncode == 0:
        return result.stdout
    file = Path(path)
    return file.read_bytes() if file.is_file() else None


def repo_root() -> Path:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    return Path(result.stdout.strip()) if result.returncode == 0 else Path.cwd()


def check_files(paths: list[str], known_secrets: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if is_forbidden_file(path):
            findings.append(Finding(path, 0, "credential file must never be committed"))
            continue
        data = staged_content(path)
        if data is not None:
            findings.extend(scan_bytes(path, data, known_secrets))
    return findings


def main(argv: list[str]) -> int:
    findings = check_files(argv, load_known_secrets(repo_root()))
    if not findings:
        return 0
    print("Possible secrets detected. Commit blocked:", file=sys.stderr)
    for finding in findings:
        print(finding, file=sys.stderr)
    print(
        "\nUnstage the file (git restore --staged <file>) or remove the secret."
        f"\nFalse positive? Add an inline '{ALLOWLIST_MARKER}' comment on that line.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
