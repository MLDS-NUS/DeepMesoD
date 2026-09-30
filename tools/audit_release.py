import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ".agents",
    "configs",
    "docs",
    "examples",
    "experiments/paper",
    "src",
    "tests",
    "tools",
)
ROOT_FILES = (".gitignore", "AGENTS.md", "README.md", "pyproject.toml")
REQUIRED = (
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "src/deepmesod/cli.py",
    "src/deepmesod/data/schema.py",
    "src/deepmesod/evaluation/report.py",
    "src/deepmesod/validation.py",
    "configs/smoke.yaml",
    "experiments/paper/figure2-ising.yaml",
    ".agents/skills/use-deepmesod/SKILL.md",
    ".agents/skills/use-deepmesod/references/report.md",
)
PRIVATE_ROOT_RULES = {
    "/data/",
    "/outputs/",
    "/results/",
    "/checkpoints/",
    "/artifacts/",
    "/reports/",
    "/dev/agent_workspace/",
}
UNSAFE_UNANCHORED_RULES = {"data/", "outputs/", "results/", "checkpoints/"}
FORBIDDEN_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".npy", ".npz", ".pyc"}
PRIVATE_PREFIXES = tuple(
    "/" + value for value in ("home/", "Users/", "mnt/")
)
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]+"),
)
CJK_PATTERN = re.compile(
    "[\\u3400-\\u4dbf\\u4e00-\\u9fff\\u3040-\\u30ff\\uac00-\\ud7af]"
)


def source_files():
    for name in ROOT_FILES:
        path = ROOT / name
        if path.exists():
            yield path
    for name in SCAN_ROOTS:
        start = ROOT / name
        if not start.exists():
            continue
        for directory, subdirectories, files in os.walk(start):
            subdirectories[:] = sorted(
                item for item in subdirectories if item not in FORBIDDEN_DIRECTORIES
            )
            for filename in sorted(files):
                if filename == ".DS_Store":
                    continue
                yield Path(directory) / filename


def audit():
    violations = []
    checked = 0
    for required in REQUIRED:
        if not (ROOT / required).is_file():
            violations.append({"path": required, "reason": "required file is missing"})
    ignore_rules = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for rule in sorted(PRIVATE_ROOT_RULES - ignore_rules):
        violations.append({"path": ".gitignore", "reason": f"missing rule: {rule}"})
    for rule in sorted(UNSAFE_UNANCHORED_RULES & ignore_rules):
        violations.append(
            {
                "path": ".gitignore",
                "reason": f"unanchored rule can hide source packages: {rule}",
            }
        )
    for path in source_files():
        checked += 1
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append({"path": relative, "reason": "generated binary artifact"})
            continue
        if path.stat().st_size > 2_000_000:
            violations.append({"path": relative, "reason": "source file exceeds 2 MB"})
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append({"path": relative, "reason": "unexpected binary file"})
            continue
        for prefix in PRIVATE_PREFIXES:
            if prefix in content:
                violations.append({"path": relative, "reason": f"private path: {prefix}"})
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                violations.append({"path": relative, "reason": "possible credential"})
        if CJK_PATTERN.search(content):
            violations.append({"path": relative, "reason": "non-English CJK text"})
    return {
        "root": str(ROOT),
        "checked_files": checked,
        "violations": violations,
        "status": "pass" if not violations else "fail",
    }


def main():
    report = audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
