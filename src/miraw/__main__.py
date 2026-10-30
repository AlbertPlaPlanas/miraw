"""`python -m miraw ...` runs the command line (same as the `miraw` command)."""

from miraw.cli import app

if __name__ == "__main__":  # spawned worker processes (macOS, Windows) re-import this module: do not rerun the CLI
    app(prog_name="miraw")
