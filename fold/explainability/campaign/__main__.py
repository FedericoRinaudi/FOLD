"""Run the cross-run campaign from the repository root."""
import subprocess

from .runner import main


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error
