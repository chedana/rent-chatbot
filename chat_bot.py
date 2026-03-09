from pathlib import Path
import runpy
import sys


def _resolve_new_root() -> Path:
    env = Path((__import__("os").environ.get("RENT_AGENT_ROOT") or "").strip())
    if env:
        return env
    # Default sibling path: /workspace/AI-assistant
    return Path(__file__).resolve().parent.parent / "AI-assistant"


def main() -> None:
    new_root = _resolve_new_root()
    main_file = new_root / "main.py"
    if not main_file.exists():
        raise FileNotFoundError(
            f"New entrypoint not found: {main_file}. "
            "Set RENT_AGENT_ROOT to the migrated project root."
        )
    if str(new_root) not in sys.path:
        sys.path.insert(0, str(new_root))
    runpy.run_path(str(main_file), run_name="__main__")


if __name__ == "__main__":
    main()
