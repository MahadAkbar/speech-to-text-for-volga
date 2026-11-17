"""CLI: python -m app.cli input.mp3 [--language en] [--format json|srt|txt]"""
import argparse
import json
import sys
from pathlib import Path

from .audio import AudioError
from .pipeline import to_srt, transcribe_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file", type=Path)
    p.add_argument("--language")
    p.add_argument("--format", choices=["json", "srt", "txt"], default="json")
    a = p.parse_args()
    if not a.file.is_file():
        print(f"Not found: {a.file}", file=sys.stderr)
        return 2
    try:
        t = transcribe_file(a.file, a.language)
    except AudioError as e:
        print(f"Invalid audio: {e}", file=sys.stderr)
        return 1
    out = {"json": lambda: json.dumps(t.to_dict(), indent=2), "srt": lambda: to_srt(t), "txt": lambda: t.text}
    print(out[a.format]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
