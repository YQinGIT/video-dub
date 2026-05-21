"""Standalone IndexTTS-2 synthesis worker — runs in IndexTTS-2's own venv.

This script is deliberately *not* imported by the `videodub` package. IndexTTS-2
pins dependencies (and a Python version) that cannot share the videodub virtual
environment, so the `videodub.tts.indextts2` backend installs IndexTTS-2 in its
own isolated venv and runs this script there as a subprocess.

The contract is a tiny JSON protocol. The single argument is the path to a job
file::

    {
      "cfg_path":        "<abs path to checkpoints/config.yaml>",
      "model_dir":       "<abs path to the checkpoints directory>",
      "device":          "cuda",
      "reference_audio": "<abs path to the speaker reference clip>",
      "segments": [{"text": "...", "output_path": "<abs path>.wav"}, ...]
    }

For each segment the worker writes one WAV to `output_path`. It exits 0 on
success; on failure it prints a traceback to stderr and exits non-zero, which
the backend turns into a `BackendError`.

It imports only the standard library and `indextts`; importing `videodub` here
would defeat the isolation, so it never does.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _indextts2_worker.py <job.json>", file=sys.stderr)
        return 2

    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    try:
        from indextts.infer_v2 import IndexTTS2
    except Exception:
        traceback.print_exc()
        print("could not import IndexTTS-2 — is its venv set up?", file=sys.stderr)
        return 3

    try:
        model = IndexTTS2(
            cfg_path=job["cfg_path"],
            model_dir=job["model_dir"],
            use_fp16=False,
            device=job["device"],
        )
    except Exception:
        traceback.print_exc()
        print("could not load the IndexTTS-2 model", file=sys.stderr)
        return 4

    for segment in job["segments"]:
        try:
            # IndexTTS-2's `infer` writes the clip itself and exposes no
            # target-duration argument — the Stage 7d timing fitter stretches
            # each clip to its slot.
            model.infer(
                spk_audio_prompt=job["reference_audio"],
                text=segment["text"],
                output_path=segment["output_path"],
                verbose=False,
            )
        except Exception:
            traceback.print_exc()
            print(f"synthesis failed for text: {segment['text']!r}", file=sys.stderr)
            return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())
