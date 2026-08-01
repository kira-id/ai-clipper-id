"""Re-download faster-whisper-large-v3 through the current HF CDN.

Run in background; prints progress to stdout. On success prints DONE.
"""
import os
import sys
import time

from huggingface_hub import hf_hub_download

REPO = "Systran/faster-whisper-large-v3"
FILES = ["model.bin", "config.json", "preprocessor_config.json",
         "tokenizer.json", "vocabulary.json", "tokenizer_config.json"]

# Force the real network path even if stale cache metadata lingered.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

t0 = time.time()
for f in FILES:
    try:
        p = hf_hub_download(
            repo_id=REPO,
            filename=f,
            repo_type="model",
            local_files_only=False,
        )
        size = os.path.getsize(p)
        print(f"[OK] {f} -> {size} bytes", flush=True)
    except Exception as e:  # surface, don't silently swallow
        print(f"[FAIL] {f}: {type(e).__name__}: {e}", flush=True)
        sys.exit(2)

print(f"DONE in {time.time() - t0:.1f}s", flush=True)
