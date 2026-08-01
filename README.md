# AI Video Clipper — Indonesian Edition

An intelligent video-to-clips converter that uses AI transcription and LLM analysis to automatically extract engaging short clips from longer videos. Optimized for Indonesian-language content.

## Web Dashboard

A browser dashboard where you upload a video, watch the pipeline run step by step, and download the resulting clips — no CLI needed.

```bash
python web_app.py                # serves on http://localhost:8000
python web_app.py --port 8080    # custom port
python web_app.py --host 0.0.0.0 # expose on your LAN
```

Open the printed URL. The dashboard shows the 9 pipeline stages as a live stepper
(upload → transcribe → prefilter → audio-energy → LLM-select → refine → extract →
render → done), a streaming log, and the finished clips with inline preview + download.
Output clips are written to `clips/<VideoName>/`.

**No extra dependencies** — the dashboard is pure Python stdlib (`http.server`). The long
pipeline steps run in a background thread; the browser polls `/api/job/<id>` every ~1s.

> The dashboard drives the exact same pipeline as `main.py` (`sosmed/web_runner.py`
> wraps `transcribe → prefilter → audio energy → find_clips → refine → extract →
> postprocess`). LLM settings come from `config.yaml` / `.env` exactly like the CLI.

## Features

- 🎬 **Automatic Transcription** — Uses `faster-whisper` for accurate, fast speech-to-text (GPU/CPU auto-detection)
- 🧠 **AI-Powered Clip Extraction** — LLM automatically decides which segments are most engaging (max 100 clips)
- 🇮🇩 **Indonesian Optimized** — Pre-filters noise, filler words, and duplicates tuned for Bahasa Indonesia
- 🎵 **Music/Lyrics Filtering** — Automatically filters out pure instrumental sections from clip analysis
- ⚡ **Sequential FFmpeg Processing** — Clips are extracted and post‑processed one at a time to avoid running multiple FFmpeg instances concurrently
- 🔌 **Multi-LLM Support** — Works with OpenRouter, Anthropic, OpenAI, or Ollama
- 📊 **Smart Ranking** — Clips are ranked by engagement score with compelling hooks extracted
- 📱 **Auto Portrait Reframing** — Intelligently reframes landscape video to 9:16 portrait with face detection
- 💬 **TikTok-Style Subtitles** — Word-by-word karaoke-highlighted subtitles burned into video

## Project Structure

```
sosmed/
├── main.py              # Main video processing script
├── sosmed/
│   ├── cli.py           # CLI argument parsing & orchestration
│   ├── transcription.py # faster-whisper transcription
│   ├── prefilter.py     # Noise/filler/duplicate removal
│   ├── extraction.py    # FFmpeg clip extraction
│   ├── postprocess.py   # Post-processing orchestrator
│   ├── reframe.py       # Smart 9:16 portrait reframing
│   ├── subtitles.py     # TikTok-style ASS subtitle generation
│   ├── postprocess.py   # Apply subtitles and other visual/audio cleanup
│   ├── utils.py         # Logging, constants, prompts
│   └── llm/             # LLM analysis backends
├── clips/
│   └── clips.json       # Generated clip metadata and timestamps (now includes clip filenames)
└── videos/              # Input video directory
```

## Setup

### Requirements

- Python 3.10+
- FFmpeg
- CUDA/cuDNN (optional, for GPU acceleration)
- OpenCV is installed automatically for face detection (used in portrait reframing)

### Installation

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

# Clone or navigate to the project
cd sosmed

# Create and activate a conda environment
conda create -n ai_clipper python=3.11 -y
conda activate ai_clipper

# Install CUDA-enabled PyTorch (recommended for GPU)
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.4

# Install dependencies
pip install -r requirements.txt
```

#### CPU-only alternative

If you do not have an NVIDIA GPU, install the CPU build instead:

```bash
conda install -c pytorch pytorch torchvision torchaudio
```

### Configuration

**LLM Priority:** The script automatically uses available LLM providers in this order:
1. **OpenRouter** (free default) — Set `OPENROUTER_API_KEY` environment variable
2. **Ollama** — Run locally (no API key needed)

Example:
```bash
export OPENROUTER_API_KEY=sk-...
python main.py video.mp4
```

## Usage

### Basic Usage

```bash
python main.py videos/your-video.mp4
```

### Advanced Options

```bash
# Specify whisper model size
python main.py video.mp4 --model large-v3

# Set custom clip duration range (in seconds)
python main.py video.mp4 --min 15 --max 90

# Limit number of clips to extract
python main.py video.mp4 --max-clips 50

# Specify language for transcription
python main.py video.mp4 --lang id

# Disable portrait reframing (keep landscape)
python main.py video.mp4 --no-reframe

# Disable subtitles
python main.py video.mp4 --no-subtitles

# Change subtitle position
python main.py video.mp4 --subtitle-position upper

# Raw clips only — no post-processing
python main.py video.mp4 --no-reframe --no-subtitles

# Default run: portrait + subtitles (no audio effects)
python main.py video.mp4
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | medium | Whisper model size (tiny, small, base, medium, large) |
| `--min` | 15 | Minimum clip duration (seconds) |
| `--max` | 60 | Maximum clip duration (seconds) |
| `--max-clips` | 200 | Maximum number of clips to extract |
| `--lang` | id | Language code (e.g., `id` for Indonesian) |
| `--reframe` / `--no-reframe` | on | Portrait (9:16) reframe with face detection |
| `--subtitles` / `--no-subtitles` | on | TikTok-style word-by-word subtitles |
| `--target-language` | en | Language to translate clip titles, captions, and subtitles into. `en`=English, `id`=Indonesian, `auto`=keep the original/transcribed language (only fixes transcription errors), or any language label/code (e.g. `Spanish`, `Japanese`) for an open/custom target |
| `--subtitle-position` | center | Subtitle position: `center`, `upper`, `lower` |

## Output

The script generates a `clips/clips.json` file with extracted clips:

```json
[
  {
    "rank": 1,
    "start": 2078.1,
    "end": 2138.5,
    "title": "Perkembangan AI dari Tahun ke Tahun",
    "reason": "Visualisasi menarik tentang kemajuan AI yang dramatis dalam waktu singkat",
    "hook": "2023, bentukannya masih seperti ini. Sepagetti sama mulut nggak ada yang tahu.",
    "engagement_score": 88
  }
]
```

**Clip files** are output as `rank01_Title_final.mp4` (post-processed) or `rank01_Title.mp4` (raw). The corresponding `clips.json` metadata now includes a `filename` field for each clip so you can easily match entries to files.

**Fields:**
- `rank` — Priority ranking (1 = highest engagement)
- `start` / `end` — Clip timestamps in seconds
- `title` — Generated title for the clip
- `reason` — Why the LLM found this segment engaging
- `hook` — Opening quote to grab attention
- `engagement_score` — Score 0-100 indicating potential viral appeal

## Post-Processing Features

### 🎵 Music & Lyrics Filtering

Automatically filters out pure instrumental and music-only sections to prevent them from being included in clip analysis.

- **Conservative approach**: Only filters segments where Whisper indicates very high confidence (>0.75) that it's non-speech
- **Safe for voice-over**: Speech recorded over background music is preserved — only pure instrumental sections are removed
- **Pattern detection**: Filters out:
  - Instrumental-only sections detected by Whisper's `no_speech_prob` confidence score
  - Pure onomatopoeia patterns (`la la la`, `doo doo`, etc.)
  - Explicit markers like `[instrumental]`, `[music]`, `[singing]`
- **Preserves speech with music**: If you're talking while the song plays, your speech will still be transcribed and included in clip analysis

### 📱 Auto Portrait Reframing (9:16)

Automatically converts landscape video to portrait format optimized for TikTok, YouTube Shorts, and Instagram Reels.

- **Face detection** (OpenCV) determines the optimal horizontal crop position
- Falls back to center crop when no faces are detected
- Scales output to 1080×1920 for optimal short-form quality
- Disable with `--no-reframe`

### 💬 TikTok-Style Subtitles

Word-by-word highlighted subtitles burned directly into the video.

- Large bold white text with black outline
- **Karaoke fill effect**: each word sweeps from white → yellow as it's spoken
- Words grouped into 2–4 word chunks for readability
- Subtle pop-in animation on each subtitle group
- Positioned center-screen (customizable with `--subtitle-position`)
- Disable with `--no-subtitles`

## Example Workflow

```bash
# Full production pipeline: portrait + subtitles
python main.py "videos/AI di Dunia Industri [AI Webinar Series - Eps 1].mp4" \
  --lang id

# Quick preview (no post-processing)
python main.py video.mp4 --no-reframe --no-subtitles --no-sfx

# Portrait clips with subtitles
python main.py video.mp4 --model large-v3

# Results appear in clips/ with:
# - Portrait (9:16) reframed clips ready for TikTok/Reels/Shorts
# - Word-by-word highlighted subtitles burned in
# - clips.json with metadata, timestamps, and hooks
```

## Performance Tips

- **GPU Acceleration**: Install CUDA for ~10x faster transcription
- **Batch Processing**: Process multiple videos with a loop
- **Model Size**: Use `--model base` for faster processing, `--model large-v3` for better accuracy
- **Automatic Caching**: Transcript and clips are cached automatically — rerunning `main.py` on the same video **skips transcription and LLM analysis**, jumping straight to extraction/post-processing
  - Transcript cache: `.cache/ai-video-clipper/{video}_transcript.json`
  - Clips cache: `clips/{video_stem}/clips.json`
- **Efficient FFmpeg Encoding**: The tool tries multiple encoding strategies (hardware-accelerated H.264, HEVC, then CPU fallbacks) to maximize compatibility and speed

## Troubleshooting

**Issue**: `ImportError: ... libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`  
→ Reinstall PyTorch via conda with CUDA support (see Installation)

**Issue**: Slow transcription  
→ Install CUDA or use a smaller model with `--model base`

**Issue**: No LLM API key error  
→ Set one of the LLM provider variables (see Configuration)

**Issue**: FFmpeg not found  
→ Install FFmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Ubuntu)

**Issue**: FFmpeg encoding failures  
→ The tool automatically tries multiple encoding strategies (H.264 hardware-accelerated, H.264/HEVC NVENC, libx264, fallback codecs). Most failures are resolved automatically. If still failing:
  - Ensure FFmpeg is installed and in PATH: `which ffmpeg`
  - Try updating FFmpeg: `apt-get install --only-upgrade ffmpeg`
  - Check available video codecs: `ffmpeg -codecs | grep hevc`
  - Run with verbose FFmpeg output in the source code (see `extraction.py` and `postprocess.py`)

## License

MIT License

Copyright (c) 2026 Samuel Koesnadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

Samuel Koesnadi

---

**Questions or improvements?** Feel free to open an issue or submit a pull request!
