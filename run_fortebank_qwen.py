"""Run a local Qwen3.5 ForteBank baseline without TorchCodec.

Examples:
  python run_fortebank_qwen_opencv.py --video X-1
  python run_fortebank_qwen_opencv.py --video X-1 --frames 32
  python run_fortebank_qwen_opencv.py --all --frames 16

Video is decoded by OpenCV. Sampled frames with explicit timestamps
are passed as images (not a video URI) to the multimodal model.
"""
import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "fortebank_qwen_blind_manifest.json"
RULES = BASE / "fortebank_procedure_rules.json"
PROMPT = BASE / "fortebank_qwen_prompt.md"
ZIP = BASE / "videos.zip"
VIDEO_DIR = BASE / "videos"
RESULTS = BASE / "qwen_results"


def norm_name(name):
    name = unicodedata.normalize("NFC", Path(name).name)
    if name.startswith(("Х-", "х-")):
        name = "X-" + name[2:]
    return name


def resolve_video(filename):
    target = norm_name(filename)
    VIDEO_DIR.mkdir(exist_ok=True)
    dest = VIDEO_DIR / target
    if dest.is_file():
        return dest
    for p in VIDEO_DIR.rglob("*"):
        if p.is_file() and norm_name(p.name).casefold() == target.casefold():
            return p
    if not ZIP.is_file():
        raise FileNotFoundError(f"Missing {target}. Put videos.zip or videos/ next to this script")
    with zipfile.ZipFile(ZIP) as archive:
        matching = [f for f in archive.infolist() if not f.is_dir()
                    and not f.filename.startswith("__MACOSX/")
                    and norm_name(f.filename).casefold() == target.casefold()]
        if not matching:
            raise FileNotFoundError(f"{target} not found in {ZIP}")
        import shutil
        with archive.open(matching[0]) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest


def sample_frames(video_path, count=16, max_side=512):
    """OpenCV frames: list of (time_seconds, PIL RGB Image). No TorchCodec."""
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or total <= 0:
            raise RuntimeError(f"Could not determine video length or fps: frames={total}, fps={fps}")
        indices = np.unique(np.linspace(0, total - 1, min(count, total)).round().astype(int))
        results = []
        for frame_no in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
            ok, bgr = cap.read()
            if not ok:
                print(f"WARNING: cannot decode frame {frame_no}")
                continue
            h, w = bgr.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                bgr = cv2.resize(bgr, (max(1, round(w*scale)), max(1, round(h*scale))),
                                 interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results.append((round(int(frame_no) / fps, 3), Image.fromarray(rgb)))
        if not results:
            raise RuntimeError("No frames could be decoded")
        return results
    finally:
        cap.release()


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first, last = text.find("{"), text.rfind("}")
        if first >= 0 and last > first:
            try:
                return json.loads(text[first:last+1])
            except json.JSONDecodeError:
                pass
    return None


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Video ID, e.g. X-1")
    group.add_argument("--all", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--frames", type=int, default=16,
                   help="Frames sampled across the entire video (try 16, 24, 32)")
    p.add_argument("--max-side", type=int, default=512, help="Longest side of an input image")
    p.add_argument("--max-tokens", type=int, default=1800)
    args = p.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    prompt = PROMPT.read_text(encoding="utf-8")
    entries = manifest["video_ids"]
    if args.video:
        name = norm_name(args.video).removesuffix(".mov")
        entries = [e for e in entries if e["id"] == name]
        if not entries:
            raise SystemExit(f"No manifest entry for {args.video}")

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    print("Loading", args.model, flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, device_map="auto", dtype="auto"
    ).eval()
    RESULTS.mkdir(exist_ok=True)

    rule_text = json.dumps({
        "sources": rules["sources"],
        "procedures": rules["procedures"],
        "critical_constraints": rules["critical_constraints"],
    }, ensure_ascii=False)
    task_text = json.dumps({k: manifest["task"][k] for k in (
        "instruction", "candidate_event_labels", "critical_constraints"
    )}, ensure_ascii=False)

    for entry in entries:
        video_id = entry["id"]
        print(f"\n=== {video_id} ===", flush=True)
        try:
            video_path = resolve_video(entry["filename"])
            frames = sample_frames(video_path, args.frames, args.max_side)
            print("Decoded", len(frames), "OpenCV frames; timestamp range:",
                  frames[0][0], "to", frames[-1][0], "seconds", flush=True)
            # Deliberately use IMAGE items, not VIDEO: no libtorchcodec required.
            # Explicit timestamps help the model reason about event order.
            content = [{"type": "text", "text": (
                f"Video ID: {video_id}; approximately {entry['duration_sec']} sec. "
                "Below are time-stamped samples of the VIDEO in chronological order. "
                "You must not claim visibility of actions between unobserved samples. "
                "All sample times are seconds from the video's start."
            )}]
            for seconds, frame in frames:
                content.append({"type": "text", "text": f"Frame at {seconds:.2f} sec:"})
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": (
                f"\nInstructions:\n{prompt}\n\n"
                f"Task / event labels:\n{task_text}\n\n"
                f"Rule catalog (hypothesis_only is NOT a law):\n{rule_text}\n\n"
                "Return ONE valid JSON object in Russian. Do not invent missing steps "
                "or claim a violation merely because a step falls between sampled frames."
            )})
            messages = [{"role": "user", "content": content}]
            # No num_frames or enable_thinking kwargs: no video decoder is called.
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False,
                                           max_new_tokens=args.max_tokens)
            completion = generated[0][inputs["input_ids"].shape[1]:]
            raw = processor.decode(completion, skip_special_tokens=True).strip()
            (RESULTS / f"{video_id}_raw.txt").write_text(raw, encoding="utf-8")
            parsed = extract_json(raw)
            if parsed is not None:
                (RESULTS / f"{video_id}.json").write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps(parsed, ensure_ascii=False, indent=2)[:2500])
            else:
                print(f"Model returned non-JSON response: {RESULTS / (video_id+'_raw.txt')}")
            del inputs, generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"ERROR in {video_id}: {type(e).__name__}: {e}")
            (RESULTS / f"{video_id}_error.txt").write_text(
                f"{type(e).__name__}: {e}", encoding="utf-8"
            )
    print("\nResults:", RESULTS)


if __name__ == "__main__":
    main()
