from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from generate_images import generate_images


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input_images"
GENERATED_DIR = BASE_DIR / "data" / "generated_images"
SKETCH_DIR = BASE_DIR / "data" / "sketches"
LORA_DIR = BASE_DIR / "loras"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
SKETCH_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_LORA_OPTIONS = [
    "loish_style.safetensors",
    "michal_sawtyruk_style.safetensors",
    "neil_ross_style.safetensors",
]

SHOWCASE = [
    {
        "title": "Loish Style",
        "image": "loish_style_megan_fox_1.png",
        "placeholder": "Greate for femail portrait generation",
    },
    {
        "title": "Michal Sawtyruk Style",
        "image": "michal_sawtyruk_style_car_scene_1.png",
        "placeholder": "Greate for illustrations with a modern color palette",
    },
    {
        "title": "Neil Ross Style",
        "image": "neil_ross_style_city_2.png",
        "placeholder": "Great for scene illustrations that mix classic and modern styles, or have a futuristic look",
    },
]

ALLOWED_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

app = Flask(__name__, template_folder="templates", static_folder="static")


@dataclass
class GenerationForm:
    prompt: str
    negative_prompt: str
    num: int
    seed: int | None
    num_inference_steps: int
    guidance_scale: float
    num_images_per_prompt: int
    controlnet_conditioning_scale: float
    guess_mode: bool
    lora_scale: float
    lora_name: str


def _get_lora_options() -> list[str]:
    discovered = sorted([item.name for item in LORA_DIR.glob("*.safetensors")])
    if not discovered:
        return DEFAULT_LORA_OPTIONS

    all_options = discovered.copy()
    for fallback in DEFAULT_LORA_OPTIONS:
        if fallback not in all_options:
            all_options.append(fallback)
    return all_options


def _parse_form_data() -> GenerationForm:
    seed_raw = request.form.get("seed", "").strip()
    seed = int(seed_raw) if seed_raw else None

    return GenerationForm(
        prompt=request.form.get("prompt", "").strip(),
        negative_prompt=request.form.get("negative_prompt", "").strip(),
        num=int(request.form.get("num", 1)),
        seed=seed,
        num_inference_steps=int(request.form.get("num_inference_steps", 30)),
        guidance_scale=float(request.form.get("guidance_scale", 7.5)),
        num_images_per_prompt=int(request.form.get("num_images_per_prompt", 1)),
        controlnet_conditioning_scale=float(
            request.form.get("controlnet_conditioning_scale", 1.0)
        ),
        guess_mode=request.form.get("guess_mode") == "on",
        lora_scale=float(request.form.get("lora_scale", 0.8)),
        lora_name=request.form.get("lora_name", "").strip(),
    )


def _is_allowed_upload(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS


def _save_uploaded_image() -> tuple[Path | None, str | None]:
    uploaded = request.files.get("input_image")
    if not uploaded or not uploaded.filename:
        return None, "Please upload an image first."

    if not _is_allowed_upload(uploaded.filename):
        return None, "Only PNG, JPG, JPEG and WEBP files are supported."

    safe_name = secure_filename(uploaded.filename)
    if not safe_name:
        return None, "Invalid file name."

    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem
    timestamp = int(time.time())
    final_name = f"{stem}_{timestamp}{suffix}"
    save_path = INPUT_DIR / final_name
    uploaded.save(save_path)
    return save_path, None


def _collect_generated_image_names() -> set[str]:
    return {item.name for item in GENERATED_DIR.glob("*.png")}


def _list_gallery_images(limit: int = 30) -> list[str]:
    files = sorted(
        GENERATED_DIR.glob("*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [item.name for item in files[:limit]]


def _build_showcase() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in SHOWCASE:
        image_path = GENERATED_DIR / entry["image"]
        items.append(
            {
                "title": entry["title"],
                "image": entry["image"],
                "exists": image_path.exists(),
                "placeholder": entry["placeholder"],
            }
        )
    return items


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    message = ""
    error = ""
    generated_now: list[str] = []

    lora_options = _get_lora_options()
    form_defaults = {
        "prompt": "",
        "negative_prompt": "",
        "num": 1,
        "seed": "",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "num_images_per_prompt": 1,
        "controlnet_conditioning_scale": 1.0,
        "guess_mode": False,
        "lora_scale": 0.8,
        "lora_name": lora_options[0] if lora_options else "",
    }

    if request.method == "POST":
        try:
            upload_path, upload_error = _save_uploaded_image()
            if upload_error:
                raise ValueError(upload_error)
            if upload_path is None:
                raise ValueError("Image upload failed.")

            form = _parse_form_data()
            before = _collect_generated_image_names()

            generate_images(
                num=form.num,
                name=upload_path.name,
                lora_name=form.lora_name,
                prompt=form.prompt,
                seed=form.seed,
                num_inference_steps=form.num_inference_steps,
                guidance_scale=form.guidance_scale,
                negative_prompt=form.negative_prompt or None,
                num_images_per_prompt=form.num_images_per_prompt,
                controlnet_conditioning_scale=form.controlnet_conditioning_scale,
                guess_mode=form.guess_mode,
                lora_scale=form.lora_scale,
            )

            after = _collect_generated_image_names()
            generated_now = sorted(list(after - before))
            if generated_now:
                message = f"Done. Generated {len(generated_now)} image(s)."
            else:
                message = "Generation finished. No new PNG detected."

            form_defaults.update(
                {
                    "prompt": form.prompt,
                    "negative_prompt": form.negative_prompt,
                    "num": form.num,
                    "seed": "" if form.seed is None else str(form.seed),
                    "num_inference_steps": form.num_inference_steps,
                    "guidance_scale": form.guidance_scale,
                    "num_images_per_prompt": form.num_images_per_prompt,
                    "controlnet_conditioning_scale": form.controlnet_conditioning_scale,
                    "guess_mode": form.guess_mode,
                    "lora_scale": form.lora_scale,
                    "lora_name": form.lora_name,
                }
            )
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        showcase=_build_showcase(),
        lora_options=lora_options,
        message=message,
        error=error,
        generated_now=generated_now,
        gallery_images=_list_gallery_images(limit=40),
        form=form_defaults,
    )


@app.route("/generated/<path:filename>")
def serve_generated(filename: str):
    return send_from_directory(GENERATED_DIR, filename)


@app.route("/download/<path:filename>")
def download_generated(filename: str):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


@app.route("/health")
def health() -> str:
    return "ok"


def main() -> None:
    app.run(host="127.0.0.1", port=7860, debug=False)


if __name__ == "__main__":
    main()
