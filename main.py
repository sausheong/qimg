import os
import uuid
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, abort
from PIL import Image

import qimg

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
app.config.update(
    UPLOAD_FOLDER=str(UPLOAD_DIR),
    OUTPUT_FOLDER=str(OUTPUT_DIR),
    MAX_CONTENT_LENGTH=20 * 1024 * 1024,  # 20 MB
)


# -----------------------------
# Simple in-memory job tracking
# -----------------------------
_jobs_lock = threading.Lock()
_completed_events: List[Dict[str, Any]] = []  # {name, url, kind, ts}


def _record_completion(name: str, kind: str):
    with _jobs_lock:
        _completed_events.append({
            "name": name,
            # Build static URL directly to avoid needing Flask app/request context in worker threads
            "url": f"/static/outputs/{name}",
            "kind": kind,
            "ts": time.time(),
        })


def _worker_generate(prompt: str, negative_prompt: str, aspect_ratio: str, steps: int, true_cfg_scale: float, seed: Optional[int], out_name: str):
    out_path = OUTPUT_DIR / out_name
    try:
        qimg.img_generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            steps=steps,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            out_path=str(out_path),
        )
        _record_completion(out_name, "generated")
    except Exception as e:
        # Log to console for now
        print(f"Generate job failed: {e}")
    finally:
        try:
            qimg.unload_pipelines()
        except Exception:
            pass


def _worker_edit(source_path: Path, prompt: str, true_cfg_scale: float, negative_prompt: str, steps: int, seed: Optional[int], out_name: str):
    out_path = OUTPUT_DIR / out_name
    try:
        with Image.open(source_path) as im:
            qimg.img_edit(
                image=im,
                prompt=prompt,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed,
                out_path=str(out_path),
            )
        _record_completion(out_name, "edited")
    except Exception as e:
        print(f"Edit job failed: {e}")
    finally:
        try:
            qimg.unload_pipelines()
        except Exception:
            pass


@app.route("/")
def index():
    # Make gallery the main page
    return redirect(url_for("gallery"))


@app.get("/generate")
def generate_page():
    return render_template("generate.html")


@app.post("/generate")
def generate_action():
    try:
        prompt = request.form.get("prompt", "").strip()
        negative_prompt = request.form.get("negative_prompt", " ")
        aspect_ratio = request.form.get("aspect_ratio", "1:1")
        steps = int(request.form.get("steps", 50))
        true_cfg_scale = float(request.form.get("true_cfg_scale", 4.0))
        seed_val = request.form.get("seed", "").strip()
        seed: Optional[int] = int(seed_val) if seed_val else None

        out_name = f"gen_{uuid.uuid4().hex[:8]}.png"

        t = threading.Thread(
            target=_worker_generate,
            args=(prompt, negative_prompt, aspect_ratio, steps, true_cfg_scale, seed, out_name),
            daemon=True,
        )
        t.start()
        flash("Generation started. The image will appear in the gallery when ready.")
        return redirect(url_for("gallery"))
    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for("generate_page"))


@app.get("/edit")
def edit_page():
    src = request.args.get("src", "").strip()
    # Only allow src within static/outputs/
    if src and not src.startswith("outputs/"):
        src = ""
    preview_url = url_for("static", filename=src) if src else None
    return render_template("edit.html", src=src, preview_url=preview_url)


@app.post("/edit")
def edit_action():
    try:
        prompt = request.form.get("prompt_edit", "").strip()
        negative_prompt = request.form.get("negative_prompt_edit", "")
        steps = int(request.form.get("steps_edit", 100))
        true_cfg_scale = float(request.form.get("true_cfg_scale_edit", 4.0))
        seed_val = request.form.get("seed_edit", "").strip()
        seed: Optional[int] = int(seed_val) if seed_val else None

        file = request.files.get("image")
        src = request.form.get("src", "").strip()
        # Resolve source image
        source_path: Optional[Path] = None
        if file and file.filename:
            # Save upload
            ext = os.path.splitext(file.filename)[1].lower() or ".png"
            up_name = f"up_{uuid.uuid4().hex[:8]}{ext}"
            up_path = UPLOAD_DIR / up_name
            file.save(up_path)
            source_path = up_path
        elif src and src.startswith("outputs/"):
            candidate = OUTPUT_DIR / Path(src).name
            if candidate.exists():
                source_path = candidate
        if source_path is None:
            flash("Please upload an image or choose one from the gallery.")
            return redirect(url_for("edit_page"))

        out_name = f"edit_{uuid.uuid4().hex[:8]}.png"
        t = threading.Thread(
            target=_worker_edit,
            args=(source_path, prompt, true_cfg_scale, negative_prompt, steps, seed, out_name),
            daemon=True,
        )
        t.start()
        flash("Edit started. The updated image will appear in the gallery when ready.")
        return redirect(url_for("gallery"))
    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for("edit_page"))


@app.get("/gallery")
def gallery():
    # List images from OUTPUT_DIR
    items = []
    try:
        for p in sorted(OUTPUT_DIR.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True):
            name = p.name
            kind = "generated" if name.startswith("gen_") else ("edited" if name.startswith("edit_") else "unknown")
            items.append({
                "name": name,
                "url": url_for("static", filename=f"outputs/{name}"),
                "kind": kind,
                "mtime": p.stat().st_mtime,
            })
    except Exception as e:
        flash(f"Failed to load gallery: {e}")
    return render_template("gallery.html", items=items)


@app.get("/jobs/updates")
def jobs_updates():
    """Return recently completed items since a timestamp.

    Query param: since (float epoch seconds). Returns JSON: { since: float, items: [...] }
    """
    try:
        since = float(request.args.get("since", "0") or 0)
    except ValueError:
        since = 0.0
    with _jobs_lock:
        items = [e for e in _completed_events if e["ts"] > since]
        now = time.time()
    return jsonify({"since": now, "items": items})


@app.get("/download/<path:filename>")
def download_output(filename: str):
    # Only allow files within OUTPUT_DIR
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        abort(404)
    return send_from_directory(app.config["OUTPUT_FOLDER"], safe_name, as_attachment=True)


@app.post("/gallery/delete")
def gallery_delete():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"ok": False, "error": "missing name"}), 400
    safe_name = Path(name).name
    target = OUTPUT_DIR / safe_name
    # Basic safety: restrict to typical image extensions in outputs
    if not target.exists() or target.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        return jsonify({"ok": False, "error": "not found"}), 404
    try:
        target.unlink()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/result/<path:filename>")
def result(filename: str):
    # Render result page with image
    return render_template("result.html", image_url=url_for("static", filename=f"outputs/{filename}"))


@app.get("/uploads/<path:filename>")
def get_upload(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
