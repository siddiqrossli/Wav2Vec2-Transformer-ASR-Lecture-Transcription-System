from __future__ import annotations

import os

# Disable Gradio 5 Node SSR proxy to prevent port collisions
os.environ["GRADIO_SSR_MODE"] = "False"

# ZeroGPU hook must be imported first before any torch or transformers module
try:
    import spaces
    print("[ZeroGPU] spaces module initialized at top-level")
except ImportError:
    pass

import sys
import time
import threading
import importlib.util
from pathlib import Path

# Add project source directories to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "web"))
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from a2wsgi import WSGIMiddleware

# Load authentic Flask application module under isolated namespace to prevent circular import with root app.py
spec = importlib.util.spec_from_file_location("web_app", ROOT / "src" / "web" / "app.py")
web_app_mod = importlib.util.module_from_spec(spec)
sys.modules["web_app"] = web_app_mod
spec.loader.exec_module(web_app_mod)
flask_app = web_app_mod.app

from config import WebConfig

# Hook Gradio App creation to filter default routes and mount authentic Flask application at /
orig_create_app = gr.routes.App.create_app


def custom_create_app(*args, **kwargs):
    fastapi_app = orig_create_app(*args, **kwargs)
    fastapi_app.router.routes = [
        r for r in fastapi_app.router.routes
        if getattr(r, "path", "") not in {"/", "/static/{path:path}"}
    ]
    fastapi_app.mount("/", WSGIMiddleware(flask_app))
    return fastapi_app


gr.routes.App.create_app = custom_create_app


# ZeroGPU Runner: This function executes inside the Gradio Event Queue with dynamic GPU lease
def _zerogpu_runner(audio_path: str, decode_mode: str) -> tuple[str, list]:
    print(f"[ZeroGPU] Active GPU execution started for: {audio_path} ({decode_mode})")
    return web_app_mod.transcribe_audio_file(audio_path, decode_mode, device="cuda")


if "spaces" in sys.modules:
    _zerogpu_gpu_fn = spaces.GPU(duration=30)(_zerogpu_runner)
    print("[ZeroGPU] @spaces.GPU(duration=30) attached to /transcribe_zerogpu endpoint")
else:
    _zerogpu_gpu_fn = _zerogpu_runner


# Construct Gradio Blocks demo (satisfies ZeroGPU runtime supervisor & event queue)
with gr.Blocks(title="ClassScript") as demo:
    gr.Markdown("# ClassScript ZeroGPU Active")

    audio_path_input = gr.Textbox(visible=False)
    decode_mode_input = gr.Dropdown(["greedy", "lm"], value="lm", visible=False)
    transcript_output = gr.Textbox(visible=False)
    segments_output = gr.JSON(visible=False)

    transcribe_trigger = gr.Button("Transcribe", visible=False)
    transcribe_trigger.click(
        fn=_zerogpu_gpu_fn,
        inputs=[audio_path_input, decode_mode_input],
        outputs=[transcript_output, segments_output],
        api_name="transcribe_zerogpu",
    )

demo.queue()


def _background_model_warmup():
    """Download model checkpoint in background after server has already bound and passed health checks."""
    time.sleep(5)
    try:
        print("[WARMUP] Starting background model preparation...")
        _ = WebConfig.get_model_path()
        print("[WARMUP] Model checkpoint successfully verified and ready!")
    except Exception as e:
        print(f"[WARMUP ERROR] {e}")


if __name__ == "__main__":
    threading.Thread(target=_background_model_warmup, daemon=True).start()
    port = int(os.environ.get("PORT", 7860 if os.environ.get("SPACE_ID") else 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting ClassScript via Gradio on http://{host}:{port}")
    demo.launch(server_name=host, server_port=port, ssr_mode=False)
