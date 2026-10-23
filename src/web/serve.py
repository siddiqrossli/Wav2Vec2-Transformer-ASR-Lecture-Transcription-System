"""
Local production WSGI server runner using Waitress.
Use this to test the production server locally on Windows before cloud deployment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app
from config import WebConfig

try:
    from waitress import serve
except ImportError:
    print("Waitress is not installed. Install via: pip install waitress")
    sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("Starting ClassScript with Waitress Production WSGI Server")
    print(f"Address: http://{WebConfig.HOST}:{WebConfig.PORT}")
    print(f"Model path: {WebConfig.MODEL_PATH}")
    print("=" * 60)
    serve(app, host=WebConfig.HOST, port=WebConfig.PORT, channel_timeout=600)
