import os
import threading
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv

load_dotenv()

_init_lock = threading.Lock()
_inited = False


def ensure_init():
    global _inited
    if _inited:
        return
    with _init_lock:
        if _inited:
            return
        project = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        if not project:
            # Allow local runs without Vertex AI if not configured
            return
        vertexai.init(project=project, location=location)
        _inited = True


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


def get_model(model_name: str | None = None) -> GenerativeModel:
    ensure_init()
    return GenerativeModel(model_name or get_model_name())
