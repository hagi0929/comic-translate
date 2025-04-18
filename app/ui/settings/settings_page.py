import os, shutil
from typing import List
import os, tomllib
from pathlib import Path

def load_toml_env(toml_path: str | Path = "settings.env.toml") -> None:
    """Read TOML file and push every leaf node into os.environ.

    Nested keys are flattened with double‑underscores:
        [credentials.microsoft_azure]
        api_key_ocr = "abc"
    becomes
        CREDENTIALS__MICROSOFT_AZURE__API_KEY_OCR = "abc"
    """
    path = Path(toml_path)
    if not path.exists():
        return

    def _walk(node, prefix=""):
        for k, v in node.items():
            key = f"{prefix}__{k}".upper() if prefix else k.upper()
            if isinstance(v, dict):
                _walk(v, key)
            else:
                # keep existing env‑var if already set (e.g. real OS env wins)
                os.environ.setdefault(key, str(v))

    with path.open("rb") as f:
        _walk(tomllib.load(f))

# Promote TOML values → os.environ *once* when this module is imported
load_toml_env()                       # defaults to "settings.env.toml"

class SettingsPage:
    """
    Pure‑Python version: all values come from environment variables
    that were filled by `load_toml_env()` above.
    """

    # ---------- simple getters ----------
    def get_language(self) -> str:
        return os.getenv("LANGUAGE", "English")

    def get_max_threads(self) -> str:
        return os.getenv("MAX_THREADS", 5)

    def get_tool_selection(self, tool_type: str) -> str:
        mapping = {
            "translator": os.getenv("TOOLS__TRANSLATOR", "GPT-4o"),
            "ocr"       : os.getenv("TOOLS__OCR", "Default"),
            "inpainter" : os.getenv("TOOLS__INPAINTER", "LaMa"),
            "detector"  : os.getenv("TOOLS__DETECTOR", "RT-DETR-V2"),
        }
        return mapping[tool_type]

    def is_gpu_enabled(self) -> bool:
        return os.getenv("TOOLS__USE_GPU", "false").lower() == "true"

    # ---------- compound getters ----------
    def get_hd_strategy_settings(self) -> dict:
        strat = os.getenv("TOOLS__HD_STRATEGY__STRATEGY", "Resize")
        cfg = {"strategy": strat}
        if strat == "Resize":
            cfg["resize_limit"] = int(os.getenv("TOOLS__HD_STRATEGY__RESIZE_LIMIT", 960))
        elif strat == "Crop":
            cfg["crop_margin"]  = int(os.getenv("TOOLS__HD_STRATEGY__CROP_MARGIN", 512))
            cfg["crop_trigger_size"] = int(os.getenv("TOOLS__HD_STRATEGY__CROP_TRIGGER_SIZE", 512))
        return cfg

    def get_llm_settings(self) -> dict:
        return {
            "extra_context": os.getenv("LLM__EXTRA_CONTEXT", ""),
            "image_input_enabled": os.getenv("LLM__IMAGE_INPUT_ENABLED", "true").lower() == "true",
        }

    def get_export_settings(self) -> dict:
        save_as_defaults = {
            ".pdf": "pdf", ".epub": "epub", ".cbr": "cbr",
            ".cbz": "cbz", ".cb7": "cb7", ".cbt": "cbt",
        }
        save_as = {}
        for ext, default in save_as_defaults.items():
            key = f"EXPORT__SAVE_AS__{ext}".upper().replace(".", "")
            save_as[ext] = os.getenv(key, default)
        return {
            "export_raw_text":  os.getenv("EXPORT__EXPORT_RAW_TEXT",  "false").lower() == "true",
            "export_translated_text": os.getenv("EXPORT__EXPORT_TRANSLATED_TEXT", "false").lower() == "true",
            "export_inpainted_image": os.getenv("EXPORT__EXPORT_INPAINTED_IMAGE", "false").lower() == "true",
            "save_as": save_as,
        }

    def get_credentials(self, service: str = "") -> dict | dict[str, dict]:
        """Return creds for one service or for all (if service == '')."""
        def _svc(s):
            base = f"CREDENTIALS__{s.upper().replace(' ', '_')}"
            env = os.getenv
            if s == "Azure":
                return {
                    "api_key_ocr":       env(f"{base}__API_KEY_OCR", ""),
                    "api_key_translator":env(f"{base}__API_KEY_TRANSLATOR", ""),
                    "region_translator": env(f"{base}__REGION_TRANSLATOR", ""),
                    "endpoint":          env(f"{base}__ENDPOINT", ""),
                }
            elif s == "Custom":
                return {
                    "api_key": env(f"{base}__API_KEY", ""),
                    "api_url": env(f"{base}__API_URL", ""),
                    "model":   env(f"{base}__MODEL", ""),
                }
            elif s == "Yandex":
                return {
                    "api_key":   env(f"{base}__API_KEY", ""),
                    "folder_id": env(f"{base}__FOLDER_ID", ""),
                }
            else:  # generic single‑key service
                return {"api_key": env(f"{base}__API_KEY", "")}

        if service:
            return _svc(service)
        else:
            # enumerate the services you care about
            services = ["Gemini"]
            return {s: _svc(s) for s in services}

    # ---------- convenience -----------
    def get_all_settings(self) -> dict:
        return {
            "language": self.get_language(),
            "tools": {
                "translator": self.get_tool_selection("translator"),
                "ocr":        self.get_tool_selection("ocr"),
                "detector":   self.get_tool_selection("detector"),
                "inpainter":  self.get_tool_selection("inpainter"),
                "use_gpu":    self.is_gpu_enabled(),
                "hd_strategy":self.get_hd_strategy_settings(),
            },
            "llm":         self.get_llm_settings(),
            "export":      self.get_export_settings(),
            "credentials": self.get_credentials(),
        }

    # ---------- optional: font import  (unchanged) ----------
    def import_font(self, file_paths: List[str]):
        # identical to your original; omitted here for brevity
        ...

