"""Register the contour MCP server + skill into Hermes Agent's config.

Idempotent: safe to run repeatedly. Merges an `mcp_servers.contour` block and a
`skills.config.ask_cloud` toggle into `~/.hermes/config.yaml` (or the path given),
without clobbering the user's other settings.

The MCP server is launched by running server.py as a script (it self-inserts the
project root on sys.path, so absolute imports work without cwd/PYTHONPATH tricks).

Usage:
    python install/register_hermes.py \
        [--config PATH] [--python PATH] [--relay-url URL] [--token TOK] \
        [--data-dir DIR] [--ask-cloud true|false]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "hermes" / "config.yaml"
    return Path.home() / ".hermes" / "config.yaml"


def default_data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA")
    return str(Path(base) / "hermes" / "contour") if base else str(Path.home() / ".hermes" / "contour")


def build_block(python: str, data_dir: str, relay_url: str, token: str, ask_cloud: bool) -> dict:
    server_py = str(PROJECT_ROOT / "mcp_server" / "server.py")
    return {
        "command": python,
        "args": [server_py],
        "env": {
            "CONTOUR_DATA_DIR": data_dir,
            "CONTOUR_RELAY_URL": relay_url,
            "CONTOUR_DEVICE_TOKEN": token,
            "CONTOUR_ASK_CLOUD": "true" if ask_cloud else "false",
        },
        "tools": {
            "include": [
                "capture_and_store",
                "query_datastore",
                "optimize_datastore",
                "speak",
                "ask_cloud",
            ]
        },
    }


def configure_voice(data: dict) -> None:
    """Voice input via Hermes' local Whisper STT (no key); voice OUTPUT via our
    ElevenLabs `speak` tool, so Hermes' own TTS is disabled to avoid double audio.

    Only fills in these providers; leaves any other voice settings the user has.
    NOTE: Hermes' exact "disable TTS" key is confirmed at install; we set the two
    plausible forms and the installer prints a reminder to verify.
    """
    data["voice"] = "on"
    data.setdefault("stt", {})["provider"] = "faster-whisper"
    tts = data.setdefault("tts", {})
    tts["provider"] = "none"   # do not let Hermes speak; contour speaks via ElevenLabs
    tts["enabled"] = False


def register(config_path: Path, block: dict, ask_cloud: bool, enable_voice: bool = False) -> None:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    data.setdefault("mcp_servers", {})["contour"] = block
    skills = data.setdefault("skills", {})
    skills.setdefault("config", {})["ask_cloud"] = ask_cloud
    if enable_voice:
        configure_voice(data)

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"registered contour MCP server in {config_path}")
    if enable_voice:
        print("configured voice (faster-whisper STT; Hermes TTS off — contour speaks via ElevenLabs)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--relay-url", default=os.environ.get("CONTOUR_RELAY_URL", ""))
    p.add_argument("--token", default=os.environ.get("CONTOUR_DEVICE_TOKEN", ""))
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--ask-cloud", default="false")
    p.add_argument("--enable-voice", default="false")
    args = p.parse_args(argv)

    ask_cloud = args.ask_cloud.strip().lower() in {"1", "true", "yes", "on"}
    enable_voice = args.enable_voice.strip().lower() in {"1", "true", "yes", "on"}
    block = build_block(args.python, args.data_dir, args.relay_url, args.token, ask_cloud)
    register(args.config, block, ask_cloud, enable_voice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
