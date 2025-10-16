"""Flask-based MCP server that fuses OCR output with GPT-4o vision reasoning."""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

import pytesseract
from flask import Flask, jsonify, request
from openai import OpenAI
from PIL import Image

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _encode_image_base64(image_path: str) -> str:
    """Return a base64 data URL for the provided image path."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _perform_ocr(image_path: str) -> str:
    """Run Tesseract OCR (English + German) on the provided image path."""
    with Image.open(image_path) as image:
        return pytesseract.image_to_string(image, lang="eng+deu")


def _extract_action(data: Dict[str, Any]) -> Optional[str]:
    """Return the requested action name from multiple possible payload shapes."""

    keys = ("tool", "tool_name", "command", "action")
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value

    type_field = data.get("type")
    if isinstance(type_field, str) and type_field:
        return type_field

    # Some clients send {"name": ..., "arguments": {...}}
    name = data.get("name")
    if isinstance(name, str) and name:
        return name

    return None


def _extract_arguments(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull out arguments from the payload regardless of nesting conventions."""

    if isinstance(data.get("arguments"), dict):
        return data["arguments"]

    if isinstance(data.get("args"), dict):
        return data["args"]

    if isinstance(data.get("input"), dict):
        return data["input"]

    return {}


def _list_tools_response() -> Any:
    """Return the static MCP tool manifest."""

    return jsonify(
        {
            "tools": [
                {
                    "name": "analyze_image",
                    "description": "Perform OCR and visual reasoning on an image.",
                    "parameters": {"image_path": "string"},
                    "returns": {
                        "ocr_text": "string",
                        "vision_analysis": "string",
                    },
                }
            ]
        }
    )


@app.post("/mcp")
def mcp_entry() -> Any:
    """Entry point that dispatches MCP tool calls based on the request payload."""

    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    action = _extract_action(data)

    if not action:
        # Default to a health response so discovery pings don't fail the client.
        return jsonify({"status": "ok", "message": "vision-fusion-mcp ready"})

    normalized_action = action.lower()

    if normalized_action in {"ping", "health", "status"}:
        return jsonify({"status": "ok"})

    if normalized_action in {"tools/list", "list_tools"}:
        return _list_tools_response()

    if normalized_action == "analyze_image":
        args = _extract_arguments(data)
        image_path = args.get("image_path") or args.get("path")
        if not image_path or not os.path.exists(image_path):
            return (
                jsonify({"error": f"Image not found: {image_path}"}),
                400,
            )

        ocr_text = _perform_ocr(image_path).strip()
        img_b64 = _encode_image_base64(image_path)

        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a multimodal perception agent. Combine visual and OCR "
                        "information for precise understanding."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "OCR text extracted:\n"
                                f"{ocr_text}\n"
                                "Now reason visually about this image."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_b64}",
                        },
                    ],
                },
            ],
        )

        answer = (result.choices[0].message.content or "").strip()
        return jsonify(
            {
                "ocr_text": ocr_text,
                "vision_analysis": answer,
            }
        )

    return (
        jsonify(
            {
                "error": "Unknown action",
                "received": action,
                "supported": ["tools/list", "analyze_image"],
            }
        ),
        400,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8500)
