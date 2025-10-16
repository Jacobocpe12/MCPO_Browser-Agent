"""Flask-based MCP server that fuses OCR output with GPT-4o vision reasoning."""
from __future__ import annotations

import base64
import os
from typing import Any, Dict

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


@app.post("/mcp")
def mcp_entry() -> Any:
    """Entry point that dispatches MCP tool calls based on the request payload."""
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    tool = data.get("tool")
    args: Dict[str, Any] = data.get("arguments", {})

    if tool == "analyze_image":
        image_path = args.get("image_path")
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
        return jsonify({
            "ocr_text": ocr_text,
            "vision_analysis": answer,
        })

    if tool == "tools/list":
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

    return jsonify({"error": "Unknown tool"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8500)
