import os
import re
import tempfile
from typing import Any

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR

app = FastAPI()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def extract_text_items(result: Any):
    text_items = []

    for page in result:
        data = page.json if hasattr(page, "json") else page

        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        if not isinstance(data, dict):
            continue

        rec_texts = data.get("rec_texts") or data.get("texts") or []
        rec_scores = data.get("rec_scores") or data.get("scores") or []

        if isinstance(rec_texts, str):
            rec_texts = [rec_texts]

        if isinstance(rec_scores, (int, float)):
            rec_scores = [rec_scores]

        for i, text in enumerate(rec_texts):
            score = 0.0
            if i < len(rec_scores):
                try:
                    score = float(rec_scores[i])
                except Exception:
                    score = 0.0

            text_items.append({
                "text": normalize_line(str(text)),
                "confidence": score,
            })

    return text_items


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Shelf OCR server is running"}


@app.post("/api/shelf-ocr")
async def shelf_ocr(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = os.path.splitext(image.filename or "")[1] or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        result = ocr.predict(tmp_path)
        text_items = extract_text_items(result)

        titles = [item["text"] for item in text_items]
        raw_text = "\n".join(titles)

        confidence_values = [item["confidence"] for item in text_items]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0
        )

        return {
            "rawText": raw_text,
            "confidence": confidence,
            "titles": titles,
            "lines": text_items,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass