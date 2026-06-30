import base64
import json
import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image

app = FastAPI()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_DIMENSION = 1600
JPEG_QUALITY = 82


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Vision OCR server is running"}


def prepare_image_for_api(image_bytes: bytes) -> str:
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

        output = BytesIO()
        img.save(output, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


@app.post("/api/shelf-ocr")
async def shelf_ocr(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        image_bytes = await image.read()
        image_data_url = prepare_image_for_api(image_bytes)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                           "text": """
You are identifying books from a shelf photo.

Return JSON only in this exact shape:

{
  "rawText": string,
  "confidence": number,
  "titles": string[],
  "moodSummary": string,
  "conversationStarters": string[]
}

Rules for titles:
- Each item should represent one visible book.
- If both title and author are visible, format as:
  "Title — Author"
- If only the title is visible, return just the title.
- Do not write "Unknown Author" or "Unknown Title".
- Only include books you can reasonably identify.
- Include partial titles if they are genuinely visible.
- Do not invent books or authors.

Rules for rawText:
- Include all visible text you can read from the image.

Rules for moodSummary:
- Write 1–2 friendly sentences.
- Describe the apparent themes, genres or interests represented by the books.
- Keep the tone warm and conversational.
- Mention uncertainty where appropriate.
- Avoid making assumptions about the owner's personal identity, beliefs, politics, health or other sensitive characteristics.

Rules for conversationStarters:
- Return 2–3 short conversation starters.
- Base them only on the books that are actually visible.
- Keep them fun, friendly and open-ended.
- They should feel like questions someone could naturally ask while looking at the bookshelf.
- Do not ask anything invasive.
- Avoid assuming the person has read every book.
- Avoid mentioning "the image" or "the shelf".

confidence should be a number between 0 and 1.
""".strip(),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                        },
                    ],
                }
            ],
        )

        text = response.output_text.strip()

        # Remove possible markdown fences just in case.
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()

        payload = json.loads(text)

        return {
            "rawText": payload.get("rawText", ""),
            "confidence": float(payload.get("confidence", 0)),
            "titles": payload.get("titles", []),
            "moodSummary": payload.get("moodSummary", ""),
            "conversationStarters": payload.get("conversationStarters", []),
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Vision model returned a non-JSON response.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision OCR failed: {e}")