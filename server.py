import base64
import json
import os
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

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
MAX_VIBE_BOOKS = 50
MAX_VIBE_TAGS = 8
MAX_QUESTION_LENGTH = 600

VIBE_CHECK_PROMPT = """
You analyze a known list of books and infer reading taste.

Return JSON only in this exact shape:
{
  "vibeSummary": string,
  "vibeTags": [{"tag": string, "confidence": number}],
  "readingPattern": string,
  "predictedNextGenre": string[],
  "confidence": number
}

Style goals:
- Playful, sharp, interesting, and specific.
- Err on the side of saying something memorable instead of blandly hedging.
- Ground every claim in the provided books.

Safety boundaries:
- Do not infer protected or highly sensitive traits (health status, politics, religion, sexual orientation, disability, trauma, legal status).
- Do not claim facts about life circumstances beyond reading taste.
- Keep tone fun, never insulting.

Field rules:
- vibeSummary: 2-4 sentences, engaging and specific.
- vibeTags: 3-6 short tags about taste profile, each with confidence 0-1.
- readingPattern: 1-2 sentences about genre balance, pacing, and thematic habits.
- predictedNextGenre: 2-4 genre/style suggestions that logically fit their list.
- confidence: overall certainty from 0 to 1.
""".strip()

BOOKS_QUESTION_PROMPT = """
You answer a user question using only a provided list of books.

Return JSON only in this exact shape:
{
    "answer": string
}

Rules:
- Ground your answer in the provided books.
- Do not invent books, authors, or details not supported by the list.
- If the question cannot be answered from the list, say that clearly and provide the closest useful response based on the available books.
- Keep the response concise (2-5 sentences), conversational, and helpful.
- Do not infer sensitive traits (health status, politics, religion, sexual orientation, disability, trauma, legal status).
""".strip()


class VibeBookInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=220)
    author: Optional[str] = Field(default=None, max_length=220)
    isbn: Optional[str] = Field(default=None, max_length=24)


class VibeCheckRequest(BaseModel):
    books: list[VibeBookInput] = Field(..., min_items=1, max_items=MAX_VIBE_BOOKS)


class BooksQuestionRequest(BaseModel):
    books: list[VibeBookInput] = Field(..., min_items=1, max_items=MAX_VIBE_BOOKS)
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)


def _strip_markdown_fence(text: str) -> str:
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return text


def _normalize_vibe_tags(raw_tags: object) -> list[dict[str, object]]:
    if not isinstance(raw_tags, list):
        return []

    tags: list[dict[str, object]] = []
    seen = set()

    for raw_tag in raw_tags:
        if not isinstance(raw_tag, dict):
            continue

        tag_name = str(raw_tag.get("tag", "")).strip()
        if not tag_name:
            continue

        key = tag_name.lower()
        if key in seen:
            continue

        seen.add(key)

        try:
            confidence = float(raw_tag.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        confidence = max(0.0, min(1.0, confidence))
        tags.append({"tag": tag_name, "confidence": confidence})

        if len(tags) >= MAX_VIBE_TAGS:
            break

    return tags


def _normalize_string_list(raw_values: object, *, min_length: int = 2) -> list[str]:
    if not isinstance(raw_values, list):
        return []

    values: list[str] = []
    seen = set()

    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue

        value = " ".join(raw_value.split()).strip()
        if len(value) < min_length:
            continue

        key = value.lower()
        if key in seen:
            continue

        seen.add(key)
        values.append(value)

    return values


def _normalize_confidence(raw_value: object, *, fallback: float = 0.55) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return fallback

    return max(0.0, min(1.0, value))


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


@app.post("/api/vibe-check")
async def vibe_check(request: VibeCheckRequest):
    try:
        books_lines = []
        for book in request.books:
            title = " ".join(book.title.split()).strip()
            author = " ".join(book.author.split()).strip() if book.author else ""
            isbn = " ".join(book.isbn.split()).strip() if book.isbn else ""

            line = f"- {title}"
            if author:
                line += f" by {author}"
            if isbn:
                line += f" (ISBN: {isbn})"
            books_lines.append(line)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": VIBE_CHECK_PROMPT,
                        },
                        {
                            "type": "input_text",
                            "text": (
                                "Analyze this reading list and return the required JSON.\n\n"
                                + "\n".join(books_lines)
                            ),
                        },
                    ],
                }
            ],
        )

        text = _strip_markdown_fence(response.output_text.strip())
        payload = json.loads(text)

        vibe_summary = str(payload.get("vibeSummary", "")).strip()
        reading_pattern = str(payload.get("readingPattern", "")).strip()

        predicted_next_genre = _normalize_string_list(
            payload.get("predictedNextGenre"),
            min_length=3,
        )
        vibe_tags = _normalize_vibe_tags(payload.get("vibeTags"))

        # Lower confidence slightly for very small book lists.
        confidence_fallback = 0.45 if len(request.books) <= 2 else 0.6
        confidence = _normalize_confidence(
            payload.get("confidence"),
            fallback=confidence_fallback,
        )

        return {
            "vibeSummary": vibe_summary,
            "vibeTags": vibe_tags,
            "readingPattern": reading_pattern,
            "predictedNextGenre": predicted_next_genre,
            "confidence": confidence,
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Vibe-check model returned a non-JSON response.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vibe-check failed: {e}")


@app.post("/api/books-question")
async def books_question(request: BooksQuestionRequest):
    try:
        question = " ".join(request.question.split()).strip()
        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty.",
            )

        books_lines = []
        for book in request.books:
            title = " ".join(book.title.split()).strip()
            author = " ".join(book.author.split()).strip() if book.author else ""
            isbn = " ".join(book.isbn.split()).strip() if book.isbn else ""

            line = f"- {title}"
            if author:
                line += f" by {author}"
            if isbn:
                line += f" (ISBN: {isbn})"
            books_lines.append(line)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": BOOKS_QUESTION_PROMPT,
                        },
                        {
                            "type": "input_text",
                            "text": (
                                "Use this reading list as the only source of truth. "
                                "Answer the question and return the required JSON.\n\n"
                                + "Books:\n"
                                + "\n".join(books_lines)
                                + "\n\nQuestion:\n"
                                + question
                            ),
                        },
                    ],
                }
            ],
        )

        text = _strip_markdown_fence(response.output_text.strip())
        payload = json.loads(text)

        answer = str(payload.get("answer", "")).strip()
        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Books-question model returned an empty answer.",
            )

        return {
            "answer": answer,
        }

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Books-question model returned a non-JSON response.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Books-question failed: {e}")


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