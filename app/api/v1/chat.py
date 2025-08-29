from typing import Optional
import json
from fastapi import APIRouter, Request, HTTPException
from app.schemas import ChatRequest, ChatResponse, Usage, ChatV1Request
from app.services.chat_service import generate_reply, generate_v1_llm_reply

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/chat")
async def chat_get() -> dict:
    return {
        "endpoint": "/api/v1/chat",
        "method": "POST",
        "content_type": "application/json or multipart/form-data",
        "example": {
            "user_type": 0,
            "user_id": "string",
            "session_id": "string",
            "user_question": "string",
            "user_department": "string",
            "user_year": "string",
            "user_semester": "string"
        },
        "form_example": {
            "user_type": "0",
            "user_id": "u1",
            "session_id": "s1",
            "user_question": "Hello",
            "user_department": "CSE",
            "user_year": "2",
            "user_semester": "4"
        }
    }


@router.get("/chat/echo", response_model=ChatResponse)
async def chat_echo(content: Optional[str] = None) -> ChatResponse:
    user_text = (content or "").strip()
    reply_text = f"You said: {user_text}" if user_text else "Hello! How can I help you?"
    usage = Usage(
        input_tokens=len(user_text.split()),
        output_tokens=len(reply_text.split()),
        total_tokens=len(user_text.split()) + len(reply_text.split()),
    )
    return ChatResponse(reply=reply_text, usage=usage)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: Request) -> ChatResponse:
    ct = req.headers.get("content-type", "").lower()
    data: dict
    if "application/json" in ct:
        data = await req.json()
    elif "multipart/form-data" in ct or "application/x-www-form-urlencoded" in ct:
        form = await req.form()
        # Build dict from form fields
        data = {
            "user_type": int(form.get("user_type", 0)) if str(form.get("user_type", "")).strip() != "" else 0,
            "user_id": form.get("user_id", ""),
            "session_id": form.get("session_id", ""),
            "user_question": form.get("user_question", ""),
            "user_department": form.get("user_department", ""),
            "user_year": form.get("user_year", ""),
            "user_semester": form.get("user_semester", ""),
        }
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Use application/json or form-data")

    try:
        parsed = ChatV1Request.model_validate(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body. Expected v1 fields")

    return generate_v1_llm_reply(parsed)
