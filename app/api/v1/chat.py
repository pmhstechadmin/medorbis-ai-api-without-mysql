from typing import Optional
import json
from fastapi import APIRouter, Request, HTTPException
from app.schemas import ChatRequest, ChatResponse, Usage
from app.services.chat_service import generate_reply

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/chat")
async def chat_get() -> dict:
    return {
        "endpoint": "/api/v1/chat",
        "method": "POST",
        "content_type": "application/json",
        "example": {"messages": [{"role": "user", "content": "Hello"}]},
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
        if "messages" in form:
            try:
                data = {"messages": json.loads(form["messages"]) if isinstance(form["messages"], str) else form["messages"]}
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid 'messages' in form; expected JSON string")
        else:
            role = (form.get("role") or "user").strip()
            content = (form.get("content") or "").strip()
            data = {"messages": [{"role": role, "content": content}]}
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Use application/json or form-data with 'messages'")

    try:
        parsed = ChatRequest.model_validate(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body. Expected { messages: [{ role, content }] }")

    return generate_reply(parsed)
