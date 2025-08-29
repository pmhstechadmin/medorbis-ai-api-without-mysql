from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI(title="MedOrbis AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str = Field(description="Role of the message sender, e.g., 'user' or 'assistant'")
    content: str = Field(description="Text content of the message")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    model: Optional[str] = None
    stream: bool = False


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    reply: str
    usage: Usage


@app.get("/api/v2/chat")
async def chat_get() -> dict:
    return {
        "endpoint": "/api/v2/chat",
        "method": "POST",
        "content_type": "application/json",
        "example": {"messages": [{"role": "user", "content": "Hello"}]},
    }


@app.post("/api/v2/chat", response_model=ChatResponse)
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
            # Allow simple role/content fields as a convenience
            role = (form.get("role") or "user").strip()
            content = (form.get("content") or "").strip()
            data = {"messages": [{"role": role, "content": content}]}
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Use application/json or form-data with 'messages'")

    try:
        parsed = ChatRequest.model_validate(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body. Expected { messages: [{ role, content }] }")

    last_user = next((m for m in reversed(parsed.messages) if m.role.lower() == "user"), None)
    user_text = last_user.content.strip() if last_user and last_user.content else ""
    reply_text = f"You said: {user_text}" if user_text else "Hello! How can I help you?"

    input_tokens = len(" ".join(m.content for m in parsed.messages).split()) if parsed.messages else 0
    output_tokens = len(reply_text.split())
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    return ChatResponse(reply=reply_text, usage=usage)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict:
    return {"message": "API is running"}
