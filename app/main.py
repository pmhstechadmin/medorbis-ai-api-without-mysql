from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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


@app.post("/api/v2/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    last_user = next((m for m in reversed(req.messages) if m.role.lower() == "user"), None)
    user_text = last_user.content.strip() if last_user and last_user.content else ""
    if user_text:
        reply_text = f"You said: {user_text}"
    else:
        reply_text = "Hello! How can I help you?"

    input_tokens = len(" ".join(m.content for m in req.messages).split()) if req.messages else 0
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
