from typing import List, Optional
from pydantic import BaseModel, Field


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
