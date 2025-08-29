from app.schemas import ChatRequest, ChatResponse, Usage


def generate_reply(req: ChatRequest) -> ChatResponse:
    last_user = next((m for m in reversed(req.messages) if m.role.lower() == "user"), None)
    user_text = last_user.content.strip() if last_user and last_user.content else ""
    reply_text = f"You said: {user_text}" if user_text else "Hello! How can I help you?"

    input_tokens = len(" ".join(m.content for m in req.messages).split()) if req.messages else 0
    output_tokens = len(reply_text.split())
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    return ChatResponse(reply=reply_text, usage=usage)
