import os
from app.schemas import ChatRequest, ChatResponse, Usage, ChatV1Request


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


def generate_v1_llm_reply(req: ChatV1Request) -> ChatResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = req.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = (
        "You are a helpful assistant. Use the provided user context to answer the question.\n\n"
        f"user_type: {req.user_type}\n"
        f"user_id: {req.user_id}\n"
        f"session_id: {req.session_id}\n"
        f"department: {req.user_department}\n"
        f"year: {req.user_year}\n"
        f"semester: {req.user_semester}\n\n"
        f"Question: {req.user_question}\n"
    )

    if not api_key:
        reply_text = (
            f"[Local] Based on your context (dept={req.user_department}, year={req.user_year}, semester={req.user_semester}), "
            f"here's a helpful response to your question: {req.user_question}"
        )
        tokens_in = len(prompt.split())
        tokens_out = len(reply_text.split())
        return ChatResponse(reply=reply_text, usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out))

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for university Q&A."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        reply_text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        reply_text = f"[LLM error] {e}. Falling back to local response: {req.user_question}"

    tokens_in = len(prompt.split())
    tokens_out = len(reply_text.split())
    return ChatResponse(reply=reply_text, usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out))
