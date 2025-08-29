import os
from app.schemas import ChatRequest, ChatResponse, Usage, ChatV1Request
from typing import List, Optional


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


def _embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    try:
        try:
            from langfuse.openai import OpenAI  # type: ignore
        except Exception:
            from openai import OpenAI
        # Prefer OpenAI embeddings if available (lighter than local models)
        key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key:
            return None
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        client = OpenAI(api_key=key, base_url=base_url) if os.getenv("OPENAI_API_KEY") else OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return None


def _qdrant_search(query: str) -> List[str]:
    try:
        from qdrant_client import QdrantClient
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        collection = os.getenv("QDRANT_COLLECTION", "documents")
        if not url:
            return []
        vecs = _embed_texts([query])
        if not vecs:
            return []
        client = QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)
        res = client.search(collection_name=collection, query_vector=vecs[0], limit=int(os.getenv("QDRANT_TOP_K", "3")))
        contexts: List[str] = []
        for p in res:
            payload = getattr(p, "payload", {}) or {}
            text = payload.get("text") or payload.get("content") or payload.get("chunk")
            if text:
                contexts.append(str(text))
        return contexts
    except Exception:
        return []


def generate_v1_llm_reply(req: ChatV1Request) -> ChatResponse:
    contexts: List[str] = []
    if req.user_type == 0:
        contexts = _qdrant_search(req.user_question)

    prompt = (
        "You are a helpful assistant. Use the provided user context to answer the question.\n\n"
        f"user_type: {req.user_type}\n"
        f"user_id: {req.user_id}\n"
        f"session_id: {req.session_id}\n"
        f"department: {req.user_department}\n"
        f"year: {req.user_year}\n"
        f"semester: {req.user_semester}\n"
        + ("\nRelevant context from search:\n" + "\n---\n".join(contexts) + "\n" if contexts else "\n")
        + f"\nQuestion: {req.user_question}\n"
    )

    # Prefer OpenRouter if configured
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            try:
                from langfuse.openai import OpenAI  # type: ignore
            except Exception:
                from openai import OpenAI
            headers = {}
            site = os.getenv("OPENROUTER_SITE_URL")
            title = os.getenv("OPENROUTER_APP_NAME")
            if site:
                headers["HTTP-Referer"] = site
            if title:
                headers["X-Title"] = title
            model = req.model or os.getenv("OPENROUTER_MODEL", "openrouter/auto")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                default_headers=headers or None,
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for university Q&A."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            reply_text = (resp.choices[0].message.content or "").strip()
            tokens_in = len(prompt.split())
            tokens_out = len(reply_text.split())
            return ChatResponse(reply=reply_text, usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out))
        except Exception as e:
            fallback = f"[OpenRouter error] {e}."
            reply_text = f"{fallback} Based on your context, here's a response: {req.user_question}"
            tokens_in = len(prompt.split())
            tokens_out = len(reply_text.split())
            return ChatResponse(reply=reply_text, usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out))

    # Fallback to OpenAI-compatible endpoint if provided
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = req.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            try:
                from langfuse.openai import OpenAI  # type: ignore
            except Exception:
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
            tokens_in = len(prompt.split())
            tokens_out = len(reply_text.split())
            return ChatResponse(reply=reply_text, usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out))
        except Exception as e:
            reply_text = f"[LLM error] {e}. Falling back to local response: {req.user_question}"
            tokens_in = len(prompt.split())
            tokens_out = len(reply_text.split())
            return ChatResponse(reply=reply_text, usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out))

    # Local fallback when no keys configured
    reply_text = (
        f"[Local] Based on your context (dept={req.user_department}, year={req.user_year}, semester={req.user_semester}), "
        f"here's a helpful response to your question: {req.user_question}"
    )
    tokens_in = len(prompt.split())
    tokens_out = len(reply_text.split())
    return ChatResponse(reply=reply_text, usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out))
