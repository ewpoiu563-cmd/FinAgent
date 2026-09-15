import asyncio
import json
import logging
import re
import uuid
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator
from orchestration.session_state import normalize_chat_history

try:
    from .agent_loop import run_agent
except ImportError:
    from agent_loop import run_agent

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    from .debug_ui import router as debug_ui_router
except ImportError:
    from debug_ui import router as debug_ui_router

app.include_router(debug_ui_router)
PING_INTERVAL_SECONDS = 5
STREAM_CHUNK_CHARS = 24
_SESSION_LOCKS = [Lock() for _ in range(64)]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    question: str
    chat_history: Optional[list] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None

    @field_validator("chat_history")
    @classmethod
    def validate_history(cls, value):
        return normalize_chat_history(value)


class QueryResponse(BaseModel):
    answer: str


def _sse_event(event: str, data: dict | None = None) -> str:
    lines = [f"event: {event}"]
    if data is not None:
        lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


def _iter_answer_chunks(answer: str):
    text = (answer or "").strip()
    if not text:
        yield "未能找到答案。"
        return

    word_chunks = [chunk for chunk in re.findall(r"\S+|\s+\S+", text) if chunk]
    if len(word_chunks) > 1:
        for chunk in word_chunks:
            yield chunk
        return

    for idx in range(0, len(text), STREAM_CHUNK_CHARS):
        yield text[idx: idx + STREAM_CHUNK_CHARS]


def _run_agent_sync(
    question: str,
    session_id: str | None = None,
    run_id: str | None = None,
    chat_history: list | None = None,
) -> str:
    """Run the async agent in a dedicated event loop (for use with to_thread)."""
    kwargs = {"session_id": session_id, "run_id": run_id}
    if chat_history:
        kwargs["chat_history"] = chat_history
    if session_id:
        # Bounded lock storage, shared by the request worker threads in this
        # single-process local demo. This is not a multi-process Redis lock.
        with _SESSION_LOCKS[hash(session_id) % len(_SESSION_LOCKS)]:
            return asyncio.run(run_agent(question, **kwargs))
    return asyncio.run(run_agent(question, **kwargs))


@app.post("/")
async def query(req: QueryRequest):
    async def stream_response():
        question = (req.question or "").strip()
        if not question:
            yield _sse_event("Message", {"answer": "无法识别问题，请重新输入。"})
            return

        task = asyncio.create_task(
            asyncio.to_thread(
                _run_agent_sync,
                question,
                req.session_id or None,
                req.run_id or None,
                req.chat_history,
            )
        )
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=PING_INTERVAL_SECONDS)
                if task in done:
                    break
                yield _sse_event("Ping")

            try:
                answer = task.result()
            except Exception as exc:
                logger.error(f"Agent error: {exc}", exc_info=True)
                answer = "本次处理未完成，请稍后重试；可在本地日志中查看具体原因。"

            for chunk in _iter_answer_chunks(answer):
                yield _sse_event("Message", {"answer": chunk})
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -----------  AG-UI Protocol  -----------

def _extract_question_from_agui(data: dict) -> str:
    """从 AG-UI 请求中提取用户最后一条消息作为 question"""
    messages = data.get("messages", [])
    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and content:
            return content
    return ""


def _agui_event(event_type: str, **kwargs) -> str:
    """构造一条 AG-UI SSE data 行"""
    payload: Dict[str, Any] = {"type": event_type}
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/ag-ui")
async def ag_ui(request: Request) -> StreamingResponse:
    data = await request.json()

    if not isinstance(data, dict) or not isinstance(data.get("messages", []), list):
        raise HTTPException(status_code=422, detail="messages must be a list")
    messages = data.get("messages", [])
    if any(not isinstance(item, dict) for item in messages):
        raise HTTPException(status_code=422, detail="messages must contain objects")
    user_index = next((i for i in range(len(messages) - 1, -1, -1)
                       if messages[i].get("role") == "user" and messages[i].get("content")), None)
    try:
        history = normalize_chat_history([
            item for item in messages[:user_index] if item.get("role") in {"user", "assistant"}
        ]) if user_index is not None else []
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    thread_id = data.get("threadId", "")
    run_id = data.get("runId", "")
    question = _extract_question_from_agui(data)
    if not isinstance(question, str):
        raise HTTPException(status_code=422, detail="user content must be text")

    async def stream_response():
        msg_id = str(uuid.uuid4())

        # RUN_STARTED
        yield _agui_event("RUN_STARTED", threadId=thread_id, runId=run_id)

        if not question:
            # 没有有效问题，直接结束
            yield _agui_event("TEXT_MESSAGE_START", messageId=msg_id, role="assistant")
            yield _agui_event("TEXT_MESSAGE_CONTENT", messageId=msg_id, delta="无法识别问题，请重新输入。")
            yield _agui_event("TEXT_MESSAGE_END", messageId=msg_id)
            yield _agui_event("RUN_FINISHED", threadId=thread_id, runId=run_id)
            return

        task = asyncio.create_task(
            asyncio.to_thread(
                _run_agent_sync,
                question,
                thread_id or None,
                run_id or None,
                history,
            )
        )
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=PING_INTERVAL_SECONDS)
                if task in done:
                    break
                yield ": keepalive\n\n"

            try:
                answer = task.result()
            except Exception as e:
                logger.error(f"Agent error: {e}", exc_info=True)
                answer = "本次处理未完成，请稍后重试；可在本地日志中查看具体原因。"
        finally:
            if not task.done():
                task.cancel()

        if not answer:
            answer = "未能找到答案。"

        # TEXT_MESSAGE_START -> CONTENT -> END
        yield _agui_event("TEXT_MESSAGE_START", messageId=msg_id, role="assistant")
        yield _agui_event("TEXT_MESSAGE_CONTENT", messageId=msg_id, delta=answer)
        yield _agui_event("TEXT_MESSAGE_END", messageId=msg_id)

        # RUN_FINISHED
        yield _agui_event("RUN_FINISHED", threadId=thread_id, runId=run_id)

    return StreamingResponse(stream_response(), media_type="text/event-stream")
