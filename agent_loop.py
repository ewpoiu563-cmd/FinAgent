"""Stable agent bridge with direct RAG retry/fallback and SQL-first dispatch."""

try:
    from .orchestration.entrypoint import build_default_entrypoint
    from .react_agent import run_react_agent
except ImportError:
    from orchestration.entrypoint import build_default_entrypoint
    from react_agent import run_react_agent


_ENTRYPOINT = build_default_entrypoint(run_react_agent)


async def run_agent(
    question: str,
    session_id: str | None = None,
    run_id: str | None = None,
    chat_history: list | None = None,
) -> str:
    """Keep the public async answer contract while selecting the source-aware path."""

    kwargs = {"session_id": session_id, "run_id": run_id}
    if chat_history:
        kwargs["chat_history"] = chat_history
    return await _ENTRYPOINT.run(question, **kwargs)


__all__ = ["run_agent"]
