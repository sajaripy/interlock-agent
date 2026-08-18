"""Model construction, with a seam for tests to swap in a scripted model."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import get_settings

_override: BaseChatModel | None = None


def set_llm(model: BaseChatModel | None) -> None:
    """Replace the model process-wide. Tests use this to run without an API key."""
    global _override
    _override = model


def get_llm() -> BaseChatModel:
    if _override is not None:
        return _override

    from langchain_anthropic import ChatAnthropic

    settings = get_settings()
    # `thinking` is deliberately not set: Claude Opus 5 runs adaptive thinking
    # by default, which is what we want for deciding when a tool is warranted.
    return ChatAnthropic(
        model=settings.model,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
    )


class ScriptedChatModel(BaseChatModel):
    """A stand-in model that replays a fixed list of messages, one per call.

    Used by the test suite and by `scripts/demo.py` so the gate can be
    exercised deterministically, with no API key and no network. It is the
    only way to assert "the graph paused *before* the side effect" without
    the model's choices varying between runs.
    """

    responses: list[BaseMessage] = []
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        # The script already decides which tools get called.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = len(self.calls) - 1
        message = self.responses[index] if index < len(self.responses) else AIMessage(content="Done.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, None, **kwargs)


def tool_call_message(name: str, args: dict[str, Any], call_id: str, text: str = "") -> AIMessage:
    """Build an AIMessage that requests one tool, as the real model would."""
    return AIMessage(
        content=text,
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
