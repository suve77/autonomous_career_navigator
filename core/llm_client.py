"""
core/llm_client.py

Single LLM abstraction layer.
Every agent imports from here — never instantiates a client directly.

Design decisions:
  - Provider agnostic — OpenRouter now, Anthropic direct in production
  - Built-in retry with exponential backoff — handles rate limits gracefully
  - Built-in token tracking — every call tracked for cost observability
  - Built-in error translation — provider errors → our custom exceptions
  - Structured logging on every call — full observability
  - Single configuration point — change model or provider in config.py only

Switching provider:
  OpenRouter → Anthropic direct:
    Change OPENROUTER_BASE_URL and OPENROUTER_API_KEY in .env
    Nothing else changes — all agents work identically

Usage:
  from core.llm_client import chat, chat_with_tools

  # Simple chat — no tools
  response = chat(
      messages = [{"role": "user", "content": "Summarise this JD"}],
      agent    = "fitment"
  )

  # Agentic chat — with tools (ReAct pattern)
  response = chat_with_tools(
      messages = messages,
      tools    = TOOLS,
      agent    = "discovery"
  )
  # Returns LLMResponse with .content, .tool_calls, .usage
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

from config.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME,
    MAX_TOKENS,
    ENVIRONMENT
)
from core.logger     import get_logger
from core.exceptions import LLMCallError

logger = get_logger(__name__)


# ── Response dataclass ────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    """
    Structured response from every LLM call.
    Agents work with this — never with raw provider response objects.
    Decouples agents from provider-specific response formats.

    If we switch from OpenRouter to Anthropic:
      Provider response format changes completely.
      LLMResponse stays identical.
      Agents never know the difference.
    """
    content   : str                     # Text response from LLM
    tool_calls: list[dict] = field(default_factory=list)  # Tool calls if any
    usage     : dict       = field(default_factory=dict)  # Token counts
    model     : str        = ""         # Model that served this request
    finish_reason: str     = ""         # stop | tool_calls | length | error

    @property
    def has_tool_calls(self) -> bool:
        """True if LLM wants to call one or more tools."""
        return len(self.tool_calls) > 0

    @property
    def is_done(self) -> bool:
        """
        True if LLM finished naturally — no more tool calls.
        Agents use this to exit the ReAct loop.
        """
        return not self.has_tool_calls

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


# ── Token accumulator ─────────────────────────────────────────────────────────
class TokenAccumulator:
    """
    Tracks token usage across all LLM calls in one agent run.
    Agents don't manage this — llm_client updates it automatically.

    In production this feeds into:
      - Cost monitoring dashboards
      - Per-user billing (Phase 2 SaaS)
      - Alert when a run exceeds budget
    """

    def __init__(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_tokens      = 0
        self.call_count        = 0

    def add(self, usage: dict):
        self.prompt_tokens     += usage.get("prompt_tokens",     0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens      += usage.get("total_tokens",      0)
        self.call_count        += 1

    def summary(self) -> dict:
        return {
            "prompt_tokens"    : self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens"     : self.total_tokens,
            "call_count"       : self.call_count,
            # Approximate cost — OpenRouter Nemotron free tier = $0
            # Anthropic claude-sonnet = ~$3 per 1M input, $15 per 1M output
            "estimated_cost_usd": round(
                (self.prompt_tokens     / 1_000_000 * 3.0) +
                (self.completion_tokens / 1_000_000 * 15.0),
                6
            )
        }


# ── Global accumulator — one per process ─────────────────────────────────────
# Tracks tokens across all agents in one orchestrator run
_global_accumulator = TokenAccumulator()


# ── Retry configuration ───────────────────────────────────────────────────────
RETRY_CONFIG = {
    "max_attempts"   : 3,
    "base_delay_secs": 2,    # First retry after 2s
    "backoff_factor" : 2,    # 2s → 4s → 8s — exponential backoff
    "max_delay_secs" : 30    # Never wait more than 30s
}


def _calculate_retry_delay(attempt: int) -> float:
    """
    Exponential backoff — standard production retry pattern.
    Attempt 1: 2s
    Attempt 2: 4s
    Attempt 3: 8s
    Never exceeds max_delay_secs.
    """
    delay = RETRY_CONFIG["base_delay_secs"] * (
        RETRY_CONFIG["backoff_factor"] ** (attempt - 1)
    )
    return min(delay, RETRY_CONFIG["max_delay_secs"])


# ── Client factory ────────────────────────────────────────────────────────────
def _get_client() -> OpenAI:
    """
    Creates OpenAI-compatible client.
    OpenRouter uses the OpenAI SDK with a different base_url.
    Anthropic direct would use the Anthropic SDK here.
    This is the one function that changes when switching providers.
    """
    return OpenAI(
        api_key  = OPENROUTER_API_KEY,
        base_url = OPENROUTER_BASE_URL,
        timeout  = 60.0    # 60s timeout — prevents hanging calls
    )


# ── Response parser ───────────────────────────────────────────────────────────
def _parse_response(raw_response: Any, agent: str) -> LLMResponse:
    """
    Translates provider-specific response into our LLMResponse.
    This is the adapter pattern — provider format in, our format out.
    """
    choice        = raw_response.choices[0]
    message       = choice.message
    finish_reason = choice.finish_reason or "stop"

    # Parse tool calls if present
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
                logger.warning(
                    "tool_args_parse_failed",
                    agent    = agent,
                    tool     = tc.function.name,
                    raw_args = tc.function.arguments[:100]
                )
            tool_calls.append({
                "id"  : tc.id,
                "name": tc.function.name,
                "args": args
            })

    # Parse usage
    usage = {}
    if raw_response.usage:
        usage = {
            "prompt_tokens"    : raw_response.usage.prompt_tokens,
            "completion_tokens": raw_response.usage.completion_tokens,
            "total_tokens"     : raw_response.usage.total_tokens
        }

    return LLMResponse(
        content      = message.content or "",
        tool_calls   = tool_calls,
        usage        = usage,
        model        = raw_response.model or MODEL_NAME,
        finish_reason= finish_reason
    )


# ── Core chat function ────────────────────────────────────────────────────────
def chat(
    messages : list[dict],
    agent    : str,
    model    : str = None,
    max_tokens: int = None,
    temperature: float = 0.1    # Low temperature = consistent, focused responses
) -> LLMResponse:
    """
    Simple chat — no tools.
    Used for summarisation, analysis, generation tasks.
    Built-in retry, token tracking, structured logging.

    Args:
      messages   : conversation history in OpenAI format
      agent      : calling agent name — for logging and error context
      model      : override default model (optional)
      max_tokens : override default max tokens (optional)
      temperature: 0.0 = deterministic, 1.0 = creative. Default 0.1 for agents.

    Returns:
      LLMResponse with .content and .usage

    Raises:
      LLMCallError if all retry attempts fail
    """
    model      = model      or MODEL_NAME
    max_tokens = max_tokens or MAX_TOKENS

    last_error = None

    for attempt in range(1, RETRY_CONFIG["max_attempts"] + 1):
        try:
            logger.debug(
                "llm_call_start",
                agent      = agent,
                model      = model,
                attempt    = attempt,
                messages   = len(messages),
                max_tokens = max_tokens
            )

            client      = _get_client()
            raw_response = client.chat.completions.create(
                model      = model,
                messages   = messages,
                max_tokens = max_tokens,
                temperature= temperature
            )

            response = _parse_response(raw_response, agent)

            # Track tokens globally
            _global_accumulator.add(response.usage)

            logger.info(
                "llm_call_complete",
                agent        = agent,
                model        = model,
                finish_reason= response.finish_reason,
                total_tokens = response.total_tokens,
                attempt      = attempt
            )

            return response

        except RateLimitError as e:
            last_error = e
            delay = _calculate_retry_delay(attempt)
            logger.warning(
                "llm_rate_limit",
                agent   = agent,
                attempt = attempt,
                delay_s = delay,
                error   = str(e)
            )
            if attempt < RETRY_CONFIG["max_attempts"]:
                time.sleep(delay)

        except APIConnectionError as e:
            last_error = e
            delay = _calculate_retry_delay(attempt)
            logger.warning(
                "llm_connection_error",
                agent   = agent,
                attempt = attempt,
                delay_s = delay,
                error   = str(e)
            )
            if attempt < RETRY_CONFIG["max_attempts"]:
                time.sleep(delay)

        except APIStatusError as e:
            last_error = e
            # 5xx errors are retryable, 4xx are not
            retryable = e.status_code >= 500
            if not retryable:
                raise LLMCallError(
                    agent       = agent,
                    model       = model,
                    reason      = str(e),
                    status_code = e.status_code,
                    retryable   = False
                )
            delay = _calculate_retry_delay(attempt)
            logger.warning(
                "llm_api_error",
                agent       = agent,
                attempt     = attempt,
                status_code = e.status_code,
                delay_s     = delay
            )
            if attempt < RETRY_CONFIG["max_attempts"]:
                time.sleep(delay)

        except Exception as e:
            # Unexpected error — don't retry
            raise LLMCallError(
                agent     = agent,
                model     = model,
                reason    = str(e),
                retryable = False
            )

    # All retries exhausted
    raise LLMCallError(
        agent   = agent,
        model   = MODEL_NAME,
        reason  = f"All {RETRY_CONFIG['max_attempts']} attempts failed: {last_error}",
        retryable = False
    )


# ── Tool-use chat function ────────────────────────────────────────────────────
def chat_with_tools(
    messages   : list[dict],
    tools      : list[dict],
    agent      : str,
    model      : str  = None,
    max_tokens : int  = None,
    temperature: float = 0.1
) -> LLMResponse:
    """
    Agentic chat — with tools.
    Used by all ReAct agents.
    LLM can respond with text OR tool call requests.

    Args:
      messages : full conversation history including tool results
      tools    : list of tool definitions in OpenAI tool format
      agent    : calling agent name
      model    : override default model (optional)
      max_tokens: override default (optional)

    Returns:
      LLMResponse with .has_tool_calls, .tool_calls, .content

    The caller (agent loop) checks response.has_tool_calls:
      True  → execute tools, append results, call again
      False → agent is done, return final response
    """
    model      = model      or MODEL_NAME
    max_tokens = max_tokens or MAX_TOKENS

    last_error = None

    for attempt in range(1, RETRY_CONFIG["max_attempts"] + 1):
        try:
            logger.debug(
                "llm_tool_call_start",
                agent      = agent,
                model      = model,
                attempt    = attempt,
                messages   = len(messages),
                tools      = len(tools)
            )

            client       = _get_client()
            raw_response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                tools       = tools,
                max_tokens  = max_tokens,
                temperature = temperature
            )

            response = _parse_response(raw_response, agent)

            # Track tokens globally
            _global_accumulator.add(response.usage)

            logger.info(
                "llm_tool_call_complete",
                agent        = agent,
                model        = model,
                has_tools    = response.has_tool_calls,
                tool_names   = [tc["name"] for tc in response.tool_calls],
                total_tokens = response.total_tokens,
                finish_reason= response.finish_reason,
                attempt      = attempt
            )

            return response

        except RateLimitError as e:
            last_error = e
            delay = _calculate_retry_delay(attempt)
            logger.warning(
                "llm_rate_limit",
                agent   = agent,
                attempt = attempt,
                delay_s = delay
            )
            if attempt < RETRY_CONFIG["max_attempts"]:
                time.sleep(delay)

        except APIConnectionError as e:
            last_error = e
            delay = _calculate_retry_delay(attempt)
            logger.warning(
                "llm_connection_error",
                agent   = agent,
                attempt = attempt,
                delay_s = delay
            )
            if attempt < RETRY_CONFIG["max_attempts"]:
                time.sleep(delay)

        except APIStatusError as e:
            last_error = e
            retryable = e.status_code >= 500
            if not retryable:
                raise LLMCallError(
                    agent       = agent,
                    model       = model,
                    reason      = str(e),
                    status_code = e.status_code,
                    retryable   = False
                )
            delay = _calculate_retry_delay(attempt)
            logger.warning(
                "llm_api_error",
                agent       = agent,
                attempt     = attempt,
                status_code = e.status_code,
                delay_s     = delay
            )
            if attempt < RETRY_CONFIG["max_attempts"]:
                time.sleep(delay)

        except Exception as e:
            raise LLMCallError(
                agent     = agent,
                model     = model,
                reason    = str(e),
                retryable = False
            )

    raise LLMCallError(
        agent     = agent,
        model     = MODEL_NAME,
        reason    = f"All {RETRY_CONFIG['max_attempts']} attempts failed: {last_error}",
        retryable = False
    )


# ── Token summary ─────────────────────────────────────────────────────────────
def get_token_summary() -> dict:
    """
    Returns token usage across all LLM calls this run.
    Orchestrator calls this at end of run for observability.
    """
    return _global_accumulator.summary()


def reset_token_accumulator():
    """
    Resets token counter.
    Call at start of each orchestrator run for clean per-run tracking.
    """
    global _global_accumulator
    _global_accumulator = TokenAccumulator()


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.config import ensure_directories
    ensure_directories()

    print("\n── LLM Client self test ──────────────────────────────────")
    print(f"Provider  : {OPENROUTER_BASE_URL}")
    print(f"Model     : {MODEL_NAME}")
    print(f"Environment: {ENVIRONMENT}\n")

    # Test 1: Simple chat
    print("[1] Testing simple chat...")
    response = chat(
        messages = [
            {
                "role"   : "system",
                "content": "You are a helpful assistant. Be concise."
            },
            {
                "role"   : "user",
                "content": "In one sentence — what is an autonomous AI agent?"
            }
        ],
        agent = "test"
    )
    print(f"    Response     : {response.content}")
    print(f"    Tokens used  : {response.total_tokens}")
    print(f"    Finish reason: {response.finish_reason}")
    print(f"    Is done      : {response.is_done}")

    # Test 2: Token tracking
    print("\n[2] Token summary after 1 call:")
    summary = get_token_summary()
    for k, v in summary.items():
        print(f"    {k:25} : {v}")

    # Test 3: Tool use chat
    print("\n[3] Testing chat with tools...")
    test_tools = [
        {
            "type": "function",
            "function": {
                "name"       : "get_weather",
                "description": "Get the current weather for a city",
                "parameters" : {
                    "type"      : "object",
                    "properties": {
                        "city": {
                            "type"       : "string",
                            "description": "City name"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    tool_response = chat_with_tools(
        messages = [
            {
                "role"   : "system",
                "content": "You are a helpful assistant with tools."
            },
            {
                "role"   : "user",
                "content": "What is the weather in Chennai?"
            }
        ],
        tools = test_tools,
        agent = "test"
    )

    print(f"    Has tool calls : {tool_response.has_tool_calls}")
    if tool_response.has_tool_calls:
        for tc in tool_response.tool_calls:
            print(f"    Tool called    : {tc['name']}")
            print(f"    Args           : {tc['args']}")
    else:
        print(f"    Response       : {tool_response.content}")

    # Final token summary
    print("\n[4] Final token summary after all calls:")
    summary = get_token_summary()
    for k, v in summary.items():
        print(f"    {k:25} : {v}")

    print("\n── LLM client tests complete ─────────────────────────────")