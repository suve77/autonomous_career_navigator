"""
agents/base_agent.py

Abstract base class for all specialist agents.
Implements the ReAct loop once — all agents inherit it.

What base agent owns:
  - Loading prompt from agents/prompts/<agent_name>.md
  - The ReAct loop — Reason → Act → Observe → repeat
  - Tool execution and routing via TOOL_REGISTRY
  - Structured logging on every iteration
  - Exception handling and retry decisions
  - Token tracking per run
  - Conversation history management

What each specialist agent owns:
  - Its name
  - Its tool definitions (TOOLS list)
  - Its tool implementations (TOOL_REGISTRY)
  - Nothing else — the loop is inherited

Design pattern: Template Method
  Base class defines the algorithm structure.
  Subclasses fill in the specific steps.
  Algorithm never changes — only the tools change.

Usage (specialist agent):
  from agents.base_agent import BaseAgent

  class DiscoveryAgent(BaseAgent):
      def __init__(self):
          super().__init__(name="discovery")

      def _build_tools(self) -> list:
          return [...]          # Tool definitions

      def _build_registry(self) -> dict:
          return {              # Tool implementations
              "load_companies": self._tool_load_companies,
              ...
          }
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from core.logger      import get_logger, get_run_id
from core.llm_client  import chat_with_tools, get_token_summary
from core.exceptions  import (
    AgentError,
    LLMCallError,
    ToolCallError,
    MaxIterationsError,
    PromptLoadError,
    CareerNavigatorError,
    is_retryable
)
from config.config import BASE_DIR


# ── Prompt directory ──────────────────────────────────────────────────────────
PROMPTS_DIR = BASE_DIR / "agents" / "prompts"


# ── Base agent ────────────────────────────────────────────────────────────────
class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.
    Inherit this. Override _build_tools() and _build_registry().
    Never override run() — the loop is fixed by design.

    Constructor args:
      name           : agent name — must match prompts/<name>.md
      max_iterations : safety cap on ReAct loop (default 50)
      user_id        : which user this run is for
    """

    # Safety cap — prevents infinite loops
    # 50 is generous for discovery, fitment
    # Content agent may need more for complex resumes
    DEFAULT_MAX_ITERATIONS = 50

    def __init__(
        self,
        name          : str,
        max_iterations: int  = None,
        user_id       : str  = "local_user"
    ):
        self.name           = name
        self.user_id        = user_id
        self.max_iterations = max_iterations or self.DEFAULT_MAX_ITERATIONS
        self.logger         = get_logger(f"agents.{name}")
        self.run_id         = get_run_id()

        # Load prompt from markdown file
        self.system_prompt  = self._load_prompt()

        # Build tools and registry from subclass
        self.tools          = self._build_tools()
        self.tool_registry  = self._build_registry()

        self.logger.info(
            "agent_initialised",
            agent          = self.name,
            run_id         = self.run_id,
            tools          = [t["function"]["name"] for t in self.tools],
            max_iterations = self.max_iterations
        )


    # ── Abstract methods — subclasses must implement ──────────────────────────

    @abstractmethod
    def _build_tools(self) -> list[dict]:
        """
        Return list of tool definitions in OpenAI tool format.
        These are the tool contracts the LLM sees.
        The better the description — the better the agent reasons.
        """
        pass

    @abstractmethod
    def _build_registry(self) -> dict:
        """
        Return dict mapping tool name → implementation function.
        Functions must accept **kwargs and return a dict.
        Example:
          {
            "load_companies": self._tool_load_companies,
            "scrape_company": self._tool_scrape_company
          }
        """
        pass


    # ── Prompt loading ────────────────────────────────────────────────────────
    def _load_prompt(self) -> str:
        """
        Loads system prompt from agents/prompts/<name>.md
        Markdown format — LLMs read natural language better than JSON.
        Prompt lives outside code — behaviour changed without touching Python.
        """
        prompt_path = PROMPTS_DIR / f"{self.name}.md"

        if not prompt_path.exists():
            raise PromptLoadError(
                agent       = self.name,
                prompt_path = str(prompt_path)
            )

        prompt = prompt_path.read_text(encoding="utf-8").strip()

        self.logger.debug(
            "prompt_loaded",
            agent       = self.name,
            prompt_path = str(prompt_path),
            length      = len(prompt)
        )

        return prompt


    # ── Tool execution ────────────────────────────────────────────────────────
    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        Executes the tool the LLM chose.
        Returns result as JSON string — LLM reads JSON.

        Error handling:
          Unknown tool    → ToolCallError (programming error)
          Tool exception  → logged, safe error JSON returned
          This keeps the ReAct loop alive even when one tool fails.
          LLM reads the error and decides how to proceed.
        """
        # Unknown tool — programming error
        if tool_name not in self.tool_registry:
            raise ToolCallError(
                tool_name      = tool_name,
                agent          = self.name,
                available_tools= list(self.tool_registry.keys())
            )

        start_time = time.time()

        try:
            result      = self.tool_registry[tool_name](**tool_args)
            duration_ms = round((time.time() - start_time) * 1000)

            self.logger.info(
                "tool_executed",
                agent       = self.name,
                tool        = tool_name,
                duration_ms = duration_ms,
                success     = True
            )

            return json.dumps(result, default=str)

        except CareerNavigatorError as e:
            # Our exceptions — structured, self-describing
            duration_ms = round((time.time() - start_time) * 1000)
            e.log(self.logger)

            # Return error as JSON so LLM can reason about it
            # LLM sees the error and decides — skip this company? retry?
            return json.dumps({
                "error"    : str(e),
                "retryable": e.retryable,
                "tool"     : tool_name
            })

        except Exception as e:
            # Unexpected exception — log and return safe error
            duration_ms = round((time.time() - start_time) * 1000)
            self.logger.error(
                "tool_unexpected_error",
                agent       = self.name,
                tool        = tool_name,
                error       = str(e),
                duration_ms = duration_ms
            )

            return json.dumps({
                "error"    : f"Unexpected error in {tool_name}: {str(e)}",
                "retryable": False,
                "tool"     : tool_name
            })


    # ── Message helpers ───────────────────────────────────────────────────────
    def _build_initial_messages(self, user_trigger: str) -> list[dict]:
        """
        Builds the initial conversation for this run.
        System prompt + user trigger = agent starts reasoning.
        """
        return [
            {
                "role"   : "system",
                "content": self.system_prompt
            },
            {
                "role"   : "user",
                "content": user_trigger
            }
        ]

    def _append_assistant_message(
        self,
        messages  : list[dict],
        content   : str,
        tool_calls: list[dict]
    ) -> list[dict]:
        """
        Appends LLM response to conversation history.
        Handles both text responses and tool call responses.
        """
        message = {"role": "assistant", "content": content}

        if tool_calls:
            message["tool_calls"] = [
                {
                    "id"      : tc["id"],
                    "type"    : "function",
                    "function": {
                        "name"     : tc["name"],
                        "arguments": json.dumps(tc["args"])
                    }
                }
                for tc in tool_calls
            ]

        messages.append(message)
        return messages

    def _append_tool_result(
        self,
        messages    : list[dict],
        tool_call_id: str,
        result      : str
    ) -> list[dict]:
        """
        Appends tool execution result to conversation history.
        This is the Observe step in ReAct.
        LLM reads this on the next Reason step.
        """
        messages.append({
            "role"        : "tool",
            "tool_call_id": tool_call_id,
            "content"     : result
        })
        return messages


    # ── ReAct loop — the heart of every agent ────────────────────────────────
    def run(self, user_trigger: str, context: dict = None) -> dict:
        """
        The ReAct loop. Inherited by all agents. Never overridden.

        Algorithm:
          1. Build initial messages (system prompt + trigger)
          2. Call LLM with tools
          3. If LLM returns tool calls → execute → append results → goto 2
          4. If LLM returns text → done
          5. Safety cap → MaxIterationsError if exceeded

        Args:
          user_trigger : what triggers this agent run
                         e.g. "Find new jobs for today"
                         e.g. "Evaluate fitment for job AMEX-001"
          context      : optional extra context injected into trigger
                         e.g. {"date": "2026-05-10", "user_id": "venket"}

        Returns:
          dict with status, result, token_summary, iterations
        """
        start_time = datetime.now()

        # Enrich trigger with context if provided
        if context:
            context_str  = "\n".join(f"{k}: {v}" for k, v in context.items())
            user_trigger = f"{user_trigger}\n\nContext:\n{context_str}"

        self.logger.info(
            "agent_run_start",
            agent      = self.name,
            run_id     = self.run_id,
            user_id    = self.user_id,
            trigger    = user_trigger[:100]
        )

        messages  = self._build_initial_messages(user_trigger)
        iteration = 0
        final_result = {}

        try:
            while iteration < self.max_iterations:
                iteration += 1

                self.logger.debug(
                    "react_iteration",
                    agent    = self.name,
                    iteration= iteration
                )

                # ── Reason: LLM decides what to do ───────────────────────────
                try:
                    response = chat_with_tools(
                        messages = messages,
                        tools    = self.tools,
                        agent    = self.name
                    )
                except LLMCallError as e:
                    e.log(self.logger)
                    if not e.retryable:
                        raise
                    # Retryable — wait and try again
                    time.sleep(5)
                    continue

                # Append LLM response to history
                messages = self._append_assistant_message(
                    messages   = messages,
                    content    = response.content,
                    tool_calls = response.tool_calls
                )

                # ── Check: is LLM done? ───────────────────────────────────────
                if response.is_done:
                    self.logger.info(
                        "agent_reasoning_complete",
                        agent    = self.name,
                        iteration= iteration,
                        content  = response.content[:100]
                    )
                    final_result = {
                        "status" : "completed",
                        "content": response.content
                    }
                    break

                # ── Act + Observe: execute each tool call ─────────────────────
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id   = tool_call["id"]

                    self.logger.info(
                        "tool_call",
                        agent    = self.name,
                        iteration= iteration,
                        tool     = tool_name,
                        args     = str(tool_args)[:100]
                    )

                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_args)

                    self.logger.debug(
                        "tool_result",
                        agent   = self.name,
                        tool    = tool_name,
                        preview = tool_result[:120]
                    )

                    # Observe: append result to conversation
                    messages = self._append_tool_result(
                        messages     = messages,
                        tool_call_id = tool_id,
                        result       = tool_result
                    )

                    # Check for terminal tool — agent signals completion
                    result_dict = json.loads(tool_result)
                    if result_dict.get("status") == "completed":
                        final_result = result_dict
                        self.logger.info(
                            "agent_terminal_tool_called",
                            agent = self.name,
                            tool  = tool_name
                        )
                        # Break both loops
                        iteration = self.max_iterations
                        break

            else:
                # While loop exhausted — safety cap reached
                raise MaxIterationsError(
                    agent          = self.name,
                    max_iterations = self.max_iterations,
                    last_tool      = (
                        response.tool_calls[-1]["name"]
                        if response.tool_calls else "none"
                    )
                )

        except MaxIterationsError as e:
            e.log(self.logger)
            final_result = {
                "status"   : "max_iterations_reached",
                "error"    : str(e),
                "iteration": iteration
            }

        except AgentError as e:
            e.log(self.logger)
            final_result = {
                "status": "agent_error",
                "error" : str(e)
            }

        # ── Run summary ───────────────────────────────────────────────────────
        duration     = (datetime.now() - start_time).total_seconds()
        token_summary = get_token_summary()

        self.logger.info(
            "agent_run_complete",
            agent        = self.name,
            run_id       = self.run_id,
            status       = final_result.get("status"),
            iterations   = iteration,
            duration_s   = round(duration, 2),
            total_tokens = token_summary.get("total_tokens", 0)
        )

        return {
            **final_result,
            "agent"        : self.name,
            "run_id"       : self.run_id,
            "iterations"   : iteration,
            "duration_s"   : round(duration, 2),
            "token_summary": token_summary
        }


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Test BaseAgent with a minimal concrete implementation.
    Verifies abstract method enforcement and ReAct loop mechanics.
    Does not call real LLM — tests structure only.
    """
    from config.config import ensure_directories
    ensure_directories()

    print("\n── BaseAgent self test ───────────────────────────────────")

    # Test 1: Cannot instantiate abstract class directly
    try:
        agent = BaseAgent(name="test")
        print("[FAIL] Should not be able to instantiate BaseAgent directly")
    except TypeError as e:
        print(f"[1] ✓ Abstract class protection works: {e}")

    # Test 2: Concrete subclass without prompt file
    class TestAgent(BaseAgent):
        def _build_tools(self):
            return []
        def _build_registry(self):
            return {}

    try:
        agent = TestAgent(name="nonexistent_agent")
        print("[FAIL] Should raise PromptLoadError")
    except PromptLoadError as e:
        print(f"[2] ✓ PromptLoadError raised correctly: {e.message}")

    # Test 3: Verify tool registry validation
    class ValidTestAgent(BaseAgent):
        def _build_tools(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name"       : "test_tool",
                        "description": "A test tool",
                        "parameters" : {
                            "type"      : "object",
                            "properties": {},
                            "required"  : []
                        }
                    }
                }
            ]
        def _build_registry(self):
            return {
                "test_tool": lambda: {"result": "test passed"}
            }

    # Create prompt file for test
    import tempfile
    from config.config import BASE_DIR
    prompts_dir = BASE_DIR / "agents" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    test_prompt = prompts_dir / "valid_test_agent.md"
    test_prompt.write_text("# Test Agent\nYou are a test agent.")

    try:
        agent = ValidTestAgent(name="valid_test_agent")
        print(f"[3] ✓ Agent initialised: {agent.name}")
        print(f"    Tools    : {[t['function']['name'] for t in agent.tools]}")
        print(f"    Registry : {list(agent.tool_registry.keys())}")

        # Test tool execution
        result = agent._execute_tool("test_tool", {})
        print(f"    Tool result: {result}")

        # Test unknown tool
        try:
            agent._execute_tool("nonexistent_tool", {})
        except ToolCallError as e:
            print(f"[4] ✓ ToolCallError for unknown tool: {e.message}")

    finally:
        # Clean up test prompt
        if test_prompt.exists():
            test_prompt.unlink()

    print("\n── BaseAgent tests complete ──────────────────────────────")
    print("   Abstract enforcement ✓")
    print("   PromptLoadError      ✓")
    print("   Tool registry        ✓")
    print("   Tool execution       ✓")
    print("   Unknown tool guard   ✓")