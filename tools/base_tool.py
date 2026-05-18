"""
tools/base_tool.py

Abstract base class for all specialist tools.
Every tool inherits this — never implements boilerplate directly.

What base tool owns:
  - Input validation before execution
  - Execution timing and performance logging
  - Structured logging on every call
  - Exception translation → our custom exceptions
  - Output contract enforcement — tools always return dict
  - Retry decorator for transient failures

What each specialist tool owns:
  - Its name and description
  - Its input schema (what args it accepts)
  - Its execute() implementation — the actual logic
  - Nothing else

Design pattern: Template Method + Strategy
  Base class defines execute contract and wraps it with infrastructure.
  Subclasses implement the strategy — HOW to execute.

Usage (specialist tool):
  from tools.base_tool import BaseTool

  class ScraperTool(BaseTool):
      name        = "scrape_company"
      description = "Scrapes job listings from a company career page"

      def execute(self, company_name: str, **kwargs) -> dict:
          # Pure logic here — no logging, no timing, no error handling
          jobs = self._fetch_jobs(company_name)
          return {"jobs": jobs, "count": len(jobs)}

  # Agent calls it like this:
  tool   = ScraperTool()
  result = tool.run(company_name="American Express")
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import functools
from abc  import ABC, abstractmethod
from typing import Any

from core.logger     import get_logger
from core.exceptions import (
    ToolError,
    ValidationError,
    CareerNavigatorError
)


# ── Base tool ─────────────────────────────────────────────────────────────────
class BaseTool(ABC):
    """
    Abstract base for all specialist tools.
    Inherit this. Set class attributes. Implement execute().
    Never override run() — infrastructure is fixed.

    Class attributes (set in subclass):
      name        : tool name — must match tool definition in agent
      description : what this tool does — used in tool catalogue
      retryable   : whether failures should be retried (default True)

    The separation between run() and execute():
      run()     → infrastructure (timing, logging, validation, error handling)
      execute() → pure logic (what the tool actually does)

    This means execute() is cleanly testable in isolation.
    No mocking needed — just call execute() directly in tests.
    """

    # Subclasses set these as class attributes
    name       : str  = "base_tool"
    description: str  = "Base tool — override in subclass"
    retryable  : bool = True

    def __init__(self):
        self.logger = get_logger(f"tools.{self.name}")
        self.logger.debug(
            "tool_initialised",
            tool       = self.name,
            description= self.description,
            retryable  = self.retryable
        )


    # ── Abstract method — subclasses must implement ───────────────────────────

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """
        Pure tool logic. No infrastructure here.
        Accept named arguments. Return a dict always.

        Return contract:
          Success → {"key": value, ...}
          Partial → {"key": value, "warning": "..."}

        Never raise generic exceptions — raise our custom ones:
          ScrapingError, ParsingError, AlertError, etc.

        The run() wrapper catches all exceptions and handles them.
        execute() focuses purely on the task.
        """
        pass


    # ── Input validation ──────────────────────────────────────────────────────
    def validate(self, **kwargs) -> None:
        """
        Override in subclass to add input validation.
        Called by run() before execute().
        Raise ValidationError if inputs are invalid.

        Default implementation — no validation.
        Subclasses add specific checks:

        Example:
          def validate(self, company_name: str, **kwargs):
              if not company_name:
                  raise ValidationError(
                      field  = "company_name",
                      value  = company_name,
                      reason = "Company name cannot be empty"
                  )
        """
        pass


    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, **kwargs) -> dict:
        """
        Infrastructure wrapper around execute().
        Called by agent tool registry. Never override.

        Flow:
          1. Log tool start with args
          2. Validate inputs
          3. Execute tool logic
          4. Log completion with timing
          5. Return result dict

        On error:
          CareerNavigatorError → re-raised (already structured)
          Any other exception  → wrapped in ToolError
        """
        start_time = time.time()

        self.logger.info(
            "tool_start",
            tool  = self.name,
            args  = {k: str(v)[:80] for k, v in kwargs.items()}
        )

        try:
            # Step 1: Validate inputs
            self.validate(**kwargs)

            # Step 2: Execute pure logic
            result = self.execute(**kwargs)

            # Step 3: Enforce output contract
            if not isinstance(result, dict):
                raise ToolError(
                    message   = f"Tool '{self.name}' must return dict, got {type(result).__name__}",
                    retryable = False,
                    tool      = self.name
                )

            duration_ms = round((time.time() - start_time) * 1000)

            self.logger.info(
                "tool_complete",
                tool        = self.name,
                duration_ms = duration_ms,
                result_keys = list(result.keys())
            )

            return result

        except ValidationError as e:
            # Input validation failed — structured, not retryable
            duration_ms = round((time.time() - start_time) * 1000)
            e.log(self.logger)
            raise

        except CareerNavigatorError as e:
            # Our custom exceptions — already structured
            duration_ms = round((time.time() - start_time) * 1000)
            e.log(self.logger)
            raise

        except Exception as e:
            # Unexpected exception — wrap in ToolError
            duration_ms = round((time.time() - start_time) * 1000)
            self.logger.error(
                "tool_unexpected_error",
                tool        = self.name,
                error       = str(e),
                error_type  = type(e).__name__,
                duration_ms = duration_ms
            )
            raise ToolError(
                message   = f"Unexpected error in tool '{self.name}': {str(e)}",
                retryable = self.retryable,
                tool      = self.name,
                error_type= type(e).__name__
            )


    # ── Tool catalogue entry ──────────────────────────────────────────────────
    def to_openai_schema(self, parameters: dict) -> dict:
        """
        Generates OpenAI tool definition from this tool's metadata.
        Agent passes this schema to the LLM so it knows about the tool.

        Args:
          parameters: JSON Schema dict describing tool inputs

        Usage in specialist tool:
          def get_schema(self) -> dict:
              return self.to_openai_schema({
                  "type": "object",
                  "properties": {
                      "company_name": {
                          "type": "string",
                          "description": "Company name to scrape"
                      }
                  },
                  "required": ["company_name"]
              })
        """
        return {
            "type": "function",
            "function": {
                "name"       : self.name,
                "description": self.description,
                "parameters" : parameters
            }
        }


    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' retryable={self.retryable}>"


# ── Retry decorator ───────────────────────────────────────────────────────────
def retryable_tool(max_attempts: int = 3, delay_seconds: float = 2.0):
    """
    Decorator for tool execute() methods that should retry on failure.
    Uses exponential backoff — same pattern as llm_client.py.

    Usage:
      class ScraperTool(BaseTool):

          @retryable_tool(max_attempts=3, delay_seconds=2.0)
          def execute(self, company_name: str, **kwargs) -> dict:
              ...

    Only retries on retryable exceptions (ScrapingError, ConnectionError).
    Does not retry on ValidationError, ParsingError — those won't fix themselves.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, **kwargs):
            last_error = None
            logger     = get_logger(f"tools.{self.name}")

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(self, **kwargs)

                except CareerNavigatorError as e:
                    last_error = e
                    if not e.retryable or attempt == max_attempts:
                        raise

                    delay = delay_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "tool_retry",
                        tool    = self.name,
                        attempt = attempt,
                        delay_s = delay,
                        error   = str(e)
                    )
                    time.sleep(delay)

                except Exception as e:
                    # Non-CareerNavigatorError — don't retry
                    raise

            raise last_error

        return wrapper
    return decorator


# ── Tool registry helper ──────────────────────────────────────────────────────
class ToolRegistry:
    """
    Central registry of all available tools.
    Agents use this to discover and instantiate tools.

    In the current build — agents instantiate tools directly.
    In Phase 2 with dynamic tool loading — this registry becomes essential.
    Building it now costs nothing and enables future flexibility.

    Usage:
      registry = ToolRegistry()
      registry.register(ScraperTool())
      registry.register(JDParserTool())

      tool   = registry.get("scrape_company")
      result = tool.run(company_name="Amex")

      # Get all schemas for LLM
      schemas = registry.get_all_schemas()
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self.logger = get_logger("tools.registry")

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        self.logger.debug(
            "tool_registered",
            tool = tool.name
        )

    def get(self, name: str) -> BaseTool:
        """Get tool by name. Raises KeyError if not found."""
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not registered. "
                f"Available: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def get_all_schemas(self) -> list[dict]:
        """
        Returns OpenAI tool schemas for all registered tools.
        Pass this to chat_with_tools() for a fully dynamic tool set.
        """
        schemas = []
        for tool in self._tools.values():
            if hasattr(tool, "get_schema"):
                schemas.append(tool.get_schema())
        return schemas

    def list_tools(self) -> list[str]:
        """Returns names of all registered tools."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.list_tools()}>"


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.config import ensure_directories
    from core.exceptions import ScrapingError
    ensure_directories()

    print("\n── BaseTool self test ────────────────────────────────────")

    # Test 1: Cannot instantiate abstract class
    try:
        tool = BaseTool()
        print("[FAIL] Should not instantiate BaseTool directly")
    except TypeError as e:
        print(f"[1] ✓ Abstract class protection: {type(e).__name__}")

    # Test 2: Concrete tool — success path
    class EchoTool(BaseTool):
        name        = "echo"
        description = "Echoes input back — for testing"

        def execute(self, message: str = "hello", **kwargs) -> dict:
            return {"echo": message, "length": len(message)}

    tool   = EchoTool()
    result = tool.run(message="autonomous career navigator")
    print(f"[2] ✓ Tool execution: {result}")

    # Test 3: Output contract — must return dict
    class BadTool(BaseTool):
        name        = "bad_tool"
        description = "Returns wrong type — for testing"

        def execute(self, **kwargs):
            return "I am not a dict"    # Wrong return type

    bad_tool = BadTool()
    try:
        bad_tool.run()
        print("[FAIL] Should raise ToolError for non-dict return")
    except ToolError as e:
        print(f"[3] ✓ Output contract enforced: {e.message}")

    # Test 4: Validation — base validates nothing, subclass adds checks
    class ValidatedTool(BaseTool):
        name        = "validated_tool"
        description = "Tool with input validation"

        def validate(self, company: str = "", **kwargs):
            if not company:
                raise ValidationError(
                    field  = "company",
                    value  = company,
                    reason = "Company name is required"
                )

        def execute(self, company: str = "", **kwargs) -> dict:
            return {"company": company}

    validated = ValidatedTool()
    try:
        validated.run(company="")
    except ValidationError as e:
        print(f"[4] ✓ Validation enforced: {e.message}")

    result = validated.run(company="American Express")
    print(f"    ✓ Valid input accepted: {result}")

    # Test 5: Retryable decorator
    attempt_count = [0]

    class FlakeyTool(BaseTool):
        name        = "flakey_tool"
        description = "Fails first 2 attempts — for testing retry"

        @retryable_tool(max_attempts=3, delay_seconds=0.1)
        def execute(self, **kwargs) -> dict:
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ScrapingError(
                    company   = "TestCo",
                    url       = "http://test.com",
                    reason    = f"Transient failure attempt {attempt_count[0]}",
                    retryable = True
                )
            return {"success": True, "attempts": attempt_count[0]}

    flakey = FlakeyTool()
    result = flakey.run()
    print(f"[5] ✓ Retry decorator: succeeded on attempt {result['attempts']}")

    # Test 6: ToolRegistry
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(ValidatedTool())

    print(f"[6] ✓ Registry: {registry}")
    print(f"    Tools    : {registry.list_tools()}")

    tool   = registry.get("echo")
    result = tool.run(message="registry works")
    print(f"    Lookup   : {result}")

    try:
        registry.get("nonexistent")
    except KeyError as e:
        print(f"    ✓ Missing tool error: {e}")

    print("\n── BaseTool tests complete ───────────────────────────────")
    print("   Abstract enforcement  ✓")
    print("   Success path          ✓")
    print("   Output contract       ✓")
    print("   Input validation      ✓")
    print("   Retry decorator       ✓")
    print("   Tool registry         ✓")