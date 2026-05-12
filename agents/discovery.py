"""
agents/discovery.py

Discovery Agent — purely agentic, ReAct pattern.

The LLM is the brain. It decides:
  - Which companies to process
  - Which tools to call
  - In what order
  - When it is done

Python's only job:
  - Give the LLM its tools and goal
  - Execute whatever tool the LLM chooses
  - Feed the result back to the LLM
  - Repeat until LLM says done

No orchestration logic lives here.
No if/then/else sequences.
No hardcoded steps.

Adding new behaviour = update the system prompt or add a tool.
Never rewrite this loop.

This is the ReAct pattern:
  Reason → Act → Observe → Reason → Act → Observe → Done
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from openai import OpenAI

from config.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME,
    MAX_TOKENS
)


# ── LLM client ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)


# ── Tool definitions ──────────────────────────────────────────────────────────
# This is what the LLM sees — descriptions drive the agent's decisions.
# The better the description, the better the reasoning.
# No logic here — just contracts. What goes in, what comes out.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_companies",
            "description": (
                "Load the list of target companies from the config file. "
                "Returns a list of company configs including name, ATS type, "
                "locations, target titles, and exclude titles. "
                "Always call this first to know which companies to process."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_company",
            "description": (
                "Scrape job listings from a company's career page. "
                "Returns a list of raw job postings with title, location, "
                "url, and description. Call once per company."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Exact company name as returned by load_companies"
                    }
                },
                "required": ["company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_title_inflation",
            "description": (
                "Check if a job title is inflated — meaning the title sounds "
                "senior but the actual responsibilities suggest lower seniority. "
                "Use this on jobs that passed location and title filters "
                "before deciding to save them. "
                "Returns: is_inflated (bool) and reason (string)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The job title to evaluate"
                    },
                    "description": {
                        "type": "string",
                        "description": "The job description text, first 500 chars is enough"
                    }
                },
                "required": ["title", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_job",
            "description": (
                "Save a job to the store. Only call this for jobs that: "
                "1) match target locations, "
                "2) match target titles, "
                "3) are not already in the store. "
                "Returns: is_new (bool) — True if first time seeing this job."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title":       {"type": "string", "description": "Job title"},
                    "company":     {"type": "string", "description": "Company name"},
                    "location":    {"type": "string", "description": "Job location"},
                    "url":         {"type": "string", "description": "Job posting URL"},
                    "description": {"type": "string", "description": "Full job description"},
                    "is_inflated": {"type": "boolean", "description": "Whether title is inflated"},
                    "inflation_reason": {"type": "string", "description": "Reason if inflated"}
                },
                "required": ["title", "company", "location", "url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_summary",
            "description": (
                "Call this when you have finished processing all companies. "
                "Report what you found, what you saved, and any issues. "
                "This ends the discovery run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "companies_processed": {
                        "type": "integer",
                        "description": "Number of companies processed"
                    },
                    "jobs_found": {
                        "type": "integer",
                        "description": "Total jobs found before filtering"
                    },
                    "jobs_saved": {
                        "type": "integer",
                        "description": "New jobs saved after all filters"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Plain English summary of what happened"
                    }
                },
                "required": ["companies_processed", "jobs_found", "jobs_saved", "summary"]
            }
        }
    }
]


# ── Tool implementations ───────────────────────────────────────────────────────
# Pure functions. Execute and return. No decisions.
# LLM decided to call these — Python just runs them.

def _tool_load_companies() -> dict:
    """Loads companies from YAML and returns as dict for LLM to reason about."""
    import yaml
    from config.config import CONFIG_DIR

    config_path = CONFIG_DIR / "companies.yaml"
    if not config_path.exists():
        return {"error": f"companies.yaml not found at {config_path}"}

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    companies = data.get("companies", [])
    print(f"[TOOL] load_companies → {len(companies)} companies")
    return {"companies": companies}


def _tool_scrape_company(company_name: str) -> dict:
    """Delegates to scraper tool. Returns raw job list."""
    from tools.scraper import scrape_company

    # Find company config from loaded list
    import yaml
    from config.config import CONFIG_DIR
    with open(CONFIG_DIR / "companies.yaml") as f:
        data = yaml.safe_load(f)

    company_config = next(
        (c for c in data.get("companies", []) if c["name"] == company_name),
        None
    )

    if not company_config:
        return {"error": f"Company '{company_name}' not found in config"}

    jobs = scrape_company(company_config)
    print(f"[TOOL] scrape_company({company_name}) → {len(jobs)} jobs")
    return {"jobs": jobs, "count": len(jobs)}


def _tool_check_title_inflation(title: str, description: str) -> dict:
    """
    Pure function — calls a simple heuristic check.
    No nested LLM call here to avoid complexity at this stage.
    We check for known inflation patterns deterministically.
    Can be upgraded to LLM call later by changing only this function.
    """
    inflation_patterns = [
        ("director", ["manage a small team", "individual contributor",
                      "reports to manager", "2-3 years"]),
        ("vp", ["manage a team of", "entry level", "0-3 years",
                "fresh graduate"]),
        ("head of", ["coordinate with", "support the team", "assist"])
    ]

    title_lower       = title.lower()
    description_lower = description.lower() if description else ""
    is_inflated       = False
    reason            = "No inflation detected"

    for senior_title, junior_signals in inflation_patterns:
        if senior_title in title_lower:
            for signal in junior_signals:
                if signal in description_lower:
                    is_inflated = True
                    reason = f"Title '{title}' but JD contains '{signal}'"
                    break

    print(f"[TOOL] check_title_inflation({title}) → inflated={is_inflated}")
    return {"is_inflated": is_inflated, "reason": reason}


def _tool_save_job(title: str, company: str, location: str, url: str,
                   description: str = "", is_inflated: bool = False,
                   inflation_reason: str = "") -> dict:
    """Saves job to store. Returns whether it was new."""
    import hashlib
    from memory.store import save_job, update_job_fitment

    # Generate stable ID
    raw    = f"{company}_{title}_{url}".lower()
    job_id = hashlib.md5(raw.encode()).hexdigest()[:16]

    job_record = {
        "job_id"     : job_id,
        "title"      : title,
        "company"    : company,
        "location"   : location,
        "url"        : url,
        "description": description
    }

    is_new = save_job(job_record)

    if is_new and is_inflated:
        update_job_fitment(
            job_id      = job_id,
            score       = 0,
            summary     = f"Title inflation: {inflation_reason}",
            is_inflated = True
        )

    print(f"[TOOL] save_job({title} @ {company}) → new={is_new}")
    return {"is_new": is_new, "job_id": job_id}


def _tool_report_summary(companies_processed: int, jobs_found: int,
                         jobs_saved: int, summary: str) -> dict:
    """Records final summary and signals the loop to end."""
    print(f"[TOOL] report_summary → saved={jobs_saved}")
    return {
        "status"              : "completed",
        "companies_processed" : companies_processed,
        "jobs_found"          : jobs_found,
        "jobs_saved"          : jobs_saved,
        "summary"             : summary
    }


# ── Tool router ───────────────────────────────────────────────────────────────
# Maps tool name → implementation function.
# LLM picks a name. Python looks it up here and runs it.
# Adding a new tool = add to TOOLS list above + add entry here.
# The agent loop never changes.

TOOL_REGISTRY = {
    "load_companies"       : _tool_load_companies,
    "scrape_company"       : _tool_scrape_company,
    "check_title_inflation": _tool_check_title_inflation,
    "save_job"             : _tool_save_job,
    "report_summary"       : _tool_report_summary
}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Executes the tool the LLM chose.
    Returns result as JSON string — LLM reads JSON.
    This is the only place Python acts. Everything else is reasoning.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = TOOL_REGISTRY[tool_name](**tool_args)
        return json.dumps(result)
    except Exception as e:
        error = {"error": str(e), "tool": tool_name, "args": tool_args}
        print(f"[TOOL ERROR] {tool_name}: {e}")
        return json.dumps(error)


# ── System prompt ─────────────────────────────────────────────────────────────
# This is where the agent's intelligence lives.
# The goal, the strategy, the judgment criteria — all in natural language.
# Change behaviour by editing this prompt. No code change needed.

SYSTEM_PROMPT = """You are an autonomous job discovery agent.

Your goal: find new relevant job openings for a senior technology leader 
(VP Technology level) and save them to the store.

Your tools:
- load_companies     → get the list of target companies
- scrape_company     → get raw job listings from a company
- check_title_inflation → verify a job title is genuine seniority
- save_job           → persist a job that passes all your checks
- report_summary     → call when done with all companies

Your decision process for each job:
1. Does the location match the user's target locations?
2. Does the title suggest VP / Director / Head of level seniority?
3. Is the title genuine — not inflated? Use check_title_inflation if unsure.
4. Is this job worth saving? If yes — call save_job.

Your judgment guidelines:
- Be selective. Quality over quantity.
- A "Director" role with junior responsibilities is not relevant — flag it.
- Locations like "India" or "Remote" are acceptable if target locations match.
- If scraping fails for a company — note it and move to the next one.
- Process all companies before calling report_summary.

Think step by step. Use tools deliberately. 
When all companies are processed — call report_summary and stop."""


# ── ReAct agent loop ──────────────────────────────────────────────────────────
def run_discovery(user_id: str = "local_user") -> dict:
    """
    The purely agentic discovery loop.

    Structure:
    1. Give LLM the system prompt (goal + guidelines)
    2. LLM reasons and picks a tool
    3. Python executes the tool
    4. Result goes back to LLM as observation
    5. LLM reasons again
    6. Repeat until LLM calls report_summary

    The LLM drives. Python executes. No logic anywhere else.
    """

    print(f"\n[DISCOVERY] Agent starting — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Conversation history — grows with each Reason→Act→Observe cycle
    messages = [
        {
            "role"   : "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role"   : "user",
            "content": f"Start the job discovery run. Today is {datetime.now().strftime('%Y-%m-%d')}. Find new jobs and save the relevant ones."
        }
    ]

    final_summary = {}
    max_iterations = 50      # Safety cap — prevents infinite loops
    iteration      = 0

    # ── The ReAct loop ────────────────────────────────────────────────────────
    while iteration < max_iterations:
        iteration += 1
        print(f"\n[DISCOVERY] ── Iteration {iteration} ──────────────────")

        # ── Reason: LLM decides what to do next ──────────────────────────────
        response = client.chat.completions.create(
            model     = MODEL_NAME,
            max_tokens= MAX_TOKENS,
            tools     = TOOLS,
            messages  = messages
        )

        message = response.choices[0].message

        # Add LLM response to conversation history
        messages.append({
            "role"      : "assistant",
            "content"   : message.content,
            "tool_calls": [
                {
                    "id"      : tc.id,
                    "type"    : "function",
                    "function": {
                        "name"     : tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in (message.tool_calls or [])
            ] or None
        })

        # ── Check: did LLM decide it is done? ────────────────────────────────
        if not message.tool_calls:
            # LLM responded with text, not a tool call
            # This means it is done reasoning
            print(f"[DISCOVERY] Agent finished reasoning: {message.content}")
            break

        # ── Act + Observe: execute each tool the LLM chose ───────────────────
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"[DISCOVERY] Agent calls: {tool_name}({tool_args})")

            # Execute the tool
            tool_result = execute_tool(tool_name, tool_args)

            print(f"[DISCOVERY] Tool result: {tool_result[:120]}...")

            # Feed observation back to LLM
            messages.append({
                "role"        : "tool",
                "tool_call_id": tool_call.id,
                "content"     : tool_result
            })

            # If LLM called report_summary — capture it and end the loop
            if tool_name == "report_summary":
                final_summary = json.loads(tool_result)
                print(f"\n[DISCOVERY] Agent complete")
                print(f"            Companies : {final_summary.get('companies_processed')}")
                print(f"            Found     : {final_summary.get('jobs_found')}")
                print(f"            Saved     : {final_summary.get('jobs_saved')}")
                print(f"            Summary   : {final_summary.get('summary')}")
                return final_summary

    # Safety exit — max iterations reached
    print(f"[DISCOVERY] Max iterations ({max_iterations}) reached")
    return {
        "status"   : "max_iterations_reached",
        "iteration": iteration,
        "summary"  : "Agent hit safety cap — check logs"
    }


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.config import ensure_directories
    from memory.store import initialise_db

    ensure_directories()
    initialise_db()

    print("=" * 60)
    print("  Autonomous Discovery Agent — ReAct Pattern")
    print("  LLM drives. Python executes. No hardcoded logic.")
    print("=" * 60)

    result = run_discovery()

    print("\n── Final Result ───────────────────────────────────────────")
    print(json.dumps(result, indent=2))