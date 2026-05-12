"""
agents/discovery.py

Discovery Agent — first specialist agent in the system.
Responsibility: find new jobs from target companies and save to store.

Agentic loop this agent runs:
  1. Read target companies from companies.yaml
  2. For each company — call the right scraper tool based on ats_type
  3. Filter results by location and title keywords
  4. Check store — is this job already seen?
  5. If new — save to store, flag for fitment evaluation
  6. Report summary back to orchestrator

Design principles:
  - Agent knows WHAT to do, tools know HOW to do it
  - Agent never writes HTML parsers — it calls tools
  - Agent never writes to DB directly — it calls store
  - Every decision logged — orchestrator can inspect
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import hashlib
from datetime import datetime
from pathlib import Path
from openai import OpenAI                    # OpenAI-compatible client works with OpenRouter

from config.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME,
    MAX_TOKENS,
    CONFIG_DIR
)
from memory.store import save_job, get_job_by_id
from tools.scraper import scrape_company


# ── LLM client ────────────────────────────────────────────────────────────────
# OpenRouter is OpenAI-compatible — same SDK, different base_url
# Switching to Anthropic direct = change base_url and api_key only
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)


# ── Load company config ───────────────────────────────────────────────────────
def load_companies() -> list:
    """
    Reads companies.yaml and returns list of company configs.
    Agent reads config — never hardcodes company names.
    Adding a new company = edit YAML only, no code change.
    """
    config_path = CONFIG_DIR / "companies.yaml"

    if not config_path.exists():
        print(f"[DISCOVERY] companies.yaml not found at {config_path}")
        return []

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    companies = data.get("companies", [])
    print(f"[DISCOVERY] Loaded {len(companies)} companies from config")
    return companies


# ── Title filter ──────────────────────────────────────────────────────────────
def passes_title_filter(job_title: str, company_config: dict) -> tuple[bool, str]:
    """
    Checks if a job title matches target and exclude lists.
    Returns (passes: bool, reason: str)

    This is deterministic logic — no LLM needed here.
    LLM is expensive. Use it only where judgment is needed.
    Simple string matching is free and fast.
    """
    title_lower = job_title.lower()

    # Check exclude list first — fast exit
    for exclude in company_config.get("exclude_titles", []):
        if exclude.lower() in title_lower:
            return False, f"Excluded — contains '{exclude}'"

    # Check target list — must match at least one
    for target in company_config.get("target_titles", []):
        if target.lower() in title_lower:
            return True, f"Matched — contains '{target}'"

    return False, "No target title match"


# ── Location filter ───────────────────────────────────────────────────────────
def passes_location_filter(job_location: str, company_config: dict) -> bool:
    """
    Checks if job location matches target locations.
    Case-insensitive. Partial match allowed — "Chennai, India" matches "Chennai"
    """
    if not job_location:
        return True                          # Unknown location — let it through

    location_lower = job_location.lower()
    target_locations = company_config.get("locations", [])

    if not target_locations:
        return True                          # No filter set — let everything through

    return any(loc.lower() in location_lower for loc in target_locations)


# ── Title inflation check via LLM ─────────────────────────────────────────────
def check_title_inflation(job_title: str, job_description: str) -> tuple[bool, str]:
    """
    Uses LLM to detect title inflation.
    Example: "Director of Engineering" JD that is actually a senior manager role.

    This IS where we use the LLM — pattern matching can't catch this.
    The LLM reads the JD and judges if the title matches the actual seniority.

    Returns (is_inflated: bool, reason: str)
    """
    prompt = f"""You are evaluating a job posting for title inflation.
Title inflation means the job title sounds senior but the actual 
responsibilities and requirements suggest a lower seniority level.

Job Title: {job_title}

Job Description (first 500 chars):
{job_description[:500] if job_description else 'Not available'}

Evaluate:
1. Does the title match the actual seniority in the description?
2. Is this a genuine VP/Director level role or is it inflated?

Respond in JSON only — no other text:
{{
  "is_inflated": true or false,
  "confidence": "high" or "medium" or "low",
  "reason": "one sentence explanation"
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior HR expert who detects title inflation in job postings. Respond only in valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Clean JSON — sometimes LLMs add markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        return result.get("is_inflated", False), result.get("reason", "")

    except Exception as e:
        # If LLM call fails — don't block the job from being saved
        # Fail safe: assume not inflated, log the error
        print(f"[DISCOVERY] Title inflation check failed: {e}")
        return False, "Check failed — defaulting to not inflated"


# ── Generate stable job ID ────────────────────────────────────────────────────
def generate_job_id(company: str, title: str, url: str) -> str:
    """
    Creates a stable unique ID for each job.
    MD5 hash of company + title + url — same job always gets same ID.
    This is how INSERT OR IGNORE in store.py prevents duplicates.
    """
    raw = f"{company}_{title}_{url}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Main discovery function ───────────────────────────────────────────────────
def run_discovery(user_id: str = "local_user") -> dict:
    """
    Main entry point — orchestrator calls this.
    Returns a summary dict the orchestrator uses to decide next steps.

    The agentic loop:
    for each company:
        scrape → filter → check duplicate → check inflation → save
    """
    print(f"\n[DISCOVERY] Starting job discovery — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    companies   = load_companies()
    total_found = 0
    total_new   = 0
    total_saved = 0
    results     = []

    for company in companies:
        company_name = company.get("name")
        ats_type     = company.get("ats_type", "generic")
        priority     = company.get("priority", "medium")

        print(f"\n[DISCOVERY] Processing: {company_name} (priority={priority})")

        # ── Step 1: Scrape jobs via tool ──────────────────────────────────────
        # Agent delegates to tool — doesn't know HOW scraping works
        raw_jobs = scrape_company(company)

        if not raw_jobs:
            print(f"[DISCOVERY] No jobs returned for {company_name}")
            results.append({
                "company": company_name,
                "status" : "no_jobs_found",
                "found"  : 0,
                "new"    : 0
            })
            continue

        print(f"[DISCOVERY] {len(raw_jobs)} raw jobs fetched from {company_name}")
        total_found += len(raw_jobs)

        company_new = 0

        for job in raw_jobs:
            title    = job.get("title", "")
            location = job.get("location", "")
            url      = job.get("url", "")

            # ── Step 2: Location filter ───────────────────────────────────────
            if not passes_location_filter(location, company):
                continue

            # ── Step 3: Title filter ──────────────────────────────────────────
            passes, reason = passes_title_filter(title, company)
            if not passes:
                continue

            print(f"[DISCOVERY] ✓ Passed filters: {title} | {location} | {reason}")

            # ── Step 4: Generate stable ID ────────────────────────────────────
            job_id = generate_job_id(company_name, title, url)

            # ── Step 5: Duplicate check ───────────────────────────────────────
            existing = get_job_by_id(job_id)
            if existing:
                print(f"[DISCOVERY] Already seen: {title} — skipping")
                continue

            # ── Step 6: Title inflation check via LLM ─────────────────────────
            # Only run LLM on jobs that passed all filters
            # Saves OpenRouter credits — don't waste LLM on filtered-out jobs
            description  = job.get("description", "")
            is_inflated, inflation_reason = check_title_inflation(title, description)

            if is_inflated:
                print(f"[DISCOVERY] ⚠ Title inflation detected: {title} — {inflation_reason}")

            # ── Step 7: Save to store ─────────────────────────────────────────
            job_record = {
                "job_id"     : job_id,
                "title"      : title,
                "company"    : company_name,
                "location"   : location,
                "url"        : url,
                "description": description,
                "salary_min" : job.get("salary_min"),
                "salary_max" : job.get("salary_max"),
            }

            is_new = save_job(job_record, user_id=user_id)

            if is_new:
                company_new += 1
                total_new   += 1
                total_saved += 1
                print(f"[DISCOVERY] ✓ Saved new job: {title} at {company_name}")

                # If inflated — update the flag in store
                if is_inflated:
                    from memory.store import update_job_fitment
                    update_job_fitment(
                        job_id     = job_id,
                        score      = 0,
                        summary    = f"Title inflation detected: {inflation_reason}",
                        is_inflated= True,
                        user_id    = user_id
                    )

        results.append({
            "company": company_name,
            "status" : "completed",
            "found"  : len(raw_jobs),
            "new"    : company_new
        })

    # ── Summary for orchestrator ──────────────────────────────────────────────
    summary = {
        "run_at"      : datetime.now().isoformat(),
        "companies"   : len(companies),
        "total_found" : total_found,
        "total_new"   : total_new,
        "total_saved" : total_saved,
        "results"     : results,
        "status"      : "completed"
    }

    print(f"\n[DISCOVERY] Complete — {total_new} new jobs found across {len(companies)} companies")
    return summary


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.config import ensure_directories
    from memory.store import initialise_db

    ensure_directories()
    initialise_db()

    summary = run_discovery()

    print("\n── Discovery Summary ──────────────────────────────")
    print(f"Companies scanned : {summary['companies']}")
    print(f"Total jobs found  : {summary['total_found']}")
    print(f"New jobs saved    : {summary['total_new']}")
    print(f"Status            : {summary['status']}")
    print("\nPer company:")
    for r in summary["results"]:
        print(f"  {r['company']:30} found={r['found']:3}  new={r['new']:3}  status={r['status']}")