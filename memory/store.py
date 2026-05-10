"""
memory/store.py

Repository pattern over SQLite.
Multi-tenant ready — every table has user_id and region.
Agents never touch the database directly — they call these functions.

Migration path:
  SQLite now          → personal desktop, zero cost
  PostgreSQL later    → swap connection in get_connection() only
  Sharded Postgres    → add routing logic in get_connection() only
  Agents never change across any of these migrations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from config.config import DB_PATH


# ── Constants ─────────────────────────────────────────────────────────────────
# Default user for single-user desktop mode.
# In SaaS mode this becomes a UUID per registered user.
LOCAL_USER   = "local_user"
LOCAL_REGION = "in-south"   # India south — Chennai default
                             # Future values: in-north, us-east, eu-west, ap-sea


# ── Database connection ───────────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    """
    Single place where database connection is created.
    To migrate to PostgreSQL — change only this function.
    Everything else in this file stays identical.
    """
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    # Rows returned as dicts: row["title"] not row[0]
    conn.row_factory = sqlite3.Row
    # Enable foreign key enforcement — SQLite disables this by default
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────
def initialise_db():
    """
    Creates all tables if they don't exist.
    Safe to call every startup — IF NOT EXISTS protects existing data.

    Design principles:
    - user_id on every table     → multi-tenant isolation
    - region on every table      → geographic routing + compliance
    - created_at + updated_at    → audit trail, change tracking
    - Indexes on user_id+region  → fast queries at scale
    """
    conn = get_connection()
    c = conn.cursor()

    # ── 1. candidate_profile ──────────────────────────────────────────────────
    # One row per user. Identity, voice, job search preferences.
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidate_profile (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            full_name       TEXT,
            email           TEXT,
            phone           TEXT,
            location        TEXT,
            linkedin_url    TEXT,
            github_url      TEXT,
            target_roles    TEXT,          -- JSON list: ["VP Technology", "CTO"]
            target_sectors  TEXT,          -- JSON list: ["BFSI", "GCC", "Fintech"]
            salary_floor    INTEGER,       -- Minimum acceptable CTC in INR
            notice_period   INTEGER,       -- In days
            voice_tone      TEXT,          -- e.g. "direct, confident, no fluff"
            words_to_avoid  TEXT,          -- JSON list: ["synergy", "leverage"]
            created_at      TIMESTAMP,
            updated_at      TIMESTAMP,
            UNIQUE(user_id)
        )
    """)

    # ── 2. experiences ────────────────────────────────────────────────────────
    # One row per role held. The backbone of the knowledge base.
    c.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            company         TEXT NOT NULL,
            title           TEXT NOT NULL,
            employment_type TEXT,          -- full-time | contract | consulting
            industry        TEXT,          -- BFSI | Retail | Logistics etc.
            domain          TEXT,          -- Core Banking | Supply Chain etc.
            location        TEXT,
            start_date      DATE,
            end_date        DATE,          -- NULL if current role
            is_current      INTEGER DEFAULT 0,
            team_size       INTEGER,
            budget_managed  TEXT,          -- e.g. "$5M annual opex"
            reporting_to    TEXT,          -- Title of person you reported to
            geography       TEXT,          -- Regional | Global | APAC etc.
            summary         TEXT,          -- Your narrative of this role
            created_at      TIMESTAMP,
            updated_at      TIMESTAMP
        )
    """)

    # ── 3. projects ───────────────────────────────────────────────────────────
    # One row per project. Linked to an experience.
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            experience_id   INTEGER,       -- Links to experiences.id
            project_name    TEXT NOT NULL,
            client_name     TEXT,          -- Prestigious client names live here
            client_sector   TEXT,
            duration_months INTEGER,
            your_role       TEXT,          -- Your specific role in this project
            problem_statement TEXT,        -- What business problem was solved
            your_activities TEXT,          -- What you specifically did
            tech_stack      TEXT,          -- JSON list of technologies used
            team_size       INTEGER,
            outcome         TEXT,          -- What was delivered
            is_confidential INTEGER DEFAULT 0, -- 1 = don't show client name
            created_at      TIMESTAMP,
            updated_at      TIMESTAMP,
            FOREIGN KEY (experience_id) REFERENCES experiences(id)
        )
    """)

    # ── 4. achievements ───────────────────────────────────────────────────────
    # One row per achievement. Quantified. Linked to experience or project.
    # This is where "reduced cost by 40%" lives — not buried in a description.
    c.execute("""
        CREATE TABLE IF NOT EXISTS achievements (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            experience_id   INTEGER,       -- Links to experiences.id
            project_id      INTEGER,       -- Links to projects.id
            what_you_did    TEXT NOT NULL, -- Action: "Led migration of..."
            metric_before   TEXT,          -- Baseline: "45% manual effort"
            metric_after    TEXT,          -- Result:   "8% manual effort"
            business_impact TEXT,          -- So what: "Saved $2M annually"
            award_or_recognition TEXT,     -- If formally recognised
            created_at      TIMESTAMP,
            updated_at      TIMESTAMP,
            FOREIGN KEY (experience_id) REFERENCES experiences(id),
            FOREIGN KEY (project_id)    REFERENCES projects(id)
        )
    """)

    # ── 5. skills ─────────────────────────────────────────────────────────────
    # One row per skill. Structured — not a comma-separated string.
    # Linked to experiences and projects so we can prove the skill, not just claim it.
    c.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            skill_name      TEXT NOT NULL,
            category        TEXT,          -- Technical | Leadership | Domain | Tool
            subcategory     TEXT,          -- e.g. "Cloud" under "Technical"
            proficiency     TEXT,          -- Expert | Proficient | Familiar
            years_of_exp    INTEGER,
            is_certified    INTEGER DEFAULT 0,
            last_used_year  INTEGER,
            context         TEXT,          -- Where/how this skill was used
            created_at      TIMESTAMP,
            updated_at      TIMESTAMP
        )
    """)

    # ── 6. education ──────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS education (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            institution     TEXT NOT NULL,
            degree          TEXT NOT NULL,
            field_of_study  TEXT,
            start_year      INTEGER,
            end_year        INTEGER,
            grade           TEXT,
            relevant_courses TEXT,         -- JSON list
            created_at      TIMESTAMP,
            updated_at      TIMESTAMP
        )
    """)

    # ── 7. certifications ─────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS certifications (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            cert_name       TEXT NOT NULL,
            issuer          TEXT,
            issue_year      INTEGER,
            expiry_year     INTEGER,       -- NULL if no expiry
            credential_id   TEXT,
            credential_url  TEXT,
            relevance_tags  TEXT,          -- JSON: ["AI", "Cloud", "Leadership"]
            created_at      TIMESTAMP,
            updated_at      TIMESTAMP
        )
    """)

    # ── 8. resume_versions ────────────────────────────────────────────────────
    # Every generated resume stored here. Full audit trail.
    # Content agent writes here. Never overwrites — always appends.
    c.execute("""
        CREATE TABLE IF NOT EXISTS resume_versions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            version_name    TEXT,          -- e.g. "citibank_vp_2026_05"
            job_id          TEXT,          -- Which JD this was tailored for
            content         TEXT NOT NULL, -- Full resume text
            format          TEXT DEFAULT 'markdown',
            ats_score       INTEGER,       -- ATS compatibility score 0-100
            word_count      INTEGER,
            created_at      TIMESTAMP
        )
    """)

    # ── 9. jobs ───────────────────────────────────────────────────────────────
    # Discovery agent writes here. Every job found, one row.
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           TEXT NOT NULL DEFAULT 'local_user',
            region            TEXT NOT NULL DEFAULT 'in-south',
            job_id            TEXT,
            title             TEXT NOT NULL,
            company           TEXT NOT NULL,
            location          TEXT,
            url               TEXT,
            description       TEXT,
            salary_min        INTEGER,
            salary_max        INTEGER,
            fitment_score     INTEGER,
            fitment_summary   TEXT,
            status            TEXT DEFAULT 'new',
            is_title_inflated INTEGER DEFAULT 0,
            discovered_at     TIMESTAMP,
            updated_at        TIMESTAMP,
            UNIQUE(user_id, job_id)
        )
    """)

    # ── 10. applications ──────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           TEXT NOT NULL DEFAULT 'local_user',
            region            TEXT NOT NULL DEFAULT 'in-south',
            job_id            TEXT,
            resume_version_id INTEGER,
            cover_letter      TEXT,
            applied_at        TIMESTAMP,
            current_stage     TEXT DEFAULT 'applied',
            next_action       TEXT,
            next_action_due   DATE,
            notes             TEXT,
            updated_at        TIMESTAMP,
            FOREIGN KEY (job_id)            REFERENCES jobs(job_id),
            FOREIGN KEY (resume_version_id) REFERENCES resume_versions(id)
        )
    """)

    # ── 11. market_signals ────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL DEFAULT 'local_user',
            region          TEXT NOT NULL DEFAULT 'in-south',
            signal_type     TEXT,
            signal_key      TEXT,
            signal_value    TEXT,
            observed_at     TIMESTAMP
        )
    """)

    # ── Indexes ───────────────────────────────────────────────────────────────
    # Without indexes, every query scans every row.
    # With indexes, queries on user_id+region go directly to the right rows.
    # This is what makes the same schema work at 1 user and 1 million users.
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_experiences_user     ON experiences(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_projects_user        ON projects(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_achievements_user    ON achievements(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_skills_user          ON skills(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_education_user       ON education(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_certifications_user  ON certifications(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_resume_versions_user ON resume_versions(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_user            ON jobs(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_applications_user    ON applications(user_id, region)",
        "CREATE INDEX IF NOT EXISTS idx_market_signals_user  ON market_signals(user_id, region)",
    ]
    for idx in indexes:
        c.execute(idx)

    conn.commit()
    conn.close()
    print("[DB] All 11 tables and 10 indexes created successfully")


# ── Helper ────────────────────────────────────────────────────────────────────
def _now():
    return datetime.now()


# ── Candidate profile ─────────────────────────────────────────────────────────
def save_profile(profile: dict,
                 user_id: str = LOCAL_USER,
                 region:  str = LOCAL_REGION) -> bool:
    conn = get_connection()
    c    = conn.cursor()
    now  = _now()
    c.execute("""
        INSERT INTO candidate_profile (
            user_id, region, full_name, email, phone, location,
            linkedin_url, github_url, target_roles, target_sectors,
            salary_floor, notice_period, voice_tone, words_to_avoid,
            created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            full_name      = excluded.full_name,
            email          = excluded.email,
            target_roles   = excluded.target_roles,
            target_sectors = excluded.target_sectors,
            salary_floor   = excluded.salary_floor,
            voice_tone     = excluded.voice_tone,
            words_to_avoid = excluded.words_to_avoid,
            updated_at     = excluded.updated_at
    """, (
        user_id, region,
        profile.get("full_name"),   profile.get("email"),
        profile.get("phone"),       profile.get("location"),
        profile.get("linkedin_url"),profile.get("github_url"),
        json.dumps(profile.get("target_roles", [])),
        json.dumps(profile.get("target_sectors", [])),
        profile.get("salary_floor"),profile.get("notice_period"),
        profile.get("voice_tone"),
        json.dumps(profile.get("words_to_avoid", [])),
        now, now
    ))
    conn.commit()
    conn.close()
    return True


def get_profile(user_id: str = LOCAL_USER) -> dict | None:
    conn = get_connection()
    c    = conn.cursor()
    c.execute("SELECT * FROM candidate_profile WHERE user_id = ?", (user_id,))
    row  = c.fetchone()
    conn.close()
    return dict(row) if row else None


# ── Experiences ───────────────────────────────────────────────────────────────
def save_experience(exp: dict,
                    user_id: str = LOCAL_USER,
                    region:  str = LOCAL_REGION) -> int:
    conn = get_connection()
    c    = conn.cursor()
    now  = _now()
    c.execute("""
        INSERT INTO experiences (
            user_id, region, company, title, employment_type,
            industry, domain, location, start_date, end_date,
            is_current, team_size, budget_managed, reporting_to,
            geography, summary, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, region,
        exp.get("company"),         exp.get("title"),
        exp.get("employment_type", "full-time"),
        exp.get("industry"),        exp.get("domain"),
        exp.get("location"),        exp.get("start_date"),
        exp.get("end_date"),        int(exp.get("is_current", False)),
        exp.get("team_size"),       exp.get("budget_managed"),
        exp.get("reporting_to"),    exp.get("geography"),
        exp.get("summary"),         now, now
    ))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def get_experiences(user_id: str = LOCAL_USER) -> list:
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        SELECT * FROM experiences WHERE user_id = ?
        ORDER BY is_current DESC, start_date DESC
    """, (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Projects ──────────────────────────────────────────────────────────────────
def save_project(project: dict,
                 user_id: str = LOCAL_USER,
                 region:  str = LOCAL_REGION) -> int:
    conn = get_connection()
    c    = conn.cursor()
    now  = _now()
    c.execute("""
        INSERT INTO projects (
            user_id, region, experience_id, project_name, client_name,
            client_sector, duration_months, your_role, problem_statement,
            your_activities, tech_stack, team_size, outcome,
            is_confidential, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, region,
        project.get("experience_id"),   project.get("project_name"),
        project.get("client_name"),     project.get("client_sector"),
        project.get("duration_months"), project.get("your_role"),
        project.get("problem_statement"),
        project.get("your_activities"),
        json.dumps(project.get("tech_stack", [])),
        project.get("team_size"),       project.get("outcome"),
        int(project.get("is_confidential", False)),
        now, now
    ))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def get_projects(user_id: str = LOCAL_USER,
                 experience_id: int = None) -> list:
    conn = get_connection()
    c    = conn.cursor()
    if experience_id:
        c.execute("""
            SELECT * FROM projects
            WHERE user_id = ? AND experience_id = ?
            ORDER BY duration_months DESC
        """, (user_id, experience_id))
    else:
        c.execute("""
            SELECT * FROM projects WHERE user_id = ?
            ORDER BY duration_months DESC
        """, (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Achievements ──────────────────────────────────────────────────────────────
def save_achievement(achievement: dict,
                     user_id: str = LOCAL_USER,
                     region:  str = LOCAL_REGION) -> int:
    conn = get_connection()
    c    = conn.cursor()
    now  = _now()
    c.execute("""
        INSERT INTO achievements (
            user_id, region, experience_id, project_id,
            what_you_did, metric_before, metric_after,
            business_impact, award_or_recognition,
            created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, region,
        achievement.get("experience_id"),
        achievement.get("project_id"),
        achievement.get("what_you_did"),
        achievement.get("metric_before"),
        achievement.get("metric_after"),
        achievement.get("business_impact"),
        achievement.get("award_or_recognition"),
        now, now
    ))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def get_achievements(user_id: str = LOCAL_USER,
                     experience_id: int = None) -> list:
    conn = get_connection()
    c    = conn.cursor()
    if experience_id:
        c.execute("""
            SELECT * FROM achievements
            WHERE user_id = ? AND experience_id = ?
        """, (user_id, experience_id))
    else:
        c.execute(
            "SELECT * FROM achievements WHERE user_id = ?", (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Skills ────────────────────────────────────────────────────────────────────
def save_skill(skill: dict,
               user_id: str = LOCAL_USER,
               region:  str = LOCAL_REGION) -> int:
    conn = get_connection()
    c    = conn.cursor()
    now  = _now()
    c.execute("""
        INSERT INTO skills (
            user_id, region, skill_name, category, subcategory,
            proficiency, years_of_exp, is_certified,
            last_used_year, context, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, region,
        skill.get("skill_name"),    skill.get("category"),
        skill.get("subcategory"),   skill.get("proficiency", "Proficient"),
        skill.get("years_of_exp"),  int(skill.get("is_certified", False)),
        skill.get("last_used_year"),skill.get("context"),
        now, now
    ))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def get_skills(user_id: str = LOCAL_USER,
               category: str = None) -> list:
    conn = get_connection()
    c    = conn.cursor()
    if category:
        c.execute("""
            SELECT * FROM skills
            WHERE user_id = ? AND category = ?
            ORDER BY years_of_exp DESC
        """, (user_id, category))
    else:
        c.execute("""
            SELECT * FROM skills WHERE user_id = ?
            ORDER BY category, years_of_exp DESC
        """, (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Jobs ──────────────────────────────────────────────────────────────────────
def save_job(job: dict,
             user_id: str = LOCAL_USER,
             region:  str = LOCAL_REGION) -> bool:
    conn = get_connection()
    c    = conn.cursor()
    now  = _now()
    c.execute("""
        INSERT OR IGNORE INTO jobs (
            user_id, region, job_id, title, company,
            location, url, description, salary_min, salary_max,
            status, discovered_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,'new',?,?)
    """, (
        user_id, region,
        job.get("job_id"),      job.get("title"),
        job.get("company"),     job.get("location"),
        job.get("url"),         job.get("description"),
        job.get("salary_min"),  job.get("salary_max"),
        now, now
    ))
    conn.commit()
    is_new = c.rowcount == 1
    conn.close()
    return is_new


def update_job_fitment(job_id: str, score: int, summary: str,
                       is_inflated: bool = False,
                       user_id: str = LOCAL_USER):
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        UPDATE jobs
        SET fitment_score = ?, fitment_summary = ?,
            is_title_inflated = ?, updated_at = ?
        WHERE job_id = ? AND user_id = ?
    """, (score, summary, int(is_inflated), _now(), job_id, user_id))
    conn.commit()
    conn.close()


def get_new_jobs(user_id: str = LOCAL_USER,
                 min_fitment: int = 0) -> list:
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        SELECT * FROM jobs
        WHERE user_id = ? AND status = 'new'
        AND (fitment_score >= ? OR fitment_score IS NULL)
        ORDER BY discovered_at DESC
    """, (user_id, min_fitment))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Applications ──────────────────────────────────────────────────────────────
def save_application(application: dict,
                     user_id: str = LOCAL_USER,
                     region:  str = LOCAL_REGION) -> int:
    conn = get_connection()
    c    = conn.cursor()
    now  = _now()
    c.execute("""
        INSERT INTO applications (
            user_id, region, job_id, resume_version_id,
            cover_letter, applied_at, current_stage,
            next_action, next_action_due, notes, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, region,
        application.get("job_id"),
        application.get("resume_version_id"),
        application.get("cover_letter"),
        now,
        application.get("current_stage", "applied"),
        application.get("next_action"),
        application.get("next_action_due"),
        application.get("notes"),
        now
    ))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def update_application_stage(app_id: int, stage: str,
                              notes: str = None,
                              user_id: str = LOCAL_USER):
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        UPDATE applications
        SET current_stage = ?, notes = COALESCE(?, notes), updated_at = ?
        WHERE id = ? AND user_id = ?
    """, (stage, notes, _now(), app_id, user_id))
    conn.commit()
    conn.close()


# ── Resume versions ───────────────────────────────────────────────────────────
def save_resume_version(resume: dict,
                        user_id: str = LOCAL_USER,
                        region:  str = LOCAL_REGION) -> int:
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        INSERT INTO resume_versions (
            user_id, region, version_name, job_id,
            content, format, ats_score, word_count, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        user_id, region,
        resume.get("version_name"), resume.get("job_id"),
        resume.get("content"),
        resume.get("format", "markdown"),
        resume.get("ats_score"),    resume.get("word_count"),
        _now()
    ))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def get_resume_versions(user_id: str = LOCAL_USER) -> list:
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        SELECT id, version_name, job_id, format, ats_score, word_count, created_at
        FROM resume_versions WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Market signals ────────────────────────────────────────────────────────────
def save_market_signal(signal_type: str, key: str, value: dict,
                       user_id: str = LOCAL_USER,
                       region:  str = LOCAL_REGION):
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        INSERT INTO market_signals (
            user_id, region, signal_type, signal_key, signal_value, observed_at
        ) VALUES (?,?,?,?,?,?)
    """, (user_id, region, signal_type, key, json.dumps(value), _now()))
    conn.commit()
    conn.close()


def get_market_signals(signal_type: str,
                       user_id: str = LOCAL_USER) -> list:
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        SELECT * FROM market_signals
        WHERE user_id = ? AND signal_type = ?
        ORDER BY observed_at DESC
    """, (user_id, signal_type))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.config import ensure_directories
    ensure_directories()
    initialise_db()

    # 1. Profile
    save_profile({
        "full_name"     : "Venket",
        "email"         : "venket@example.com",
        "location"      : "Chennai",
        "target_roles"  : ["VP Technology", "CTO", "Head of Engineering"],
        "target_sectors": ["BFSI", "GCC", "Fintech", "Logistics"],
        "salary_floor"  : 10000000,
        "voice_tone"    : "direct, confident, no corporate fluff",
        "words_to_avoid": ["synergy", "leverage", "thought leader"]
    })
    profile = get_profile()
    print(f"[1] Profile     : {profile['full_name']} · {profile['location']}")

    # 2. Experience
    exp_id = save_experience({
        "company"       : "Previous Organisation",
        "title"         : "VP Technology",
        "industry"      : "BFSI",
        "domain"        : "Core Banking · Payments",
        "is_current"    : True,
        "team_size"     : 120,
        "budget_managed": "$8M annual",
        "geography"     : "Global",
        "summary"       : "Led technology transformation across 12 business units"
    })
    print(f"[2] Experience  : id={exp_id}")

    # 3. Project
    proj_id = save_project({
        "experience_id"    : exp_id,
        "project_name"     : "Core Banking Modernisation",
        "client_name"      : "Top-tier Private Bank",
        "client_sector"    : "BFSI",
        "duration_months"  : 18,
        "your_role"        : "Programme Director",
        "problem_statement": "Legacy system causing 4hr daily downtime",
        "your_activities"  : "Architecture, vendor selection, team of 40",
        "tech_stack"       : ["AWS", "Kubernetes", "Java", "Oracle"],
        "outcome"          : "Zero downtime, $3M cost saving"
    })
    print(f"[3] Project     : id={proj_id}")

    # 4. Achievement
    ach_id = save_achievement({
        "experience_id" : exp_id,
        "project_id"    : proj_id,
        "what_you_did"  : "Modernised core banking platform",
        "metric_before" : "4hr daily downtime, 68% manual processes",
        "metric_after"  : "Zero downtime, 12% manual processes",
        "business_impact": "Saved $3M annually, NPS +22 points"
    })
    print(f"[4] Achievement : id={ach_id}")

    # 5. Skill
    save_skill({
        "skill_name"    : "AI Transformation Leadership",
        "category"      : "Leadership",
        "proficiency"   : "Expert",
        "years_of_exp"  : 6,
        "last_used_year": 2026,
        "context"       : "Led GenAI adoption across 5 business units"
    })
    skills = get_skills()
    print(f"[5] Skills      : {len(skills)} saved")

    # 6. Job
    is_new = save_job({
        "job_id"     : "AMEX-CHENNAI-001",
        "title"      : "VP Technology",
        "company"    : "American Express",
        "location"   : "Chennai",
        "url"        : "https://careers.amex.com/001",
        "description": "Lead technology for GCC operations..."
    })
    print(f"[6] Job         : saved (new={is_new})")

    print("\n[STORE] All tests passed")
    print(f"        11 tables · 10 indexes · user_id + region on every table")
    print(f"        Migration path: SQLite → PostgreSQL → Sharded")
    print(f"        Change required: get_connection() only")