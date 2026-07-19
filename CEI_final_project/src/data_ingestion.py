"""
data_ingestion.py

Pulls raw source material for PatchContext:
  - Commit history (message + SHA + diff stats)
  - Pull request threads (title, body, review comments)
  - Issue threads (title, body, comments)

from the FastAPI GitHub repo, using the public GitHub REST API.

Output: data/commits.json, data/pull_requests.json, data/issues.json
Each record carries its own citation info (SHA / PR number / issue number)
so later pipeline stages never lose the ability to cite a source.
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "fastapi/fastapi")
API_ROOT = "https://api.github.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def _get(url, params=None):
    """GET with basic rate-limit awareness."""
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset - time.time(), 1)
        print(f"Rate limited. Sleeping {wait:.0f}s...")
        time.sleep(wait)
        return _get(url, params)
    resp.raise_for_status()
    return resp


def fetch_commits(max_pages=5, per_page=30):
    """Fetch recent commit history with messages (used for 'why was X changed')."""
    commits = []
    for page in range(1, max_pages + 1):
        url = f"{API_ROOT}/repos/{GITHUB_REPO}/commits"
        resp = _get(url, params={"per_page": per_page, "page": page})
        batch = resp.json()
        if not batch:
            break
        for c in batch:
            commits.append({
                "sha": c["sha"],
                "short_sha": c["sha"][:7],
                "message": c["commit"]["message"],
                "author": (c["commit"]["author"] or {}).get("name"),
                "date": (c["commit"]["author"] or {}).get("date"),
                "url": c["html_url"],
            })
        print(f"Fetched commit page {page} ({len(batch)} commits)")
    return commits


def fetch_pull_requests(max_pages=5, per_page=30, state="closed"):
    """Fetch merged/closed PRs with body text (design rationale usually lives here)."""
    prs = []
    for page in range(1, max_pages + 1):
        url = f"{API_ROOT}/repos/{GITHUB_REPO}/pulls"
        resp = _get(url, params={
            "state": state, "per_page": per_page, "page": page,
            "sort": "updated", "direction": "desc",
        })
        batch = resp.json()
        if not batch:
            break
        for pr in batch:
            prs.append({
                "number": pr["number"],
                "title": pr["title"],
                "body": pr.get("body") or "",
                "merged_at": pr.get("merged_at"),
                "user": pr["user"]["login"] if pr.get("user") else None,
                "url": pr["html_url"],
            })
        print(f"Fetched PR page {page} ({len(batch)} PRs)")
    return prs


def fetch_issues(max_pages=5, per_page=30, state="closed"):
    """Fetch issue threads (bug reports / design discussions), skipping PRs
    (GitHub's issues endpoint returns both, so we filter PRs out)."""
    issues = []
    for page in range(1, max_pages + 1):
        url = f"{API_ROOT}/repos/{GITHUB_REPO}/issues"
        resp = _get(url, params={
            "state": state, "per_page": per_page, "page": page,
            "sort": "updated", "direction": "desc",
        })
        batch = resp.json()
        if not batch:
            break
        for issue in batch:
            if "pull_request" in issue:
                continue  # skip, handled by fetch_pull_requests
            issues.append({
                "number": issue["number"],
                "title": issue["title"],
                "body": issue.get("body") or "",
                "user": issue["user"]["login"] if issue.get("user") else None,
                "url": issue["html_url"],
                "closed_at": issue.get("closed_at"),
            })
        print(f"Fetched issue page {page} ({len(batch)} issues)")
    return issues


def save(records, filename):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} records -> {path}")


if __name__ == "__main__":
    print(f"Ingesting data from {GITHUB_REPO}...")
    commits = fetch_commits(max_pages=5)
    save(commits, "commits.json")

    prs = fetch_pull_requests(max_pages=5)
    save(prs, "pull_requests.json")

    issues = fetch_issues(max_pages=5)
    save(issues, "issues.json")

    print("Ingestion complete.")
