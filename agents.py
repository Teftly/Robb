#!/usr/bin/env python3
"""
Portfolio Agent Team — Claude Agent SDK multi-agent workflow.

Enables CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS so the orchestrator can
dispatch sub-tasks to specialised agents in parallel.

Setup:
    pip install claude-agent-sdk anyio

Usage:
    python agents.py [task description]

Examples:
    python agents.py "review the stock forecaster code for bugs"
    python agents.py "improve accessibility on the CV page"
    python agents.py          # runs the default portfolio review
"""

import sys
import anyio

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    query,
)

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

AGENTS: dict[str, AgentDefinition] = {
    "analyst": AgentDefinition(
        description=(
            "Scans the codebase and produces a structured report of bugs, "
            "accessibility issues, performance problems, and technical debt."
        ),
        prompt=(
            "You are a senior code analyst specialising in web technologies and Python. "
            "Read files, identify issues, and produce a clear structured report. "
            "Focus on: correctness, accessibility, performance, and maintainability. "
            "Always cite the specific file path and line number for each finding."
        ),
        tools=["Read", "Glob", "Grep"],
    ),
    "web-dev": AgentDefinition(
        description=(
            "Implements targeted HTML / CSS / JavaScript improvements to the "
            "portfolio pages based on the analyst's findings."
        ),
        prompt=(
            "You are a front-end web developer. "
            "Apply minimal, targeted fixes identified by the analyst. "
            "Prefer editing existing files over creating new ones. "
            "Do not touch code that was not flagged."
        ),
        tools=["Read", "Edit", "Write", "Glob", "Grep"],
    ),
    "python-dev": AgentDefinition(
        description=(
            "Improves Python scripts (stock forecaster, setup scripts): "
            "type hints, error handling, docstrings, and optimisations."
        ),
        prompt=(
            "You are a Python developer. "
            "Improve code quality in Python files flagged by the analyst. "
            "Add type annotations, improve error handling, and add concise docstrings. "
            "Keep changes focused — do not refactor unflagged code."
        ),
        tools=["Read", "Edit", "Write", "Glob", "Grep"],
    ),
}

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

DEFAULT_TASK = (
    "Review the portfolio codebase and produce a prioritised list of "
    "improvements. Use the analyst agent to scan all files, then delegate "
    "fixes to the web-dev or python-dev agents as appropriate."
)


async def run(task: str) -> None:
    print(f"Task : {task!r}")
    print(f"Agents: {', '.join(AGENTS)}\n")

    async for message in query(
        prompt=task,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep", "Agent"],
            env={"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"},
            agents=AGENTS,
            max_turns=40,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            session_id = message.data.get("session_id", "unknown")
            print(f"Session: {session_id}\n")
        elif isinstance(message, ResultMessage):
            print("\n=== Result ===")
            print(message.result)
            print(f"\nStop reason: {message.stop_reason}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TASK
    anyio.run(run, task)
