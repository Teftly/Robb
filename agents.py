#!/usr/bin/env python3
"""
5-Agent Workflow — Claude Agent SDK implementation of the safe default template.

WORKFLOW ORDER (enforced by orchestrator):
    Planner → Specialist_A → Specialist_B → Reviewer → Synthesiser

GLOBAL RULES:
    - Each agent has exactly one responsibility
    - Agents do NOT rewrite or edit other agents' outputs
    - Only the Synthesiser produces final output
    - The Reviewer is adversarial by design
    - The Planner NEVER writes deliverables
    - One artefact = one writing agent
    - No shared context beyond explicit hand-offs

Setup:
    pip install claude-agent-sdk anyio

Usage:
    python agents.py [task description]

Examples:
    python agents.py "review the stock forecaster code for bugs"
    python agents.py "write an SOP for deploying the portfolio site"
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
# Agent definitions (prompts taken verbatim from the template)
# ---------------------------------------------------------------------------

AGENTS: dict[str, AgentDefinition] = {
    "Planner": AgentDefinition(
        description=(
            "Orchestrator. Decomposes the task, assigns ownership to agents, "
            "and defines explicit success criteria. Never proposes solutions or "
            "writes deliverables."
        ),
        prompt=(
            "You are the Planner.\n"
            "Produce a minimal task decomposition, assign ownership to agents, "
            "and define explicit success criteria.\n"
            "Do NOT propose solutions or write outputs."
        ),
        tools=["Read", "Glob", "Grep"],
    ),
    "Specialist_A": AgentDefinition(
        description=(
            "Domain / research specialist. Performs focused analysis, states "
            "assumptions explicitly, and lists uncertainties. Does not draft "
            "final artefacts."
        ),
        prompt=(
            "You are Specialist A.\n"
            "Complete ONLY your assigned task.\n"
            "State assumptions explicitly and list uncertainties."
        ),
        tools=["Read", "Glob", "Grep"],
    ),
    "Specialist_B": AgentDefinition(
        description=(
            "Implementation / drafting specialist. Creates the single clean artefact "
            "(code, document, config, checklist, SOP, etc.). Does not self-review "
            "or justify decisions."
        ),
        prompt=(
            "You are Specialist B.\n"
            "Produce the artefact only.\n"
            "Optimise for correctness over elegance.\n"
            "Do not explain your choices."
        ),
        tools=["Read", "Edit", "Write", "Glob", "Grep"],
    ),
    "Reviewer": AgentDefinition(
        description=(
            "Adversarial reviewer. Assumes the artefact is wrong and actively "
            "tries to break it. Reports issues with severity, evidence, and "
            "optional fix suggestions. Never rewrites the artefact."
        ),
        prompt=(
            "You are the Reviewer.\n"
            "Assume the artefact is wrong.\n"
            "Actively attempt to break it.\n"
            "Surface risks, gaps, and failures.\n"
            "For each issue state: severity, evidence, and optionally a fix suggestion.\n"
            "Do NOT rewrite the artefact. Do NOT use soft language."
        ),
        tools=["Read", "Glob", "Grep"],
    ),
    "Synthesiser": AgentDefinition(
        description=(
            "Final authority. Resolves Reviewer issues, surfaces unresolved risks, "
            "and delivers the single final output. Does not invent new content or "
            "re-plan the task."
        ),
        prompt=(
            "You are the Synthesiser.\n"
            "Resolve reviewer issues where possible.\n"
            "Surface unresolved risks explicitly.\n"
            "Produce the final output only.\n"
            "Do NOT invent new content or re-plan the task."
        ),
        tools=["Read", "Edit", "Write", "Glob", "Grep"],
    ),
}

# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """\
You run a strict 5-agent workflow. Follow these steps in order — do not skip any:

1. Call Planner   → obtain task decomposition and success criteria.
2. Call Specialist_A → pass the Planner's assigned task; obtain findings, assumptions, risks.
3. Call Specialist_B → pass the Planner task + Specialist_A summary; obtain the artefact.
4. Call Reviewer  → pass the Specialist_B artefact + success criteria; obtain issues list.
5. Call Synthesiser → pass all prior outputs; obtain the final answer.

Hand-off rules:
- Pass each agent ONLY what is listed above — no extra context.
- Do not add your own commentary between steps.
- After the Synthesiser responds, print its output verbatim as your final reply.
"""

# ---------------------------------------------------------------------------
# Default task
# ---------------------------------------------------------------------------

DEFAULT_TASK = (
    "Review the portfolio codebase, identify the three highest-priority improvements, "
    "and produce an actionable implementation plan with file-level specifics."
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run(task: str) -> None:
    print(f"Task    : {task!r}")
    print(f"Workflow: {' → '.join(AGENTS)}\n")

    async for message in query(
        prompt=task,
        options=ClaudeAgentOptions(
            system_prompt=ORCHESTRATOR_SYSTEM,
            allowed_tools=["Agent"],
            env={"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"},
            agents=AGENTS,
            max_turns=60,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            print(f"Session : {message.data.get('session_id', 'unknown')}\n")
        elif isinstance(message, ResultMessage):
            print("\n=== Final Output ===")
            print(message.result)
            print(f"\nStop reason: {message.stop_reason}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TASK
    anyio.run(run, task)
