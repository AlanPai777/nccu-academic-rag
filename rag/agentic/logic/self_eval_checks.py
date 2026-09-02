"""
rag/agentic/logic/self_eval_checks.py
Stage 1 of self_eval_node's two-stage design (Migration Step 7,
docs/phase_h_agentic_rag_migration_plan.md §6.2/§6.3, Part 5 Step 7) --
deterministic, zero-LLM-cost checks that run BEFORE the Stage 2 LLM
judgment (agentic_main.py's existing self_eval_node/_SELF_EVAL_PROMPT,
ported into rag/agentic/nodes/self_eval.py). A Stage 1 failure produces a
concrete, deterministic correction hint (the checklist already knows
what's missing, no LLM needed to describe it) and skips Stage 2 entirely.

Adapted from production's rag/nodes/self_eval.py's _SELF_EVAL_CRITERIA
(sources/procedure_format keyword checks) plus a structural-completeness
spirit borrowed from its stations/person_names checks -- but NOT a
literal port, because production's structural checks read a separate
extraction_checklist state field that doesn't exist in this package's
state (see rag/agentic/state.py's docstring on why: Migration Step 1
corrected state.py to match agentic_main.py's actual AgenticState, which
has no such field). Here, the equivalent structural signal is read
directly from state["messages"] -- resource_node's own
"[表單站點審核層級偵測]" block and contact_node's own "[辦公室聯絡資訊...]"
block, both already-computed data that's sitting in the message history
rather than a parallel state field.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

_CONTACT_NAME_RE = re.compile(r'•\s*([^\s（(]{2,10})[（(]')
_STATION_ROLE_LINE_RE = re.compile(r'站點\d+：([^（(]+)')


def _extract_contact_names(messages: list) -> list[str]:
    """Pulls real contact names out of contact_node's own
    "[辦公室聯絡資訊...]" injected HumanMessage(s) -- e.g. "• 劉吉軒（教務長）
    分機 62160" -> "劉吉軒". Scans ALL such messages this turn, not just the
    last one, since a multi-turn KNOWLEDGE run can trigger contact_node
    more than once."""
    names: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage) and "[辦公室聯絡資訊" in str(m.content):
            for match in _CONTACT_NAME_RE.finditer(str(m.content)):
                name = match.group(1)
                if name not in names:
                    names.append(name)
    return names


def _extract_station_roles(messages: list) -> list[str]:
    """Pulls station role keywords out of resource_node's own
    "[表單站點審核層級偵測]" block -- e.g. "- 站點7：組長、教務長（多層審核...)"
    -> ["組長", "教務長"]. Only present when a fetched form's table actually
    had a multi-role station (D15's _extract_station_roles finding
    something) -- most forms won't trigger this block at all, which is
    expected, not a gap."""
    roles: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage) and "[表單站點審核層級偵測]" in str(m.content):
            for line_match in _STATION_ROLE_LINE_RE.finditer(str(m.content)):
                for role in line_match.group(1).split('、'):
                    role = role.strip()
                    if role and role not in roles:
                        roles.append(role)
    return roles


def stage1_checklist(state: dict) -> list[str]:
    """Deterministic, free checks. Returns a list of human-readable
    failure descriptions (usable directly as a correction hint); an empty
    list means Stage 1 passed and Stage 2 (the LLM judgment) should run.

    Deliberately NOT exhaustive -- these are the checks with the lowest
    false-positive risk (matches production's own sources/procedure_format
    criteria, plus a completeness check against data resource_node/
    contact_node already computed and left in messages). A broader
    checklist_blocks-style completeness check (verifying N.5's routing-
    detail findings specifically show up) was considered and deferred --
    building a reliable presence-check against arbitrary bullet text
    without more real-case testing risks the same kind of unreliable
    deterministic rule this project has repeatedly had to walk back
    elsewhere (see docs/phase_g_clean_pipeline_design.md's D12/_STRIP_RE
    lessons)."""
    answer = state.get("answer") or ""
    messages = state.get("messages") or []
    failures: list[str] = []

    if not any(kw in answer for kw in ("來源", "https://")):
        failures.append("答案缺少引用來源。請在最後加上來源URL。")

    # Step-format is only expected for procedure-shaped questions --
    # mirrors _SYNTHESIS_PROMPT rule 2's own distinction ("單純查資訊/查
    # 電話這類非流程類問題不受此規則限制"). query_type carries plan_node's
    # classification label even though PROCEDURE/KNOWLEDGE share one graph
    # path (Migration Step 1 decision) -- the label itself still exists
    # for exactly this kind of downstream distinction.
    if state.get("query_type") == "procedure":
        if not any(kw in answer for kw in ("第一步", "步驟", "①", "第1步", "1.", "1、")):
            failures.append("答案未使用步驟清單格式。請改為以編號步驟列出辦理流程。")

    contact_names = _extract_contact_names(messages)
    if contact_names:
        hits = sum(1 for name in contact_names if name in answer)
        if hits == 0:
            preview = "、".join(contact_names[:5])
            failures.append(
                f"contact_node查到了真實承辦人姓名（例如：{preview}），但答案裡完全沒有引用任何一位。"
                "請在對應步驟列出承辦人姓名（不能只寫辦公室名稱）。"
            )

    station_roles = _extract_station_roles(messages)
    if station_roles:
        hits = sum(1 for role in station_roles if role in answer)
        if hits == 0:
            preview = "、".join(station_roles)
            failures.append(
                f"表單記載某個站點需要多層審核（{preview}），但答案裡沒有反映這個多層審核結構。"
                "請把每一層審核都列出來，不要只挑一層。"
            )

    return failures
