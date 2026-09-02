"""
rag/agentic/logic/form_extraction.py
Form metadata lookup + deterministic structural extraction from fetched
form text (Migration Step 4, docs/phase_h_agentic_rag_migration_plan.md
Part 5). Ported directly from rag/agentic_main.py -- D15's structural
extraction (station roles, checklist blocks) plus _judge_forms(), the
fallback path used only when resource_node is entered directly from
plan_node's RESOURCE route (no page text to extract form_ids from yet).
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from rag.llm_client import simple_chat
from rag.agent_tools import CRAWLER_OUTPUT
from rag.domain_router import _layer1_match, _keyword_table

# Marker get_page_tool embeds in its returned text when extract_form_ids()
# finds a moltke form reference, so _after_tools can route to resource_node
# without re-running detection itself.
_FORM_MARKER_RE = re.compile(r"\[偵測到表單編號: ([^\]]+)\]")

_FORM_JUDGE_PROMPT = """你要判斷回答這個問題需不需要抓取特定表單的全文，以及該抓哪一份。

使用者問題：{query}

目前已有的內容（可能是已抓取的頁面/表單全文，也可能還沒有任何內容）：
{context}

已知的表單清單（含表單編號、標題、用途說明、承辦單位——這份清單本來就存在，不是搜尋結果）：
{forms}

請判斷：這份表單清單裡，有沒有表單是回答這題需要抓取全文查看細節的？頁面敘述常常只是概略帶過，實際細節（蓋章站點、費用標準、資格條件等）常常只寫在表單裡；但也可能已有內容已經足夠，不需要任何表單。

只回傳需要抓取的表單編號，用逗號分隔；如果都不需要，回傳「無」。不要有其他文字。"""


def _list_forms_metadata(subdomain_hint: str | None) -> list[dict]:
    """Reads supplementary_map.json's own records directly -- form_id/
    form_title/form_description/form_unit already exist there for every
    form. A bounded lookup over already-indexed metadata, NOT a search
    capability. Scoped to subdomain_hint's own file when given (~20-40
    forms per subdomain); globs all subdomains only when no hint exists
    yet (still just reading existing files, not searching)."""
    pattern = (f"{CRAWLER_OUTPUT}/{subdomain_hint}/supplementary_map.json"
               if subdomain_hint else f"{CRAWLER_OUTPUT}/*/supplementary_map.json")
    forms: list[dict] = []
    for path in glob.glob(pattern):
        try:
            for r in json.loads(Path(path).read_text()):
                fid = r.get("form_id")
                if fid:
                    forms.append({
                        "form_id": fid,
                        "title": r.get("form_title", ""),
                        "description": r.get("form_description", ""),
                        "unit": r.get("form_unit", ""),
                    })
        except (json.JSONDecodeError, OSError):
            pass
    return forms


def _judge_forms(query: str, context_text: str, subdomain_hint: str | None) -> list[str]:
    """Single unified judgment step, used only when resource_node has no
    page text to extract form_ids from (the direct plan_node RESOURCE
    route) -- always judges against the FULL known-forms metadata list
    (_list_forms_metadata(), a bounded lookup, never a search), not a
    candidate set narrowed by extract_form_ids(). A candidate-pool-
    narrowing optimization was tried and reverted here (§N.1/N.3 in the
    design doc this was ported from): rigorous re-testing (8 runs per
    condition, same query/context/timestamp) showed the FULL 32-candidate
    version scored 8/8 correct while a narrowed-to-2-candidates version
    scored 0/8 -- the opposite of what the optimization assumed. Do not
    re-add candidate-pool narrowing here without re-verifying against a
    large sample first."""
    scope = subdomain_hint or _layer1_match(query)
    forms = _list_forms_metadata(scope)
    if not forms:
        return []
    forms_str = "\n".join(f"- {f['form_id']}：{f['title']}（{f['unit']}）{f['description']}" for f in forms)
    prompt = _FORM_JUDGE_PROMPT.format(
        query=query,
        context=context_text if context_text else "（尚無內容）",
        forms=forms_str,
    )
    raw = simple_chat(messages=[{"role": "user", "content": prompt}], max_tokens=150).strip()
    valid_ids = {f["form_id"] for f in forms}
    return [fid for fid in valid_ids if fid in raw]


_STATION_ROLE_RE = re.compile(r'[（(](\d+)[）)]\s*([^\s|]{1,20})')
_ROLE_KEYWORDS = ['組長', '教務長', '主任', '批示', '核准', '簽核', '審核', '核可', '秘書']
_CHECKBOX_SKIP_LABELS = {
    '原因', '性別 Sex', '房間人數 Room Type', '學生身份 Student Category',
}  # generic student self-declared demographic fields, not decision/routing
   # info the answer should relay -- not a complete list, extend as more
   # forms surface the same kind of identity-field noise.


def _extract_station_roles(text: str) -> dict[str, list[str]]:
    """Deterministic table-row parser -- for each numbered station
    ("（N）label"), scans OTHER cells in the SAME row for short (<=15
    chars) cells containing a role keyword (組長/教務長/etc). A row with
    exactly ONE station marker plus extra role cells means that ONE
    station has multiple approval layers; a row with several station
    markers and no extra cells means those are independent single-role
    stations. Purely structural (counts/matches table cells), no LLM
    call."""
    roles: dict[str, list[str]] = {}
    for line in text.split('\n'):
        if '|' not in line:
            continue
        cells = [c.strip() for c in line.split('|') if c.strip()]
        stations = [m.group(1) for c in cells for m in _STATION_ROLE_RE.finditer(c)]
        if len(stations) != 1:
            continue
        extras = []
        for c in cells:
            c_clean = c.replace(' ', '').replace('　', '')
            if _STATION_ROLE_RE.search(c):
                continue
            if len(c_clean) <= 15 and any(kw in c_clean for kw in _ROLE_KEYWORDS):
                extras.append(c_clean)
        if extras:
            roles[stations[0]] = extras
    return roles


def _offices_from_role_keywords(role_cells: list[str]) -> list[str]:
    """Deterministic reverse lookup closing a gap _detect_offices()'s LLM
    judge leaves uncaught: a role cell like "教務長批示" contains the role
    keyword "教務長" -- rather than matching the WHOLE noisy cell text
    against the catalog, substring-match the ALREADY-ISOLATED short
    keyword against catalog names instead. "教務長" is a literal substring
    of catalog entry "教務長室" -- confirmed to match uniquely. "組長"
    deliberately matches no catalog entry (too generic a title to be an
    office name by itself)."""
    catalog_names = [name for name, _sub in _keyword_table()]
    found: list[str] = []
    for cell in role_cells:
        for kw in _ROLE_KEYWORDS:
            if kw not in cell:
                continue
            for name in catalog_names:
                if kw in name and name not in found:
                    found.append(name)
    return found


_LABEL_FRAGMENT_MAX_LEN = 6  # e.g. 申請/處理/方式 -- short table-cell labels, not prose


def _extract_checklist_blocks(text: str) -> list[dict]:
    """Deterministic □-option block parser. Two passes, because forms
    linearize their HTML tables into markdown rows in two different
    shapes:

    Pass 1: a SINGLE line with 2+ checkbox markers and a plausible
    descriptive label in the SAME row -- works when the label and all its
    choices survive extraction on one row (e.g. 休學's 領取方式 line).

    Pass 2: some forms' tables instead linearize a merged/vertical label
    cell into SEPARATE short rows, each paired with its OWN single-□
    option row (confirmed via QP-M03-01-04's 500元 delayed-fee option,
    whose label "申請/處理/方式" is split across 3 separate rows). Pass 1's
    "2+ □ on one line" requirement never matches this shape at all. Pass 2
    groups RUNS of consecutive lines that are each either a single-□
    option or a short label fragment (<=6 chars) into one block; any other
    line (blank row, a numbered station marker, long prose) ends the run.
    A run needs 2+ □ lines before being kept, matching pass 1's own bar.

    Both passes filter obvious student-fill-in fields via
    _CHECKBOX_SKIP_LABELS -- NOT a complete filter, may need further
    iteration against more forms."""
    blocks = []
    claimed_lines: set[int] = set()
    lines = text.split('\n')

    for i, line in enumerate(lines):
        if line.count('□') < 2:
            continue
        cells = [c.strip() for c in line.split('|') if c.strip()]
        label = next((c for c in cells if '□' not in c and len(c) < 20), None)
        options = next((c for c in cells if '□' in c), '')
        if label is None or label in _CHECKBOX_SKIP_LABELS:
            continue
        blocks.append({"label": label, "options": options})
        claimed_lines.add(i)

    group_options: list[str] = []
    group_labels: list[str] = []

    def _flush() -> None:
        if len(group_options) >= 2:
            label = ''.join(group_labels) if group_labels else '（表單勾選項目）'
            if label not in _CHECKBOX_SKIP_LABELS:
                blocks.append({"label": label, "options": ' '.join(group_options)})
        group_options.clear()
        group_labels.clear()

    for i, line in enumerate(lines):
        if i in claimed_lines or '|' not in line:
            _flush()
            continue
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if line.count('□') == 1:
            group_options.append(next(c for c in cells if '□' in c))
            for c in cells:
                if '□' not in c and 0 < len(c) <= _LABEL_FRAGMENT_MAX_LEN and c not in group_labels:
                    group_labels.append(c)
            continue
        if line.count('□') == 0 and len(cells) == 1 and 0 < len(cells[0]) <= _LABEL_FRAGMENT_MAX_LEN:
            if cells[0] not in group_labels:
                group_labels.append(cells[0])
            continue
        _flush()
    _flush()
    return blocks
