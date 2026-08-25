"""
rag/skills/office_lookup_skill.py
Per-office contact info lookup for NCCU procedure answers.

Resolves floor number, phone extensions, and responsible person for each
stop in a 7-stop procedure chain.

Data sources (in priority order):
  1. office_contacts_index.jsonl — Playwright-extracted staff directory pages
     (Phase F Step 0c, 2026-08-25), parsed dynamically by parse_office_contacts().
     PRIMARY source — not hardcode-primary (Phase F condition 4).
  2. KNOWN_CONTACTS — static fallback only, used when (1) finds nothing for an
     office (missing from the index, or its page's text doesn't match either
     parser tier). Kept intentionally small; this is what verification tests
     against a real-but-uncovered office name to confirm graceful degradation
     rather than a crash or blank answer.
  3. Dynamic grep → get_page for per-office member/contact pages (existing,
     independent of (1)/(2) — searches extracted_texts.jsonl, not the contacts index)
  4. KNOWN_FLOORS fallback for floors absent from crawled data (moltke-verified)
"""

import json
import re
from pathlib import Path

from rag.agent_tools import grep_texts, get_page

# Floor info from moltke QP-T01-03-02 + osa refund page.
# Floors for 生僑組 and 原資中心 are not present in crawled HTML;
# they share 行政大樓 3 樓 with 住宿組 (confirmed in eval_baseline).
KNOWN_FLOORS = {
    "出納組":         "行政大樓 5 樓",
    "住宿組":         "行政大樓 3 樓",
    "生僑組":         "行政大樓 3 樓",
    "原資中心":       "行政大樓 3 樓",
    "國際合作事務處": "行政大樓 8 樓",
    "國合處":         "行政大樓 8 樓",
    "教務處":         "行政大樓 4 樓",
    "教務長":         "行政大樓 4 樓",
    "註冊組":         "行政大樓 4 樓",
    "圖書館":         "中正圖書館 / 達賢圖書館 / 各院分館",
}


# ── Condition 4: dynamic contacts from office_contacts_index.jsonl ──────────

_CONTACTS_INDEX_PATH = Path(__file__).parent.parent.parent / "output" / "office_contacts_index.jsonl"

# Maps the office names used elsewhere in this codebase (_PROCEDURE_OFFICES,
# KNOWN_FLOORS keys) to the "name" field(s) actually present in
# office_contacts_index.jsonl — these don't match 1:1 by substring (e.g. the
# aca homepage's own "教務處" record is a near-empty landing page; the real
# staff directory lives under "教務長室"/"註冊組"). Built by inspecting the
# index directly (Phase F Step 3), not guessed.
_OFFICE_NAME_MAP: dict[str, list[str]] = {
    "教務處":         ["教務長室", "註冊組"],
    "教務長":         ["教務長室"],
    "註冊組":         ["註冊組"],
    "生僑組":         ["生活事務暨僑生輔導組"],
    "住宿組":         ["住宿輔導組"],
    "出納組":         ["出納組"],
    "國際合作事務處": ["國際合作事務處"],
    "國合處":         ["國際合作事務處"],
    "圖書館":         ["圖書館(各組聯絡資訊)"],
}

# Used by Tier 2 (_parse_flattened_table) to reject a "pure name line"
# candidate that's actually a title (aca's 職稱 column, e.g. "教務長"). Kept
# as its own set (broader than _ADMIN_TITLES below, which also includes
# 教授-family titles _ADMIN_TITLES deliberately excludes) — over-excluding
# here only makes Tier 2 more conservative, never worse.
_TITLE_WORDS = {
    "組長", "副組長", "編審", "組員", "秘書", "祕書", "主任", "館長", "教授", "副教授",
    "助理教授", "專員", "行政專員", "一級行政專員", "一級行政組員", "兼任助理",
    "行政組員", "職員", "教務長", "副教務長", "國合長", "院長", "主管", "館員", "一級行政",
    "莊敬內舍生活輔導員", "莊敬九舍生活輔導員", "自強九舍生活輔導員", "自強十舍生活輔導員",
    "資深約用心理師", "新生書院導師", "書院專案助理", "專案行政專員", "定期約用人員",
    "計畫專任助理", "住宿組組長", "約用心理師", "約用社工師", "約用社工員", "約用護理師",
    "約用職代", "專任助理", "中心主任", "職涯總監", "中心秘書", "中心組長", "輔導員",
    "護理師", "一級技術師", "工程師", "技術師", "技士",
}

# Tier 1: explicitly-labeled fields ("分機"/"校內分機" immediately before the
# extension) — covers osa's per-person div-card pages and cashier's
# PageStaffing format. Negative lookaround excludes "職務代理人：王小明（分機：
# 62221）"-style deputy REFERENCES embedded in someone else's own entry —
# without it, the deputy mention (which appears earlier in the text) wins the
# de-dup race against that person's own real, fuller entry.
_EXT_LABELED_RE  = re.compile(r'(?<!（)(?:校內)?分機\s*[：:]?\s*\n*\s*(\d{4,5})(?!\s*）)')
_NAME_LABELED_RE  = re.compile(r'姓\s*名\s*\n*\s*([一-鿿]{2,6})')
_TITLE_LABELED_RE = re.compile(r'職\s*稱\s*\n*\s*([^\n]{2,20})')
_EMAIL_RE         = re.compile(r'([\w.+-]+@nccu\.edu\.tw)')

# When no "姓名" label exists (e.g. cashier's "劉桂芬 組長" run-on format),
# only trust a name immediately adjacent to a KNOWN administrative title —
# NOT a bare "last 2-4 hanzi before the extension" guess. That looser
# version fabricated names from institutional/duty prose on oic's page
# (e.g. "...研究所教授" → wrongly extracted "斯研究所" as a person's name,
# since oic's bios end in a title-like word with no real name label or
# adjacency anywhere nearby).
#
# _ADMIN_TITLES is empirically harvested (Phase F Step 3, 2026-08-25), not
# manually enumerated from the 6 offices we happened to test — a hand-built
# list would only ever cover titles we'd already seen, which is exactly the
# overfitting risk for the other 138 subdomains in office_contacts_index.jsonl
# nobody has read. Harvest method: scan every record where BOTH "職稱" and
# "姓名" labels are present (Tier 1's own high-confidence signal), and take
# the 職稱 value of every block that also has a 姓名 nearby — i.e. let the
# labeled format teach the unlabeled-format parser its own title vocabulary,
# corpus-wide. This is what actually generalizes past our 6 test offices.
# Deliberately excludes 教授/副教授/助理教授 (oic's false-positive cause,
# and they never got harvested anyway — no page pairs a professor title with
# a 姓名 label) plus a few that never appeared: 工程師/一級技術師/技術師 are
# added by hand from a page (圖書館系統資訊組) confirmed by direct manual
# read, since the harvest can only find titles that occur SOMEWHERE in
# labeled form. The same blind spot bit "編審"/"兼任助理"/"副組長" here too —
# those never got harvested despite being real, previously-verified titles on
# cashier's page (also unlabeled, same structural reason 工程師 was missed) —
# caught only by re-running the 6-office regression check after switching to
# the harvested list, which is exactly why that check exists. So this is the
# UNION of the corpus-wide harvest with every title manually confirmed on the
# specific unlabeled pages read directly (cashier, 圖書館系統資訊組) — not a
# straight replacement of the hand-curated list, which would have silently
# dropped real titles the harvest structurally can't see.
_ADMIN_TITLES = sorted([
    # harvested corpus-wide from labeled-format pages
    "莊敬內舍生活輔導員", "莊敬九舍生活輔導員", "自強九舍生活輔導員", "自強十舍生活輔導員",
    "資深約用心理師", "一級行政專員", "一級行政組員", "新生書院導師", "書院專案助理",
    "專案行政專員", "定期約用人員", "計畫專任助理", "住宿組組長", "約用心理師",
    "約用社工師", "約用社工員", "約用護理師", "行政專員", "約用職代", "專任助理",
    "行政組員", "中心主任", "職涯總監", "中心秘書", "中心組長", "輔導員", "護理師",
    "組長", "組員", "技士", "專員", "主任",
    # manually confirmed on unlabeled-format pages the harvest can't reach
    "編審", "副組長", "兼任助理", "秘書", "祕書", "館長", "副館長", "館員",
    "一級技術師", "工程師", "技術師",
], key=len, reverse=True)
_NAME_TITLE_RE = re.compile(rf'([一-鿿]{{2,4}})\s*(?:{"|".join(_ADMIN_TITLES)})')

# Tier 2: flattened HTML tables (Joomla "人員職掌" pages, e.g. aca's 教務長室/
# 註冊組/課務組/綜合業務組) — the table header ("職稱/姓名/業務項目/分機/職務
# 代理人") appears once; per-row values then run together with no per-field
# labels. The one reliable anchor is each row's OWN extension: a bare 4-5
# digit number NOT wrapped in parens (deputy extensions ARE parenthesized,
# e.g. "(62846)", which is exactly how this format writes the same
# 職務代理人 references Tier 1 has to exclude a different way).
_PURE_NAME_LINE_RE = re.compile(r'^\s*\[?([一-鿿]{2,4})\]?(?:\(https?://[^)]*\))?\s*$')
_EXT_BARE_RE        = re.compile(r'(?<!\()(?<!\d)(\d{5})(?!\d)(?!\))')


def _parse_labeled(text: str) -> list[dict]:
    ext_matches = list(_EXT_LABELED_RE.finditer(text))
    contacts: list[dict] = []
    seen: set[str] = set()
    for i, m in enumerate(ext_matches):
        ext = m.group(1)
        if ext in seen:
            continue
        block_start = ext_matches[i - 1].end() if i > 0 else max(0, m.start() - 300)
        block = text[block_start:m.start()].rstrip()
        name_match = _NAME_LABELED_RE.search(block)
        if name_match:
            name = name_match.group(1)
        else:
            # Search the WHOLE block, not just its tail — the name+title pair
            # sits right before the extension in cashier's format ("...代理人
            # \n\n{name} {title}\n\n校內分機") but right at the block's START
            # in another format (圖書館系統資訊組: "{name} {title}\n\n{long
            # duty paragraph}\n\n分機：{ext}"). Take the FIRST match in the
            # block, since block boundaries are already per-person (start at
            # the previous person's extension), so the first name+title pair
            # found belongs to this block's own owner.
            nt_match = _NAME_TITLE_RE.search(block)
            name = nt_match.group(1) if nt_match else ""
        if not name:
            continue
        seen.add(ext)
        title_match = _TITLE_LABELED_RE.search(block)
        duty = title_match.group(1).strip() if title_match else ""
        block_end = ext_matches[i + 1].start() if i + 1 < len(ext_matches) else m.end() + 200
        email_match = _EMAIL_RE.search(text[m.end():block_end])
        contacts.append({
            "name": name, "ext": ext,
            "email": email_match.group(1) if email_match else None,
            "duty": duty,
        })
    return contacts


def _parse_flattened_table(text: str) -> list[dict]:
    matches = list(_EXT_BARE_RE.finditer(text))
    contacts: list[dict] = []
    seen: set[str] = set()
    prev_end = 0
    for m in matches:
        ext = m.group(1)
        block = text[prev_end:m.start()]
        prev_end = m.end()
        if ext in seen:
            continue
        name = ""
        for line in (l.strip() for l in block.split("\n") if l.strip()):
            pm = _PURE_NAME_LINE_RE.match(line)
            if pm and pm.group(1) not in _TITLE_WORDS:
                name = pm.group(1)  # last pure-name line before the ext wins
        if not name:
            continue
        seen.add(ext)
        email_match = _EMAIL_RE.search(text[m.end():m.end() + 150])
        contacts.append({
            "name": name, "ext": ext,
            "email": email_match.group(1) if email_match else None,
            "duty": "",
        })
    return contacts


def parse_office_contacts(text: str) -> list[dict]:
    """
    Extract [{name, ext, email, duty}, ...] from one office_contacts_index.jsonl
    record's raw text. Tries the labeled-field parser first (works for most
    CMS layouts observed); falls back to the flattened-table parser only when
    BOTH the labeled parser found nothing AND "分機" appears just a handful of
    times in the whole page (<=3) — the fingerprint of a genuine flattened
    table, where the column header ("職稱/姓名/業務項目/分機/..." on aca's
    pages, "館名/單位/...校內分機/..." on the library's) appears exactly
    once and per-row values are bare, unlabeled numbers. A page where "分機"
    appears many times (oic: 36) means every person already has their OWN
    labeled "分機：" — tier 1 correctly tried all of them and correctly found
    no reliable name for any (oic's bios end in institutional phrases with no
    name nearby), and falling back to the flattened-table guess there
    misfired on oic's own short standalone duty-list lines (e.g. "校務評鑑",
    "總務庶務") which happen to also match that parser's "short pure-hanzi
    line" heuristic — producing confident-looking garbage names for a page
    where no name is reliably in the text at all. Not a claim of 100%
    precision on every subdomain's CMS quirks — good enough to identify the
    right office and a plausible contact for it, which is the bar Phase F
    set for this (see Step 0c contact.csv discussion); an office this can't
    parse falls all the way back to KNOWN_CONTACTS via
    dynamic_contacts_for_office(), not to a guess.
    """
    contacts = _parse_labeled(text)
    if not contacts and text.count("分機") <= 3:
        contacts = _parse_flattened_table(text)
    return contacts


_contacts_index_cache: dict[str, list[dict]] | None = None  # name -> raw text, loaded once


def _load_contacts_index() -> dict[str, list[dict]]:
    global _contacts_index_cache
    if _contacts_index_cache is not None:
        return _contacts_index_cache

    by_name: dict[str, list[dict]] = {}
    if _CONTACTS_INDEX_PATH.exists():
        with _CONTACTS_INDEX_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_name.setdefault(rec.get("name", ""), []).append(rec)
    _contacts_index_cache = by_name
    return by_name


def dynamic_contacts_for_office(office: str) -> list[dict]:
    """
    Primary contact lookup for `office` (Phase F condition 4) — queries
    office_contacts_index.jsonl via _OFFICE_NAME_MAP, parses each matching
    record's raw text, and merges results (deduped by extension). Returns []
    if the office isn't in _OFFICE_NAME_MAP or none of its mapped records
    parse to anything — caller falls back to KNOWN_CONTACTS in that case.
    """
    index = _load_contacts_index()
    record_names = _OFFICE_NAME_MAP.get(office, [])

    contacts: list[dict] = []
    seen_exts: set[str] = set()
    for record_name in record_names:
        for rec in index.get(record_name, []):
            for c in parse_office_contacts(rec.get("text", "")):
                if c["ext"] in seen_exts:
                    continue
                seen_exts.add(c["ext"])
                contacts.append(c)
    return contacts


# Static fallback only — used when dynamic_contacts_for_office() finds nothing
# for an office (not in _OFFICE_NAME_MAP, or its page failed to parse). Kept
# intentionally small. Extracted via Playwright 2026-05-02; 教務長 corrected
# 2026-08-25 (林啓屏 → 劉吉軒, confirmed via office_contacts_index.jsonl —
# 林啓屏 is now 副校長, not 教務長). Each entry: {name, ext, email, duty}.
KNOWN_CONTACTS: dict[str, list[dict]] = {
    "生僑組": [
        {"name": "盧誼甄", "ext": "62226", "email": "karena@nccu.edu.tw",    "duty": "休退學退費"},
        {"name": "黃湘妮", "ext": "63013", "email": "shani107@nccu.edu.tw",  "duty": "僑生業務"},
        {"name": "傅秀平", "ext": "62227", "email": "pingfu@nccu.edu.tw",    "duty": "陸生業務"},
        {"name": "陳韻帆", "ext": "62224", "email": "annachen@nccu.edu.tw",  "duty": "學雜費減免"},
        {"name": "洪子茹", "ext": "62223", "email": "n91074@nccu.edu.tw",    "duty": "獎助學金"},
    ],
    "住宿組": [
        {"name": "陳哲良", "ext": "62222", "email": "nccudorm@nccu.edu.tw",  "duty": "學士班新生宿舍"},
        {"name": "徐廷芝", "ext": "62228", "email": "dorm@nccu.edu.tw",      "duty": "學士班舊生宿舍"},
        {"name": "曾偉哲", "ext": "63251", "email": "grad_dorm@nccu.edu.tw", "duty": "碩博士宿舍"},
    ],
    "出納組": [
        {"name": "蕭毓璇", "ext": "62127", "email": "beru@nccu.edu.tw",      "duty": "註冊繳費"},
    ],
    "圖書館": [
        {"name": "中正圖書館",    "ext": "63222", "email": None, "duty": "借還書、離校手續（主館）"},
        {"name": "達賢圖書館",    "ext": "77017", "email": None, "duty": "借還書、離校手續（主館）"},
        {"name": "綜合院分館",    "ext": "50107", "email": None, "duty": "借還書、離校手續"},
        {"name": "商學院分館",    "ext": "84006", "email": None, "duty": "借還書、離校手續"},
        {"name": "傳播學院分館",  "ext": "67152", "email": None, "duty": "借還書、離校手續"},
    ],
    "國際合作事務處": [
        {"name": "國合處", "ext": "62040", "email": "oic@nccu.edu.tw",       "duty": "國際學生業務"},
    ],
    "國合處": [
        {"name": "國合處", "ext": "62040", "email": "oic@nccu.edu.tw",       "duty": "國際學生業務"},
    ],
    "教務處": [
        {"name": "劉吉軒", "ext": "62160", "email": None,                    "duty": "教務長"},
        {"name": "曹惠莉", "ext": "62163", "email": None,                    "duty": "秘書（教務長室）"},
        {"name": "黃婉綝", "ext": "63281", "email": "tr63279@nccu.edu.tw",   "duty": "學士班學籍統籌（註冊組）"},
        {"name": "王揚忠", "ext": "63273", "email": None,                    "duty": "註冊組組長"},
    ],
    "教務長": [
        {"name": "劉吉軒", "ext": "62160", "email": None,                    "duty": "教務長"},
        {"name": "曹惠莉", "ext": "62163", "email": None,                    "duty": "秘書"},
    ],
    "註冊組": [
        {"name": "黃婉綝", "ext": "63281", "email": "tr63279@nccu.edu.tw",   "duty": "學士班學籍統籌"},
        {"name": "王揚忠", "ext": "63273", "email": None,                    "duty": "組長"},
    ],
}

# (subdomain, grep_keyword) pairs to find the member/contact page per office
OFFICE_SOURCES = {
    "生僑組":         [("osa",  "休退學暨研究生畢業退費"),  # refund page has phone
                       ("osa",  "工作職掌")],              # member page has 承辦人
    "住宿組":         [("osa",  "休退學暨研究生畢業退費"),  # has 行政大樓三樓住宿組
                       ("osa",  "退離宿程序")],
    "出納組":         [("osa",  "休退學暨研究生畢業退費"),  # has 分機62123
                       ("aca",  "出納")],
    "國際合作事務處": [("osa",  "休退學暨研究生畢業退費"),  # has 國合處(分機62040)
                       ("oic",  "國合處聯絡方式")],       # Post/800 has floor + direct line
    "教務處":         [("osa",  "休退學暨研究生畢業退費"),  # has 行政大樓四樓教務處
                       ("aca",  "教務處")],
    "圖書館":         [("www.lib", "各組聯絡資訊"),   # contact page has direct lines + extensions
                       ("www.lib", "館長室")],         # director's page has ext 77072/62028
}

_EXT_RE   = re.compile(r'分機[：:\s]*(\d{4,5})')
_PHONE_RE = re.compile(r'886-2-(\d{4}-?\d{4})|02-(\d{4}-?\d{4})')

# Short aliases used in running text (e.g. osa refund page uses 國合處, not full name)
OFFICE_ALIASES: dict[str, list[str]] = {
    "國際合作事務處": ["國合處", "國際教育組"],
    "教務處":         ["教務處", "註冊組"],
}

# Office keywords used as boundaries when extracting phones
_OFFICE_BOUNDARIES = [
    "住宿組", "出納組", "生僑組", "國際合作", "國合處",
    "教務處", "圖書館", "系所", "原資中心",
]


def _extract_phones_near(text: str, office: str) -> list[str]:
    """
    Extract extensions for a specific office by scanning from each occurrence
    of the office name until the next office name, sentence break, or 150 chars.
    This prevents phones for adjacent offices (e.g. 國合處 62040) from being
    attributed to the current office (e.g. 生僑組 63013).
    """
    phones = []
    idx = 0
    while True:
        pos = text.find(office, idx)
        if pos == -1:
            break
        end = pos + 150
        # Stop at the next different office mention
        for boundary in _OFFICE_BOUNDARIES:
            if boundary == office:
                continue
            nxt = text.find(boundary, pos + len(office))
            if nxt != -1 and nxt < end:
                end = nxt
        # Also stop at sentence end
        for sep in ("。", "\n"):
            nxt = text.find(sep, pos + len(office))
            if nxt != -1 and nxt < end:
                end = nxt
        snippet = text[pos:end]
        phones.extend(_EXT_RE.findall(snippet))
        idx = pos + 1
    return list(dict.fromkeys(phones))


def _page_mentions_office(text: str, office: str) -> bool:
    """Check if page text mentions the office by name or any alias."""
    return office in text or any(a in text for a in OFFICE_ALIASES.get(office, []))


def _find_pages(office: str) -> list[dict]:
    """Return all matching pages for an office (up to 2)."""
    pages = []
    seen_urls: set[str] = set()
    for subdomain, keyword in OFFICE_SOURCES.get(office, []):
        results = grep_texts(keyword, subdomain=subdomain, max_results=5)
        for r in results:
            if r["url"] in seen_urls:
                continue
            page = get_page(r["url"])
            if "error" not in page and _page_mentions_office(page["text"], office):
                seen_urls.add(r["url"])
                pages.append(page)
                if len(pages) >= 2:
                    return pages
    return pages


class OfficeLookupSkill:
    """
    Usage:
        skill = OfficeLookupSkill()
        info = skill.run(["生僑組", "住宿組", "出納組", "教務處"])
        # info["生僑組"] = {"floor": "行政大樓 3 樓", "phones": [...], "note": "..."}
    """

    def run(self, offices: list[str]) -> dict[str, dict]:
        result = {}
        for office in offices:
            # Condition 4: dynamic lookup is primary; KNOWN_CONTACTS is the
            # fallback only when the dynamic index has nothing for this office.
            contacts = dynamic_contacts_for_office(office) or KNOWN_CONTACTS.get(office, [])
            entry: dict = {
                "office":    office,
                "floor":     KNOWN_FLOORS.get(office),
                "phones":    [],
                "contacts":  contacts,
                "page_url":  None,
                "note":      "",
            }

            pages = _find_pages(office)
            if not pages:
                result[office] = entry
                continue

            entry["page_url"] = pages[0]["url"]

            # Merge phones from all found pages (extensions first, then direct lines)
            all_phones: list[str] = []
            search_names = [office] + OFFICE_ALIASES.get(office, [])
            for page in pages:
                for name in search_names:
                    all_phones.extend(_extract_phones_near(page["text"], name))
                if not all_phones:
                    # Fallback: first direct phone number on page (e.g. oic Post/800)
                    matches = _PHONE_RE.findall(page["text"])
                    if matches:
                        all_phones.append(matches[0][0] or matches[0][1])
            # Supplement with extensions from KNOWN_CONTACTS if dynamic lookup found none
            if not all_phones and entry["contacts"]:
                all_phones = [c["ext"] for c in entry["contacts"] if c.get("ext")]
            entry["phones"] = list(dict.fromkeys(all_phones))

            # Look for responsible person (姓名 + 職掌 containing 休退學/退費)
            # Priority: 休退學 > 退費 (to avoid matching 學雜費退費 handlers)
            for page in pages:
                text = page["text"]
                blocks = re.split(r'職稱', text)
                best: tuple[int, str] = (99, "")  # (priority, note)
                for block in blocks:
                    if "姓名" not in block:
                        continue
                    priority = 99
                    if "休退學" in block:
                        priority = 0
                    elif "退費" in block and priority == 99:
                        priority = 1
                    if priority == 99:
                        continue
                    # Chinese name only (stop before spaces or English)
                    name_match = re.search(r'姓名\s*\n?\s*([一-鿿]{2,5})', block)
                    ext_match  = re.search(r'分機\s*\n?\s*(\d{4,5})', block)
                    if name_match and priority < best[0]:
                        name = name_match.group(1)
                        ext  = ext_match.group(1) if ext_match else ""
                        best = (priority, f"承辦人（休退學）：{name}" + (f"，分機 {ext}" if ext else ""))
                if best[1]:
                    entry["note"] = best[1]
                    break

            result[office] = entry
        return result

    def format_context(self, lookup_result: dict[str, dict]) -> str:
        """Return a compact text block suitable for LLM synthesis context."""
        lines = ["【各辦公室聯絡資訊】"]
        for office, info in lookup_result.items():
            parts = [office]
            if info["floor"]:
                parts.append(info["floor"])
            if info["phones"]:
                parts.append("分機 " + "／".join(info["phones"][:3]))
            if info["note"]:
                parts.append(f"（{info['note']}）")
            lines.append("  " + "　".join(parts))
            # Per-contact details (name + ext + email)
            for c in info.get("contacts", []):
                detail = f"    • {c['name']}（{c['duty']}）分機 {c['ext']}"
                if c.get("email"):
                    detail += f"  {c['email']}"
                lines.append(detail)
        return "\n".join(lines)


if __name__ == "__main__":
    skill = OfficeLookupSkill()
    offices = ["生僑組", "住宿組", "出納組", "國際合作事務處", "教務處"]
    result = skill.run(offices)

    for office, info in result.items():
        print(f"\n=== {office} ===")
        print(f"  floor    : {info['floor']}")
        print(f"  phones   : {info['phones']}")
        print(f"  contacts : {info['contacts']}")
        print(f"  page_url : {info['page_url']}")
        print(f"  note     : {info['note']}")

    print("\n--- Formatted context ---")
    print(skill.format_context(result))
