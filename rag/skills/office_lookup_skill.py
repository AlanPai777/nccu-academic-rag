"""
rag/skills/office_lookup_skill.py
Per-office contact info lookup for NCCU procedure answers.

Resolves floor number, phone extensions, and responsible person for each
stop in a 7-stop procedure chain.

Data sources:
  1. KNOWN_CONTACTS — Playwright-extracted staff directory (2026-05-02)
  2. Dynamic grep → get_page for per-office member/contact pages
  3. KNOWN_FLOORS fallback for floors absent from crawled data (moltke-verified)
"""

import re

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

# Staff contacts extracted via Playwright 2026-05-02.
# Each entry: {name, ext, email, duty}. Government emails rarely change.
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
        {"name": "林啓屏", "ext": "62160", "email": None,                    "duty": "教務長"},
        {"name": "曹惠莉", "ext": "62163", "email": None,                    "duty": "秘書（教務長室）"},
        {"name": "黃婉綝", "ext": "63281", "email": "tr63279@nccu.edu.tw",   "duty": "學士班學籍統籌（註冊組）"},
        {"name": "王揚忠", "ext": "63273", "email": None,                    "duty": "註冊組組長"},
    ],
    "教務長": [
        {"name": "林啓屏", "ext": "62160", "email": None,                    "duty": "教務長"},
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
            entry: dict = {
                "office":    office,
                "floor":     KNOWN_FLOORS.get(office),
                "phones":    [],
                "contacts":  KNOWN_CONTACTS.get(office, []),
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
