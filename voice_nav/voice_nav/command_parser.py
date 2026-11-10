"""Phan tich cau lenh tieng Viet -> lo trinh cac khu vuc. Khong phu thuoc ROS nen test duoc rieng."""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List


def normalize(s: str) -> str:
    """Chu thuong, bo dau, bo ky tu la: 'Đi đến khu vực A.' -> 'di den khu vuc a'."""
    s = s.lower().replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


TRIGGERS = r"\b(?:tu|den|toi|ve|vao|qua|ra|to|from)\b"
CANCEL = re.compile(r"\b(?:dung lai|huy|dung|stop|cancel)\b")


@dataclass
class ParseResult:
    kind: str                      # navigate | cancel | unknown_zone | unrecognized
    route: List[dict] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)


class CommandParser:
    def __init__(self, zones: List[dict]):
        self.zones = zones
        self.ids = {normalize(z["id"]): z for z in zones}
        self.alias_tok = {}  # 'khu_vuc_a' -> zone
        for z in zones:
            names = [z["name"], *z.get("aliases", []), f"khu vuc {z['id']}"]
            for n in names:
                n = normalize(n)
                if n:
                    self.alias_tok[n.replace(" ", "_")] = z
        # alias dai truoc de tranh khop nham ('khu vuc a' truoc 'khu a')
        self.ordered = sorted(self.alias_tok, key=len, reverse=True)

    def _protect(self, t: str) -> str:
        for tok in self.ordered:
            t = re.sub(rf"\b{re.escape(tok.replace('_', ' '))}\b", tok, t)
        return t

    def _zone_of(self, phrase: str):
        toks = phrase.split()
        for tk in toks:
            if tk in self.alias_tok:
                return self.alias_tok[tk]
        if len(toks) == 1 and toks[0] in self.ids:  # "den a"
            return self.ids[toks[0]]
        return None

    def parse(self, text: str) -> ParseResult:
        t = self._protect(normalize(text))
        phrases = [p.strip() for p in re.split(TRIGGERS, t)[1:] if p.strip()]

        route, unknown = [], []
        for p in phrases:
            z = self._zone_of(p)
            if z:
                route.append(z)
            else:
                unknown.append(re.sub(r"^(?:hay |di |chuyen )+", "", p).replace("_", " "))

        if not phrases:  # cau khong co 'den/toi' -> thu tim khu vuc o bat ky dau
            route = [self.alias_tok[tk] for tk in t.split() if tk in self.alias_tok]

        if unknown:
            return ParseResult("unknown_zone", unknown=unknown)
        if route:
            dedup = [z for i, z in enumerate(route) if i == 0 or z["id"] != route[i - 1]["id"]]
            return ParseResult("navigate", route=dedup)
        if CANCEL.search(t):
            return ParseResult("cancel")
        return ParseResult("unrecognized")
