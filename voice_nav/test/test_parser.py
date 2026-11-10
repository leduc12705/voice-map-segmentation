from voice_nav.command_parser import CommandParser

ZONES = [
    {"id": "A", "name": "Khu vực A", "aliases": ["khu vực a", "khu a"]},
    {"id": "B", "name": "Khu vực B", "aliases": []},
    {"id": "C", "name": "Phòng thí nghiệm", "aliases": ["phòng thí nghiệm", "phòng lab"]},
]
P = CommandParser(ZONES)
ids = lambda r: [z["id"] for z in r.route]


def test_single():
    assert ids(P.parse("Đi đến khu vực A.")) == ["A"]


def test_from_to():
    assert ids(P.parse("Di chuyển từ khu vực A đến khu vực B")) == ["A", "B"]


def test_named():
    assert ids(P.parse("đi đến phòng thí nghiệm")) == ["C"]
    assert ids(P.parse("hãy tới phòng lab")) == ["C"]


def test_no_diacritics_asr():
    assert ids(P.parse("di den khu vuc b")) == ["B"]


def test_unknown():
    r = P.parse("đi đến khu vực Z")
    assert r.kind == "unknown_zone" and r.unknown
    assert P.parse("đi đến phòng giám đốc").kind == "unknown_zone"


def test_partial_unknown_does_not_move():
    assert P.parse("đi đến khu vực A rồi đến khu vực Z").kind == "unknown_zone"


def test_cancel_and_garbage():
    assert P.parse("dừng lại").kind == "cancel"
    assert P.parse("hôm nay trời đẹp").kind == "unrecognized"
