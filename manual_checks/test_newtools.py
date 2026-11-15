"""Smoke-test the new Phase 4 tools."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import tools

tests = [
    ("calculate",        lambda: tools.calculate("15 * (4 + 2)")),
    ("calculate (safe)", lambda: tools.calculate("__import__('os')")),  # must be blocked
    ("wikipedia",        lambda: tools.wikipedia("Iron Man")[:80]),
    ("news_headlines",   lambda: tools.news_headlines("technology")[:80]),
    ("convert_currency", lambda: tools.convert_currency(100, "USD", "EUR")),
    ("define_word",      lambda: tools.define_word("serendipity")[:80]),
    ("get_directions",   lambda: tools.get_directions("Springfield, PA", "Philadelphia, PA")),
    ("fetch_url",        lambda: tools.fetch_url("https://example.com")[:80]),
    ("fetch_url (SSRF)", lambda: tools.fetch_url("http://127.0.0.1:8765")),  # must be blocked
    ("open_url (bad)",   lambda: tools.open_url("file:///C:/Windows/system.ini")),  # must be blocked
    ("clipboard write",  lambda: tools.clipboard("write", "bastion test")),
    ("clipboard read",   lambda: tools.clipboard("read")[:60]),
]

for name, fn in tests:
    try:
        result = fn()
        print(f"[{name:18}] {result}")
    except Exception as e:
        print(f"[{name:18}] EXCEPTION: {e}")

print("\nNote: media_control / set_volume not auto-tested (they'd change your volume).")
