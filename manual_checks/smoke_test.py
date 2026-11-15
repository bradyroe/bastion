"""Quick smoke-test — run from the bastion directory."""
import json
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import config
import security
import tools
from security import SecurityError, safe_read_path, safe_string

print("1. imports OK")

# Security: path traversal
try:
    safe_read_path("C:/Windows/System32/bad.txt")
    print("FAIL: path traversal not blocked")
except SecurityError:
    print("2. path-traversal blocked OK")

# Security: length limit
try:
    safe_string("x" * 9999, 10, "test")
    print("FAIL: length limit not enforced")
except SecurityError:
    print("3. length-limit blocked OK")

# Ollama connection
import ollama
available = [m.model for m in ollama.list().models]
print(f"4. Ollama models found: {available}")
print(f"5. Configured model available: {config.MODEL in available}")

# DB init
tools.init_db()
print("6. DB init OK")

# System status tool
result = tools.get_system_status()
data = json.loads(result)
gpu = data.get("gpu_name", "not detected")
print(f"7. System status OK — CPU {data['cpu_pct']}%  RAM {data['ram_pct']}%  GPU: {gpu}")

# Weather tool. Uses the CONFIGURED location, not a hardcoded city: this line
# used to read get_weather("Brisbane") and the Australian result in the output
# was mistaken for IP geolocation overriding config. It was just the test's own
# argument.
wx = tools.get_weather(config.DEFAULT_LOCATION)
print(f"8. Weather tool OK — {wx[:100]}")
assert config.DEFAULT_LOCATION.split(",")[0] in wx, (
    f"weather came back for somewhere other than {config.DEFAULT_LOCATION}: {wx[:200]}"
)

# Memory tool
tools.remember("smoke_test", "passed")
recalled = tools.recall("smoke_test")
assert "passed" in recalled, "recall failed"
print("9. Memory (remember/recall) OK")

print("\nAll checks passed. Run bastion.py to start.")
