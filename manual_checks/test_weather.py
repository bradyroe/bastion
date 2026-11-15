import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import json
import tools

for loc in ["Springfield, PA", "Springfield", "Brisbane", "London, UK", "Springfield, IL"]:
    result = tools.get_weather(loc)
    try:
        d = json.loads(result)
        print(f"{loc:20} -> {d['location']}  {d['temp_f']}F / {d['temp_c']}C  {d['condition']}")
    except json.JSONDecodeError:
        print(f"{loc:20} -> {result}")
