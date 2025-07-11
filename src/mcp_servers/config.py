import json
from pathlib import Path

config_file = Path(__file__).parent / "mcp_config.json"
if not config_file.exists():
    raise FileNotFoundError(f"mcp_config.json file {config_file} does not exist")

with open(config_file, "r") as f:
    config = json.load(f)

mcp_config = config["mcpServers"]