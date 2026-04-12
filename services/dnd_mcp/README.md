# dnd_mcp — D&D 5e MCP Service

A [FastMCP 3](https://gofastmcp.com) service that wraps the [`dnd-5e-core`](https://pypi.org/project/dnd-5e-core/) rules engine and exposes it as Model Context Protocol (MCP) tools. Designed as the AI backbone of the TTRPG project.

---

## Stack

| Layer | Technology |
|-------|-----------|
| MCP framework | FastMCP 3 (streamable-HTTP transport) |
| Rules engine | dnd-5e-core (100 % offline, 8.7 MB bundled data) |
| Observability | OpenTelemetry SDK — console exporter (dev) |
| Validation | Pydantic v2 |
| Runtime | Python ≥ 3.10 |

---

## Setup

```bash
# 1. Create & activate a virtual environment (use python3.10+ explicitly)
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install all dependencies
pip install -e ".[dev]"

# 3. (optional) Copy env template
cp .env.example .env
```

---

## Running the server

```bash
# Activate venv first if not already active
source .venv/bin/activate

python server.py
# → Listening on http://0.0.0.0:8000
# → MCP endpoint: http://localhost:8000/mcp/
```

OTEL spans are printed to **stdout** in JSON format on every tool call.

---

## Inspecting with MCP Inspector

```bash
# In a separate terminal (no venv needed — uses npx)
npx @modelcontextprotocol/inspector http://localhost:8000/mcp/
```

The Inspector UI opens in your browser. You can:
- Browse the three registered tools
- Call `dnd_create_character`, `dnd_load_monster`, `dnd_get_magic_item` interactively
- Verify OTEL spans appear in the server terminal

---

## Available Tools (Ticket 1.1)

| Tool | Description |
|------|-------------|
| `dnd_create_character` | Generate a full 5e character sheet (level, race, class, name) |
| `dnd_load_monster` | Load any of the 332 bundled monster stat blocks |
| `dnd_get_magic_item` | Retrieve properties for any of the 49 bundled magic items |

---

## Verification checklist (Ticket 1.1 AC)

- [ ] `python server.py` starts on port 8000 without errors
- [ ] `curl http://localhost:8000/mcp/` returns a valid MCP response
- [ ] MCP Inspector lists all three tools
- [ ] Calling a tool emits an OTEL span JSON block on stdout
- [ ] `dnd-5e-core` resolves correctly (no import errors)

---

## Project layout

```
services/dnd_mcp/
├── server.py          # FastMCP server + tool definitions
├── pyproject.toml     # Dependencies & build config
├── .env.example       # Environment variable template
└── README.md          # This file
```
