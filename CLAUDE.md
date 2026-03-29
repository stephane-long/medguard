# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**medguard** — Python 3.13, currently in initial scaffolding phase.

The repo also contains `Netino.html`, a standalone HTML/JS frontend (SmartAdmin + Bootstrap 3 + jQuery) that appears to be a reference UI template.

## Setup & Commands

This project uses `uv` (inferred from `pyproject.toml` and `.python-version`).

```bash
# Install dependencies
uv sync

# Run the main entry point
uv run python main.py
```

## Structure

- `main.py` — entry point, currently a stub `main()` function
- `pyproject.toml` — project metadata, Python ≥ 3.13, no dependencies yet
- `.python-version` — pins Python 3.13
- `Netino.html` — standalone HTML frontend template (SmartAdmin dashboard framework)
