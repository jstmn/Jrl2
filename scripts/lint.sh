#!/bin/bash

uv run --active  ruff  check  src/jrl2/*.py --fix
uv run --active  ruff  check  tests/*.py --fix
uv run --active  ruff  check  scripts/*.py --fix

uv run --active  black src/jrl2/*.py --line-length 120
uv run --active  black tests/*.py --line-length 120
uv run --active  black scripts/*.py --line-length 120