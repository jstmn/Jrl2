#!/bin/bash

uv run pytest -W "ignore::DeprecationWarning" -W "ignore::UserWarning" --capture=no
