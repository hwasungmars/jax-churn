#!/usr/bin/env bash
# -*- coding: utf8 -*-

# Author: Hwasung Lee

set -eux

script_dir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)

uv run ruff check --fix
uv run ruff format
uv run ty check
