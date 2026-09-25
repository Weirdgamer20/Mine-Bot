#!/usr/bin/env bash
set -e

cd /mnt/d/minecraft_learning_bot/agent
export PYTHONPATH=.:..
exec /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.runtime --mode stream --port 9099
