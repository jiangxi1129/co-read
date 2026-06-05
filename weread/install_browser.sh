#!/usr/bin/env bash
# Install Playwright Chromium + system deps for weread chapter scraping.
# Run once after deploying.
set -e
cd "$(dirname "$0")"

# Activate venv if user passed one, otherwise assume current python is correct
if [ -n "$1" ]; then
    source "$1/bin/activate"
fi

pip install -r requirements.txt
python -m playwright install chromium
python -m playwright install-deps chromium
echo "weread browser ready"
