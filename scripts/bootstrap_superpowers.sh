#!/usr/bin/env bash
# Vendor obra/superpowers for offline skill reference (npm package does not exist).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/vendor/superpowers"
if [[ -d "${DEST}/.git" ]]; then
  echo "Superpowers already present at ${DEST}"
  exit 0
fi
mkdir -p "${ROOT}/vendor"
git clone --depth 1 https://github.com/obra/superpowers.git "${DEST}"
echo "Cloned Superpowers to ${DEST}. In Cursor, also install the marketplace plugin: /add-plugin superpowers"
