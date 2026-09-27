#!/usr/bin/env bash
# The web build recipe: CI runs it for the stable and beta channels (deploy.yml, promote.yml),
# and you can run it locally. It builds the web host for <channel> into web/dist, from a clean install.
# It touches only the shared and web workspaces, so desktop work can't break a web deploy.
set -euo pipefail

channel="${1:-}"
case "$channel" in
  stable) base="/x-veon/" ;;
  beta)   base="/x-veon/beta/" ;;
  dev)    base="/" ;;
  *) echo "usage: scripts/build-web.sh <stable|beta|dev>" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
export ELECTRON_SKIP_BINARY_DOWNLOAD=1

npm ci --workspace shared --workspace web
npm test --workspace shared
npm run build:wasm --workspace shared
npm run build:lensfun --workspace shared
XV_CHANNEL="$channel" npm run build --workspace web
npm run typecheck --workspace shared --workspace web

# What the workflows' sanity checks look for.
test -f web/dist/index.html
grep -q "${base}assets/" web/dist/index.html
test -f web/dist/checkpoints/models.json
test -s web/dist/lensfun/index.json
grep -q '"file"' web/dist/lensfun/index.json
echo "web/dist is the ${channel} build"
