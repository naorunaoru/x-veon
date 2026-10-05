#!/usr/bin/env bash
# The web build recipe: CI runs it for the stable and beta channels (deploy.yml, promote.yml),
# and you can run it locally. It builds the web host for <channel> into web/dist, from a clean install.
# Shared dependencies and host contracts are checked across all three workspaces.
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

npm ci --workspace shared --workspace web --workspace desktop
npm test --workspace shared
npm test --workspace web
npm test --workspace desktop
cargo test --locked -p xveon-encode
npm run build:wasm --workspace shared
npm run build:wasm --workspace web
npm run build:lensfun --workspace shared
XV_CHANNEL="$channel" npm run build --workspace web
npm run typecheck --workspace shared --workspace web --workspace desktop

# What the workflows' sanity checks look for.
test -f web/dist/index.html
grep -q "${base}assets/" web/dist/index.html
test -f web/dist/checkpoints/models.json
test -s web/dist/lensfun/index.json
grep -q '"file"' web/dist/lensfun/index.json
echo "web/dist is the ${channel} build"
