#!/bin/sh
# Build a scratch copy of the CSS-network page (index-css.html) that loads a given model.css.
# Copies the page and its scripts from the main checkout; never touches the root files.
# Usage: make-page.sh <model.css> <page dir>
set -eu
ROOT=/Users/kyle/Desktop/dino-game
CSS=$1; OUT=$2
mkdir -p "$OUT"
for f in index-css.html style.css core.js pages.js render.js policy.js css-nn.js; do cp "$ROOT/$f" "$OUT/$f"; done
cp "$CSS" "$OUT/model.css"
# The shipped page always plays the latest rules (pages.js GAME_VERSION); this scratch
# copy plays the rules the model.css was trained for (its --nn-game; these are rules v2).
GAME=$(sed -n 's/.*--nn-game: "\([a-z0-9]*\)".*/\1/p' "$OUT/model.css" | head -n 1)
[ -n "$GAME" ] || { echo "no --nn-game in $CSS" >&2; exit 1; }
sed "s/^const GAME_VERSION = .*;\$/const GAME_VERSION = '$GAME';/" "$ROOT/pages.js" > "$OUT/pages.js"
grep -q "^const GAME_VERSION = '$GAME';\$" "$OUT/pages.js" || { echo "could not set the rules in $OUT/pages.js" >&2; exit 1; }
echo "page $OUT ($(wc -c < "$OUT/model.css" | tr -d ' ') B model.css)"
