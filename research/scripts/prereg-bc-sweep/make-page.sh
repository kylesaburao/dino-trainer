#!/bin/sh
# Build a scratch copy of the CSS-network page (index-css.html) that loads a given model.css.
# Copies the page and its scripts from the main checkout; never touches the root files.
# Usage: make-page.sh <model.css> <page dir>
set -eu
ROOT=/Users/kyle/Desktop/dino-game
CSS=$1; OUT=$2
mkdir -p "$OUT"
for f in index-css.html style.css core.js render.js policy.js css-nn.js; do cp "$ROOT/$f" "$OUT/$f"; done
cp "$CSS" "$OUT/model.css"
echo "page $OUT ($(wc -c < "$OUT/model.css" | tr -d ' ') B model.css)"
