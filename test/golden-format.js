'use strict';
// Shared trace format for the golden recorder (baseline 2da34de) and the parity test.
// Kept tiny and free of game logic: it only names the fields both sides snapshot.
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const GOLDEN_DIR = path.join(__dirname, 'golden');
const STATE_CODES = { idle: 0, running: 1, over: 2 };
const TYPE_CODES = { small: 0, large: 1, ptero: 2 };

// Per-frame record:
// [state, dino.y, dino.vy, jumping, ducking, speedDrop, speed, distance, rngCalls,
//  nObstacles, ...per obstacle (type, x, y, size, w)]
function snapshot(s) {
  const row = [
    STATE_CODES[s.state], s.dino.y, s.dino.vy, s.dino.jumping ? 1 : 0, s.dino.ducking ? 1 : 0,
    s.dino.speedDrop ? 1 : 0, s.speed, s.distance, s.rngCalls, s.obstacles.length,
  ];
  for (const o of s.obstacles) row.push(TYPE_CODES[o.t.name], o.x, o.y, o.size, o.w);
  return row;
}

const FIELDS = ['state', 'dino.y', 'dino.vy', 'jumping', 'ducking', 'speedDrop', 'speed', 'distance', 'rngCalls', 'nObstacles'];
function fieldName(i) {
  if (i < FIELDS.length) return FIELDS[i];
  const k = i - FIELDS.length;
  return `obstacle[${Math.floor(k / 5)}].${['type', 'x', 'y', 'size', 'w'][k % 5]}`;
}

// Inputs are one char per frame: bit 0 = jump held, bit 1 = down held.
const decodeKeys = (ch) => { const c = ch.charCodeAt(0) - 48; return { jump: !!(c & 1), down: !!(c & 2) }; };

function writeTrace(trace) {
  fs.mkdirSync(GOLDEN_DIR, { recursive: true });
  fs.writeFileSync(path.join(GOLDEN_DIR, trace.name + '.json.gz'), zlib.gzipSync(JSON.stringify(trace), { level: 9 }));
}

function readTraces() {
  return fs.readdirSync(GOLDEN_DIR).filter(f => f.endsWith('.json.gz')).sort()
    .map(f => JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(GOLDEN_DIR, f))).toString()));
}

module.exports = { snapshot, fieldName, decodeKeys, writeTrace, readTraces, GOLDEN_DIR };
