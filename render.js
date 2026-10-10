// Canvas renderer for a DinoCore game: sprites, font, ground texture and the
// frame drawing. Shared by the game page (game.js) and the training dashboard's
// live playback, so both draw exactly the same picture. Classic <script> (sets
// the global `DinoRender`; needs core.js first) and Node require() (no DOM is
// touched until createRenderer() is called).
(function (root) {
'use strict';

const Core = (typeof module !== 'undefined' && module.exports) ? require('./core.js') : root.DinoCore;
const { W, H, GROUND, CFG, SPRITES } = Core;
const DinoCore = Core;
const INK = '#535353';

// Pixel font (5x6 glyphs)
const FONT = {
  0: ['.###.', '#...#', '#...#', '#...#', '#...#', '.###.'],
  1: ['..#..', '.##..', '..#..', '..#..', '..#..', '.###.'],
  2: ['.###.', '#...#', '...#.', '..#..', '.#...', '#####'],
  3: ['####.', '....#', '.###.', '....#', '....#', '####.'],
  4: ['#..#.', '#..#.', '#####', '...#.', '...#.', '...#.'],
  5: ['#####', '#....', '####.', '....#', '....#', '####.'],
  6: ['.###.', '#....', '####.', '#...#', '#...#', '.###.'],
  7: ['#####', '....#', '...#.', '..#..', '.#...', '.#...'],
  8: ['.###.', '#...#', '.###.', '#...#', '#...#', '.###.'],
  9: ['.###.', '#...#', '#...#', '.####', '....#', '.###.'],
  H: ['#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  I: ['#####', '..#..', '..#..', '..#..', '..#..', '#####'],
  G: ['.####', '#....', '#..##', '#...#', '#...#', '.###.'],
  A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#'],
  M: ['#...#', '##.##', '#.#.#', '#...#', '#...#', '#...#'],
  E: ['#####', '#....', '####.', '#....', '#....', '#####'],
  O: ['.###.', '#...#', '#...#', '#...#', '#...#', '.###.'],
  V: ['#...#', '#...#', '#...#', '#...#', '.#.#.', '..#..'],
  R: ['####.', '#...#', '####.', '#.#..', '#..#.', '#...#'],
};

// createRenderer(canvas) builds the sprite canvases and the ground texture (which
// uses Math.random: it is purely visual) and sizes the canvas for the device
// pixel ratio. Returns { draw(game, now, hi) }.
function createRenderer(canvas) {
  // -------------------------------------------------------------------------
  // Sprites: '#' = ink pixel, each map pixel is 2x2 on screen
  // -------------------------------------------------------------------------
  function sprite(rows, color = INK, s = DinoCore.PIXEL) {
    const w = Math.max(...rows.map(r => r.length));
    const c = document.createElement('canvas');
    c.width = w * s;
    c.height = rows.length * s;
    const g = c.getContext('2d');
    g.fillStyle = color;
    rows.forEach((row, y) => {
      for (let x = 0; x < row.length; x++) {
        if (row[x] === '#') g.fillRect(x * s, y * s, s, s);
      }
    });
    return c;
  }

  const S = {
    stand: sprite(SPRITES.stand),
    blink: sprite(SPRITES.blink),
    crash: sprite(SPRITES.crash),
    run: SPRITES.run.map(r => sprite(r)),
    duck: SPRITES.duck.map(r => sprite(r)),
    cactusSmall: sprite(SPRITES.cactusSmall),
    cactusLarge: sprite(SPRITES.cactusLarge),
    ptero: SPRITES.ptero.map(r => sprite(r)),
    cloud: sprite(SPRITES.cloud, '#dadada'),
  };
  const OBSTACLE_FRAMES = {};
  for (const t of DinoCore.OBSTACLE_TYPES) OBSTACLE_FRAMES[t.name] = [].concat(S[t.sprite]);

  const GLYPHS = {};
  for (const ch in FONT) GLYPHS[ch] = sprite(FONT[ch]);
  const GLYPH_W = 10;

  function makeGround() {
    const c = document.createElement('canvas');
    c.width = DinoCore.GROUND_TILE_W;
    c.height = 12;
    const g = c.getContext('2d');
    g.fillStyle = INK;
    for (let x = 0; x < c.width;) {
      if (Math.random() < 0.012 && x < c.width - 8) {
        g.fillRect(x, 1, 1, 1);
        g.fillRect(x + 1, 0, 4, 1);
        g.fillRect(x + 5, 1, 1, 1);
        x += 6;
        continue;
      }
      g.fillRect(x, 2, 1, 1);
      if (Math.random() < 0.05) g.fillRect(x, 4 + Math.floor(Math.random() * 6), Math.random() < 0.5 ? 1 : 2, 1);
      x++;
    }
    return c;
  }
  const groundImg = makeGround();

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.imageSmoothingEnabled = false;

  let game, dino, hi;

  function drawText(str, x, y, gap = 2) {
    for (const ch of str) {
      if (GLYPHS[ch]) ctx.drawImage(GLYPHS[ch], x, y);
      x += GLYPH_W + gap;
    }
  }
  const textWidth = (str, gap = 2) => str.length * (GLYPH_W + gap) - gap;
  const SCORE_DIGITS = 6;                               // display caps at 999999
  const SCORE_MAX = 10 ** SCORE_DIGITS - 1;
  const ZEROS = '0'.repeat(SCORE_DIGITS);
  const pad = (n) => String(Math.min(n, SCORE_MAX)).padStart(SCORE_DIGITS, '0');

  function drawNightSky() {
    const nightAlpha = game.nightAlpha;
    if (nightAlpha <= 0) return;
    const moonX = game.moonX;
    ctx.globalAlpha = nightAlpha;
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(moonX, 35, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(moonX + 5, 32, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = INK;
    for (const s of game.stars) {
      ctx.fillRect(s.x, s.y - 2, 1, 5);
      ctx.fillRect(s.x - 2, s.y, 5, 1);
    }
    ctx.globalAlpha = 1;
  }

  function drawScore(now) {
    const right = W - 10;
    let shown = pad(game.score());
    const flashStart = game.flashStart;
    if (flashStart >= 0) {
      const t = now - flashStart;
      if (t < CFG.flashMs * 2 * CFG.flashTimes) {
        shown = Math.floor(t / CFG.flashMs) % 2 === 0 ? '' : pad(game.flashScore);
      }
    }
    const scoreX = right - textWidth(ZEROS);
    drawText(shown, scoreX, 6);
    if (hi > 0) {
      ctx.globalAlpha = 0.8;
      drawText('HI ' + pad(hi), scoreX - 20 - textWidth('HI ' + ZEROS), 6);
      ctx.globalAlpha = 1;
    }
  }

  function drawGameOver() {
    const label = 'GAME OVER';
    drawText(label, Math.round((W - textWidth(label, 8)) / 2), 40, 8);

    const cx = W / 2, cy = 80, r = 11;
    const a0 = -Math.PI * 0.35, a1 = Math.PI * 1.45;
    ctx.strokeStyle = INK;
    ctx.fillStyle = INK;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(cx, cy, r, a0, a1);
    ctx.stroke();
    // arrowhead at the end of the arc, pointing clockwise
    const px = cx + r * Math.cos(a1), py = cy + r * Math.sin(a1);
    const tx = -Math.sin(a1), ty = Math.cos(a1);
    const nx = Math.cos(a1), ny = Math.sin(a1);
    ctx.beginPath();
    ctx.moveTo(px + tx * 7, py + ty * 7);
    ctx.lineTo(px + nx * 6, py + ny * 6);
    ctx.lineTo(px - nx * 6, py - ny * 6);
    ctx.closePath();
    ctx.fill();
  }

  function drawFrame(now) {
    const state = game.state;
    ctx.clearRect(0, 0, W, H);
    drawNightSky();

    for (const c of game.clouds) ctx.drawImage(S.cloud, Math.round(c.x), Math.round(c.y));

    const gx = Math.round(game.groundX);
    ctx.drawImage(groundImg, -gx, GROUND - 4);
    ctx.drawImage(groundImg, groundImg.width - gx, GROUND - 4);

    for (const o of game.obstacles) {
      const img = OBSTACLE_FRAMES[o.t.name][o.frame];
      for (let i = 0; i < o.size; i++) ctx.drawImage(img, Math.round(o.x) + i * img.width, o.y);
    }

    let img;
    if (dino.crashed) img = S.crash;
    else if (dino.ducking) img = S.duck[dino.frame];
    else if (state === 'running' && !dino.jumping) img = S.run[dino.frame];
    else img = now < dino.blinkUntil ? S.blink : S.stand;
    ctx.drawImage(img, dino.x, dino.ducking ? GROUND - img.height : Math.round(dino.y));

    if (state !== 'idle') drawScore(now);
    if (state === 'over') drawGameOver();
  }

  return {
    draw(g, now, highScore) {
      game = g;
      dino = g.dino;
      hi = highScore;
      drawFrame(now);
    },
  };
}

const DinoRender = { INK, createRenderer };

root.DinoRender = DinoRender;
if (typeof module !== 'undefined' && module.exports) module.exports = DinoRender;
})(typeof globalThis !== 'undefined' ? globalThis : this);
