// What the two game pages (index.html via game.js, index-css.html via css-nn.js)
// share about which rules they play and which model they offer. Classic <script>
// (sets the global `DinoPages`; needs core.js first) and Node require().
//
// The pages always play the LATEST game rules: the last entry of
// DinoCore.GAME_VERSIONS (user order, 2026-10-12). There is no URL option and no
// UI to pick other rules, and the model selectors offer only models trained for
// these rules. A remembered or ?model= choice for other rules (or an unknown id)
// is ignored, a remembered one is also forgotten, and the page plays its root
// model (model.js / model.css). If the root model was trained for other rules, the
// page still plays the latest ones and shows rulesWarning(). The training dashboard
// does not use this file: it plays every run on that run's own rules.
//
// Research harnesses that copy a page into a scratch directory and must play a
// candidate trained for other rules replace the GAME_VERSION line below in THEIR
// copy (test/css-agreement.js does); the shipped pages never do.
(function (root) {
'use strict';

const Core = (typeof module !== 'undefined' && module.exports) ? require('./core.js') : root.DinoCore;

const GAME_VERSION = Core.GAME_VERSIONS[Core.GAME_VERSIONS.length - 1];

// localStorage key of the selected model id, shared by both pages.
const MODEL_KEY = 'dino-model';

// The entries (models.js or models-css/index.js) trained for the rules the pages play.
const forPages = (entries) => entries.filter(m => m.gameVersion === GAME_VERSION);

// The selected entry: ?model=<id> in `search`, else the remembered id, each used
// only if it names an entry for GAME_VERSION. A remembered id that doesn't is
// removed. Null means: play the root model.
function chooseModel(entries, search, getStorage = () => root.localStorage) {
  const usable = (id) => (id && forPages(entries).find(m => m.id === id)) || null;
  const asked = usable(new URLSearchParams(search).get('model'));
  let remembered = null;
  try {
    const storage = getStorage();
    const stored = storage.getItem(MODEL_KEY);
    remembered = usable(stored);
    if (stored !== null && !remembered) storage.removeItem(MODEL_KEY);
  } catch (e) { /* storage unavailable */ }
  return asked || remembered;
}

// A selector pick: remember it (or forget it for the root model, id '') and reload
// with only ?model=<id>.
function pickModel(id) {
  try {
    if (id) root.localStorage.setItem(MODEL_KEY, id);
    else root.localStorage.removeItem(MODEL_KEY);
  } catch (e) { /* storage unavailable */ }
  root.location.search = id ? new URLSearchParams({ model: id }).toString() : '';
}

// Short status-line warning when the model was trained for other rules ('' if not).
const rulesWarning = (modelVersion) => modelVersion && modelVersion !== GAME_VERSION
  ? `this model was trained for rules ${modelVersion}; the page plays rules ${GAME_VERSION}` : '';

const DinoPages = { GAME_VERSION, MODEL_KEY, forPages, chooseModel, pickModel, rulesWarning };

root.DinoPages = DinoPages;
if (typeof module !== 'undefined' && module.exports) module.exports = DinoPages;
})(typeof globalThis !== 'undefined' ? globalThis : this);
