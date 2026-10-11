# WIP: model selector and 1x-4096x speed on both browser pages

Approved design (user request: "update both html and htmlcss game so you can select the model, and update the speed (1x-4096x)"):

1. `node trainer/cli.js export-models` writes `models.js` (global `DINO_MODELS`) and `models-css/<id>.css`; never touches the root `model.js`/`model.css`.
2. `index.html`: model select (?model=<id>, localStorage), speed select 1..4096 with a per-frame step budget and achieved-speed readout.
3. `index-css.html`: same model select (swaps the <link> to `models-css/<id>.css`), speed to 4096x, honest achieved speed.
4. `test/browser-check.js`: speed lists, per-model CSS/JS agreement, default = shipped model.
