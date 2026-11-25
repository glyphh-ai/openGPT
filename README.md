# openGPT

A GPT built from scratch in plain numpy, with a local app to learn how it works,
change it, and watch it train.

The model is word-level, with hand-written backpropagation and Adam: no autograd,
no framework. It grew out of `timmetim/thinktank` (`version_0_base`, one block and
one head, and `version_14/15`, multi-head attention) and now takes any number of
blocks and heads.

## Run it

```
uv sync
uv run python -m opengpt          # then open http://127.0.0.1:8642
```

It works straight away on `tinytales`, a small made-up story corpus generated on
first use. For the real thing:

```
uv run python -m opengpt fetch    # TinyStories validation split, about 19 MB
```

## The tabs

| Tab | What it is for |
| --- | --- |
| Learn | How a GPT works in 34 steps. Each step shows the math, the Python function, and one tiny sentence pushed through that function. |
| Data | What the model reads: corpus numbers, the rare-word cut, the 90/10 hold-out, and a tokenizer to try. |
| Train | Settings, Start / Pause / Stop, and a live dashboard: loss, accuracy on held-out text, gradient norm, learning rate, samples, and the four checks that show it learned. |
| Generate | Give a trained run a start; see what it writes, how sure it was of each word, and what every attention head looked at. |
| Runs | Every run is saved; lay their held-out loss on one chart. |
| Code | The source as it is on disk, and a button to re-check the gradients. |

## Doing research with it

The whole model is `opengpt/model.py`, one function per step. Edit it in your
editor: the next run reloads the file, and the Learn tab reads its functions and
re-runs its example from the code as it stands, so the lesson follows your change.
After touching the maths, check the gradients (Code tab, or the command below).
Then run the same settings before and after and compare them in Runs.

```
uv run python -m opengpt check                       # hand-written gradients vs measured slopes
uv run python -m opengpt train steps=2000 n_layer=2  # train in the terminal, any setting as key=value
uv run pytest                                        # maths, training loop, every app call
uv run python tests/ui_check.py                      # draw every tab in headless Chrome, no server
```

## Layout

```
opengpt/model.py        the GPT: forward, backward, Adam, one function per step
opengpt/data.py         tokenize, vocabulary, hold-out, batches
opengpt/trainer.py      runs you can start, pause, stop and watch; saved in runs/
opengpt/walkthrough.py  builds the Learn tab from the live code
opengpt/api.py          every call the app makes, as plain functions
opengpt/server.py       a small local web server and the live stream
opengpt/static/         the page (plain JavaScript, Chart.js)
```

## A reference point

The original single-block, single-head model on TinyStories
(`corpus=tinystories min_freq=10 T=32 C=128 n_layer=1 n_head=1 steps=700`) reaches
a held-out loss of 3.89 and 26.5% next-word accuracy in about 35 seconds on a
laptop CPU, the same as `thinktank/version_0_base` records.


## License

Apache-2.0. See LICENSE and NOTICE. Copyright 2026 Chris Timmerman / Glyphh AI LLC. More about it: https://glyphh.ai/projects/opengpt
