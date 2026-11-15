"""The model's maths and the training loop, checked for real."""

import numpy as np

from opengpt import data, model, trainer


def _tiny(n_layer, n_head, seed=1):
    rng = np.random.default_rng(seed)
    V, T, C, H = 11, 5, 8, 16
    p = model.init_params(rng, V, T, C, H, n_layer)
    for k in p:                                   # no zeros or ones hiding a wrong gradient
        p[k] = p[k] + rng.normal(0, 0.3, p[k].shape)
    x = rng.integers(0, V, size=(2, T))
    y = rng.integers(0, V, size=(2, T))
    return p, x, y


def test_gradients_match_measured_slopes():
    for n_layer, n_head in [(1, 1), (1, 2), (2, 4), (3, 2)]:
        p, x, y = _tiny(n_layer, n_head)
        err = model.grad_check(p, x, y, n_head)
        assert err < 1e-6, (n_layer, n_head, err)


def test_one_head_one_layer_is_the_original_attention():
    p, x, y = _tiny(1, 1)
    _, _, c = model.forward(p, x, y, 1)
    bc = c["blocks"][0]
    q, k = bc["ln1"] @ p["L0.Wq"], bc["ln1"] @ p["L0.Wk"]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    mask = np.tril(np.ones((5, 5), bool))
    assert np.allclose(np.where(mask, scores, 0), np.where(mask, bc["scores"][:, 0], 0))


def test_causal_mask_hides_the_future():
    p, x, _ = _tiny(2, 2)
    a, _ = model.forward(p, x, None, 2)
    x2 = x.copy()
    x2[:, -1] = (x2[:, -1] + 1) % 11              # change only the last word
    b, _ = model.forward(p, x2, None, 2)
    assert np.allclose(a[:, :-1], b[:, :-1])      # earlier positions cannot see it
    assert not np.allclose(a[:, -1], b[:, -1])


def test_data_pipeline():
    ds = data.load("tinytales", 1)
    assert ds.words[0] == "<unk>" and ds.stats["unk_rate"] == 0
    assert len(ds.train) + len(ds.val) == len(ds.data)
    assert abs(len(ds.train) / len(ds.data) - 0.9) < 1e-3
    x, y = data.get_batch(ds.train, np.random.default_rng(0), 4, 8)
    assert (x[:, 1:] == y[:, :-1]).all()          # y is x moved one word on
    toks, ids = ds.encode_text("The cat sat. Zzzqq!")
    assert toks[:2] == ["the", "cat"] and ids[-2] == 0   # unknown word -> <unk>


def test_training_learns_and_holds_up_on_unseen_text():
    run = trainer.Run(dict(steps=300, eval_every=100, C=32, n_layer=1, n_head=2, T=12, seed=0),
                      run_id="_test")
    run.save = lambda: None                       # keep the test off the disk
    run._loop()
    first, last = run.evals[0], run.evals[-1]
    assert run.status == "done" and run.step == 300 and len(run.metrics) == 300
    assert first["val_loss"] > 0.9 * run.info["chance_loss"]        # starts at guessing
    assert last["val_loss"] < 0.5 * first["val_loss"]               # learned
    assert last["val_acc"] > run.info["baseline_acc"] + 0.2         # beats the dumb guess
    assert abs(last["train_acc"] - last["val_acc"]) < 0.08          # not memorising
    assert run.samples and run.samples[-1]["text"]


def test_pause_and_stop():
    import time
    run = trainer.Run(dict(steps=100000, eval_every=100000, C=16, n_layer=1, n_head=1, T=8),
                      run_id="_test2")
    run.save = lambda: None
    run.start()
    while run.step < 20:
        time.sleep(0.01)
    run.pause()
    time.sleep(0.1)
    s = run.step
    time.sleep(0.2)
    assert run.status == "paused" and run.step - s <= 1
    run.resume()
    while run.step < s + 20:
        time.sleep(0.01)
    run.stop()
    run.join(5)
    assert run.status == "stopped" and run.step < 100000


def test_bad_settings_are_refused():
    import pytest
    with pytest.raises(ValueError):
        trainer.clean_config(dict(C=64, n_head=5))
    with pytest.raises(ValueError):
        trainer.clean_config(dict(lr=0))
