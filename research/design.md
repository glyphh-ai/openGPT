# Settle: stored states as memories

## The idea

A model's weights are used the same way on every pass; what differs is the path a
sequence takes through them. Record that path once (the residual stream at every
layer for an input and the output it gave) and treat it as a memory. When a related
prompt comes in, run the same frozen weights again and again and let the output
resettle onto the recorded path: a fixed point, x* = f(x*), pulled toward a stored
state, with no training.

Close relatives, so it is not claimed as new: Hopfield memory (stored patterns as
attractors; attention is one), deep equilibrium models and looped transformers
(the same weights run to a fixed point), function vectors, in-context vectors and
activation addition (recorded states reused on new prompts). What may be new is the
whole combination: a library of recorded paths, each one pulling a related prompt's
forward passes until they settle.

## What is built

`research/settle.py`, on run `20261115-091000` (tinytales, 2 blocks, 4 heads, T=16):

- record: one forward pass on input + output, keep every layer's states;
- gate: cosine (average state removed) between the prompt's states and each
  record's input states; below tau nothing fires;
- settle: lay the output down left to right, then sweep it word by word, on
  `log p(output) + beta * similarity(path(prompt + output), recorded path)`,
  re-running the forward pass each time, until nothing changes.

First result (2026-11-22): three memories recorded once each ("one day sam found a
red ball", "mia hid a old cake", "tom lost a blue key"). On fresh stories ending
"one day NAME", at beta 80: the recorded output 69 of 87 times (the model alone 0),
6 of 153 other names wrongly given a memory, "there sam" wording carried over, a
different slot ("sam was very") left alone. Limits: beta was picked on the test
results, and once the gate fires the pull mostly copies the recorded words, so a
lookup on the name would score about the same.

## Bigger test: a memory in a slot and wording it was never recorded in

`uv run python research/settle.py bend`. The same three memories; any output word
may now reinstate any recorded word's state, and the gate looks for the cue across
the whole 16-word window (with only the last 8 words the cue was out of view and
the reorder test scored 0). tau and beta are chosen on calibration prompts, then
scored on fresh ones (2026-11-22):

| test | prompt ends | memory given (model alone) | other names pushed onto a memory |
| --- | --- | --- | --- |
| slot | `sam was very sad . sam showed the` -> ball | 100 of 100 (31) | 0 of 140 |
| reorder | `the father said , " what a` -> red ball | 94 of 99 (0) | 24 of 141 |
| conflict | `one day sam found a green hat . ... sam showed the` | memory 64, story's own thing 20, of 109 | 0 of 131 |

So the memory does change shape: the recorded "found a red ball" answers a
different question ("showed the ...") and comes out in a different frame ("what a
red ball"), from forward passes only. What is not good yet: the wide gate leaks in
the reorder test (17% of other names), and when the story itself says something
else, the memory overrides it most of the time. It still only supplies the
recorded words in new places; a memory that has to combine with new content (not
just move) is untested.

Check against a plain lookup (if a recorded name is in view, add 10 to the log
probability of that record's words; no states at all): slot 100 of 100, reorder 99
of 99, conflict 109 of 109, and 0 other names pushed in any test. The lookup beats
settling everywhere, so these tests do not yet show the recorded states doing
anything a dictionary on the name cannot. The next test has to be one the lookup
fails: a cue that is not the same word (a description of sam instead of his name,
a nickname the model has seen used the same way), so the memory can only be found
through the states.

## Honest test (protocol written before running, 2026-11-22)

The tests above let a word lookup pass, and beta was first picked on test results.
This one is fixed in advance and run once.

**Question.** Do recorded states recall a memory when the cue is not one word that
a lookup can match: when it has to be the character who is acting, in a story that
has other characters in it?

**Corpus** (`research/rolestales.py`, made up): no names; two characters per story,
each "the ADJ ANIMAL", either one may act, e.g. `the kind bear and the shy fox lived
in a park . the shy fox saw the kind bear . one day the kind bear found a red ball .`
A base model is trained on it in the ordinary way (2 blocks, 4 heads, C=64, T=32);
the memories themselves get no training.

**Memories.** Six characters, each recorded once with a fixed find, in a story of
its own: `... one day the kind bear found a red ball`. The output judged is the two
words after `found a` (colour, thing).

**Probe sets** (fresh stories, never the recorded ones):
- A, actor remembered, other character not: the right answer is the actor's memory.
- B, other character remembered, actor not: the right answer is no memory.
- C, both remembered: the right answer is the actor's memory, not the other's.
- D, actor not remembered but shares its adjective with one memory and its animal
  with another (`the kind fox` when `the kind bear` and `the shy fox` are memories):
  the right answer is no memory.

**Systems.** Model alone; bag-of-words retrieval (generic, what RAG would do): each
memory scores the count of its description words in view, the best one fires above
a threshold and its two words get +10 log probability; settle (as built, words lined
up with the recorded slot, gate on the state before the output).

Added before running: a second bag-of-words that breaks ties toward the memory
mentioned last in view (a recency rule), since in C both memories match equally;
the pass bar is against the stronger of the two.

**Base model**, chosen on a check of the base alone, before any memory test: after
`the ADJ` the model must name that character's animal. 2 blocks x 64 wide: 80.4%;
3 blocks x 128 wide, 6,000 steps (run `20261122-061558`): 94.0%. The test uses the
second.

**Score.** Mean of A and C accuracy minus mean of the B and D rates of giving a
memory. Thresholds, tau and beta are picked on a calibration set (other seeds) to
maximise that score, then the test set is scored once.

**Pass bar.** Settle beats bag-of-words by at least 10 points and beats the model
alone. Anything less is reported as a fail.

**Result** (`uv run python research/honest.py`, run once, 400 fresh probes;
calibration picked bag-of-words threshold 2, settle beta 20, tau 0.960):

| system | score | A | C | B wrongly | D wrongly |
| --- | --- | --- | --- | --- | --- |
| model alone | +0.0% | 7% | 3% | 4% | 6% |
| bag-of-words | +21.0% | 100% | 48% | 100% | 6% |
| bag-of-words + recency | +47.0% | 100% | 100% | 100% | 6% |
| settle | +15.5% | 25% | 16% | 4% | 6% |

**FAIL**: settle needed +57.0% and scored +15.5%. The one thing it did that the
lookups did not: it never handed a bystander's memory to the character who acted
(B stayed at the model's own 4%, where both lookups gave it 100% of the time). But
it recalled the actor's memory only 25% and 16% of the time, so the stored states
tell the characters apart and still mostly fail to bring the memory back.

## Iterating (rules set before the first change)

The probe sets, systems, score and pass bar above stay fixed. The seed-2002 set has
now been seen, so it becomes the dev set: changes to settle are tried on calibration
(seed 1001) and dev (2002) only. A new locked test set (seed 3003, 100 per set) is
run once, only when a version passes the bar on dev, and that run is the result.
If no version passes after a fair run of ideas, the idea is dropped.

## Change 1: no gate (written before running)

Chris: "there is no gate, the overlap on the two always fire." The gate was my
addition, not the idea. So every memory always pulls, in proportion to how much its
recorded path overlaps the prompt's path:

    E(O) = log p(O) + beta * sum_r w_r * sim(path(prompt + O), recorded output path r)
    w_r  = softmax_r( overlap(prompt path, recorded input path r) / temp )

The overlap is the same measure the gate used (cosine, average state removed, every
layer, the state just before the output), so the only change is that the hard
threshold becomes a soft weight on every memory. beta and temp are picked on the
calibration set (seed 1001); the score is from the locked test set (seed 3003),
never seen, run once, with both bag-of-words baselines run on the same set. Same
score and pass bar as above.

**Result** (`uv run python research/soft.py`, locked test seed 3003, run once;
calibration picked beta 80, temp 0.3):

| system | score | A | C | B wrongly | D wrongly |
| --- | --- | --- | --- | --- | --- |
| model alone | -4.5% | 5% | 4% | 6% | 12% |
| bag-of-words | +21.0% | 100% | 54% | 100% | 12% |
| bag-of-words + recency | +44.0% | 100% | 100% | 100% | 12% |
| settle, no gate | +2.5% | 38% | 40% | 27% | 46% |

**FAIL** (bar +54.0%). With every memory always pulling, the right memory comes
back more often (38% and 40%, from 25% and 16%), but memories now also land on
characters that have none (27% and 46%), so the overlap between paths does not
pick out the acting character well enough on its own.

## Change 2: attention in memory, taught by generation (written before running)

Chris: what you learn adjusts what you know, so generation teaches; the model needs
a way to know how much emphasis to put on what it is told, attention in memory and
not in generation; two pulls on the output, and which one wins is the point.

A first draft had a "nothing applies" slot; Chris: you don't turn memory on and off,
and the memory should be a word model. So the memory is its own word model over the
same vocabulary, always on, and the two word models pull on every next word:

    entries e: every recorded output word, key k_e = the recorded state just before it
               (every layer, average removed, each layer scaled to length 1), value v_e = the word
    q        = the same kind of state at the word being generated now
    a_e      = softmax_e(q^T W k_e)                           attention in memory
    p_mem(w) = (1 - eps) * sum_e a_e [v_e = w] + eps / V      the memory's word model
    lam      = softplus(u^T q + b)                            how much emphasis memory gets here
    p(w)     ∝ p_model(w) * p_mem(w) ** lam                   the two pulls; never switched off

W, u, b are the only learned numbers (hand-written gradient, Adam, settings fixed
here: W starts at 5 x identity, u = 0, b = 0, eps = 0.001, lr 0.001, 3 passes). They
learn from the model's own generations, never from the probes: the base model
samples stories; each episode records six generated stories as memories and asks
for the find in one more generated story. The right words are the find of the
memory whose character is the one acting in it, or else that story's own find.
Episodes are drawn at random (half the time the asked story's actor has a memory),
not shaped like the A-D sets, and no story that contains a character from the six
test memories is used in training. The model's weights do not change. Output is
greedy, word by word, each step a forward pass on what has been written so far.

No knob is picked on probes. Each change gets a test set no version has seen: this
one is seed 4004, run once, with both bag-of-words baselines on the same set. Same
score and pass bar.

**Result** (`uv run python research/wordmem.py`, run once): the model wrote 3,767
stories, 3,000 usable. On its own held-out generated episodes the memory got both
words right 18.9% after 3 passes (the most possible is about 50%, since in the
other half the right words are the story's own random find). Locked test, seed 4004:

| system | score | A | C | B wrongly | D wrongly |
| --- | --- | --- | --- | --- | --- |
| model alone | -1.5% | 2% | 6% | 4% | 7% |
| bag-of-words | +21.0% | 100% | 49% | 100% | 7% |
| bag-of-words + recency | +46.5% | 100% | 100% | 100% | 7% |
| word memory | -62.5% | 45% | 30% | 100% | 100% |

**FAIL** (bar +56.5%). The memory wins every time, on every character: the
emphasis learned to always lean on memory (in training, when no memory is right,
nobody can guess the random find anyway, so trusting memory costs little), and the
attention did not learn which memory belongs to the acting character.

## Change 3: memory of the route, recalled at the start (written before running)

Chris: the same weights are used in both; write each generation into memory as the
sequence of weights used; the first pass has no key; on the next generation, math
on the memorised routes decides which weights to use for the next token. And the
recall is not per token but once, at the start, over the whole thing, relative in
space: two trajectories that start close stay alike, and the positions nearest the
tokens to influence count most. That is Hopfield.

- **Route.** At each position, which MLP units in every block fire (ReLU on or off;
  3 blocks x 512 = 1,536 per position). Attention patterns are left out of this
  first version.
- **Write.** Every generation's pass over its sequence is kept as its route, one
  vector per position. The first pass has nothing to compare with; it is only
  written.
- **Recall, once, at the start.** The prompt's route is compared with each
  memory's input route, lined up from the end: `s_g = sum_d gamma^d * cos(prompt
  route at d, memory route at d)`, d words back from the last prompt word, firing
  rates averaged over the corpus removed, gamma < 1 so the nearest positions weigh
  most. Hopfield recall: `a = softmax(beta * s)`, and the recalled route for output
  word j is `R_j = sum_g a_g * (memory g's route at its output word j)`, a firing
  probability for every unit.
- **Generate.** At output word j, every block's MLP pre-activations at the current
  position are pushed toward the recalled route: `z <- z + alpha * sd_u * (R_j,u -
  rate_u)`, with each unit's own spread and firing rate from the corpus; the ReLU
  then decides which weights are used, and the word is the model's top word.
  The model's weights never change.

beta, gamma and alpha are picked on calibration (seed 1001) from beta {5, 20, 80},
gamma {0.5, 0.8, 1.0}, alpha {1, 2, 4}. Test set seed 6006, never seen, run once,
with both bag-of-words baselines. Same score and pass bar.

**Result** (`uv run python research/routes.py`, run once): no setting on calibration
scored above +0.0% (picked beta 20, gamma 0.5, alpha 4). Locked test, seed 6006:

| system | score | A | C | B wrongly | D wrongly |
| --- | --- | --- | --- | --- | --- |
| model alone | -2.0% | 7% | 4% | 4% | 11% |
| bag-of-words | +17.5% | 100% | 46% | 100% | 11% |
| bag-of-words + recency | +44.5% | 100% | 100% | 100% | 11% |
| route memory | +0.0% | 0% | 0% | 0% | 0% |

**FAIL** (bar +54.5%). Pushing the recalled route into the MLP units moved the
output, but not to any memory: it went to the same kind of answer on every probe
("shiny hat", "yellow hat"), so the right memory never came back and no wrong one
was given either.

That run did not do what Chris said: the memories were not the model's own
generations, I fed it the finds ("shiny stone") and recorded that route.

## Change 3, as said: the model's own generations (written before running)

Everything as in change 3, except the memory is what the model itself generated:
the first pass is the same six memory prompts, and the model writes the two words
after `found a` itself (sampled, temperature 1, seed 7001, so the finds are its
own and not its usual answer); the route of that pass is what is written. The
second pass is the probe sets; the right answer for A and C is the actor's own
first-pass find, for B and D none of the six first-pass finds. Bag-of-words uses
the same first-pass finds. Knobs on calibration (seed 1001) from the same grid;
test seed 7007, never seen, run once.

**Result** (`uv run python research/routes_own.py`): first-pass finds the model
wrote: kind bear tiny hat, shy fox shiny flower, brave cat green cake, funny dog
green ball, clever bird yellow apple, happy frog red flower. Calibration best
+0.0% (beta 5, gamma 0.5, alpha 2). Locked test, seed 7007: model alone -3.5%,
bag-of-words +21.5%, with recency +46.5%, route memory -1.0% (A 0%, C 0%, B 0%,
D 2%). **FAIL** (bar +56.5%). Second pass, e.g. `...there the funny dog found a`
-> "old flower", where its first pass wrote "green ball": the recalled route moves
the words but never back to the remembered ones.

Chris: that is wrong, the model was asked to guess in both passes; it was never
given a memory. Its first-pass "find" was a guess with nothing behind it.

## Change 3, with something to remember (written before running)

The first prompt tells the model the fact: a whole story with `one day the funny
dog found a green ball . ...` up to `the big mouse said , " what a`, and the model
generates the next two words itself (from what it was told). The route of that
whole pass, reading and generating, is written into memory. The second prompt is
a new story that ends `the funny dog found a`, without the fact. The six facts are
the original ones (red ball, old cake, blue key, green hat, tiny book, shiny
stone). Recall and push as in change 3; knobs on calibration (seed 1001) from the
same grid; test seed 8008, never seen, run once.

**Result** (`uv run python research/routes_told.py`): on the first pass the model
gave back the fact it was told every time (red ball, old cake, blue key, green hat,
tiny book, shiny stone). Knobs from calibration: beta 80, gamma 0.8, alpha 4.
Locked test, seed 8008: model alone -6.5%, bag-of-words +18.0%, with recency +45.5%,
route memory +15.0% (A 26%, C 38%, B wrongly 10%, D wrongly 24%). **FAIL** (bar
+55.5%). E.g. `...there the brave cat found a` -> "blue key" (told: blue key);
`...there the happy frog found a` -> "shiny book" (told: shiny stone).

## Route memory vs kNN-LM (written before running, 2026-11-22)

`research/fitw.py` builds what Chris described: each pass is written as its route
(which MLP units fire at every word); at every next word, all memories are
searched for the place whose route history best fits the routes so far (Hopfield
recall, nearest words weighing most, gamma), and the first block's weights give,
in one calculation over the vocabulary, how well each word would fire that
place's next route; the word is `argmax log p_model + beta * fit`. On four
hand-made examples it wrote the remembered words, but storing the next word
(kNN-LM) would have done the same. This test asks whether it beats kNN-LM.

**Model**: TinyStories run `20261122-071357` (3 blocks x 128, 8,000 steps).

**Memories**: 50 at once. Each is 8 words of held-out TinyStories text followed by
a told word the model would not have said there: the model's 10th-to-50th ranked
word at that point, picked at random (so it fits the sentence but has under a few
percent probability).

**Probes**, each scored on the next word only:
- partial: a different 6-word span of held-out text, then the memory's last 3
  words before the told word; right if the next word is the told word.
- control: 10 words of held-out text not used anywhere else; right if the next
  word is the model's own top word (the memory should leave it alone).

**Systems**: model alone; kNN-LM (key = the model's final state before the
output, every memory word is an entry with its next word, `p = (1 - lam) p_model
+ lam * softmax(-distance / tau)` over entries); route memory.

**Score**: partial accuracy minus the share of controls whose next word changed.
Knobs (kNN-LM lam {0.25, 0.5, 0.75, 0.9} x tau {1, 10, 100}; route beta {50,
100, 200, 400} x gamma {0.5, 0.8}) are picked on a calibration set with its own
50 memories and probes (seed 11); the test set (seed 22: 50 new memories, 200
partial, 200 control) is run once.

**Pass bar**: route memory beats kNN-LM by at least 10 points.

**Result** (`uv run python research/vsknn.py`, run once; calibration picked kNN-LM
lam 0.5 tau 10, route beta 100 gamma 0.5):

| system | score | partial: told word recalled | control: next word changed |
|---|---|---|---|
| model alone | +0.0% | 0.0% | 0.0% |
| kNN-LM | +13.5% | 71.5% | 58.0% |
| route memory | +10.0% | 81.0% | 71.0% |

**FAIL**: route memory -3.5 points against kNN-LM. It recalls the told word more
often from a partial cue (81% vs 71.5%) but disturbs unrelated prompts more (71%
vs 58%); with no switch, both memories change most prompts they should leave alone.

## Attention on the deltas, with a scale (written before running)

Chris: apply attention to the deltas with a scale. At each next word, every place
in every memory gets a delta, how far the routes so far are from that place's
route history (1 minus the recency-weighted share of units that match, gamma 0.5
as calibrated before), and a weight `exp(-scale * delta)`, not normalised, so a
close place pulls hard and a far one fades to almost nothing, with no switch.
The pull on each word is the weighted sum over all places of how well the word
fires that place's next route: `score = log p_model + beta * sum_places w * fit`.

Same memories, probes, score and pass bar as the kNN-LM test. Knobs picked on
calibration (seed 11): scale {10, 30, 100} x beta {100, 1000, 10000}. The seed-22
test has been seen, so the test is a new set, seed 33, run once, with kNN-LM
(its knobs re-picked on calibration the same way) on the same set.

**Result** (`uv run python research/attn_deltas.py`, run once; calibration picked
kNN-LM lam 0.5 tau 10, deltas scale 100 beta 10000, the corner of the grid):

| system | score | partial: told word recalled | control: next word changed |
|---|---|---|---|
| model alone | +0.0% | 0.0% | 0.0% |
| kNN-LM | +21.0% | 73.5% | 52.5% |
| route memory, attention on deltas | +15.5% | 26.5% | 11.0% |

**FAIL** (-5.5 points against kNN-LM). The scale did what it was meant to:
unrelated prompts changed 11% of the time (from 71%), far less than kNN-LM's
52.5%, but recall from a partial cue fell to 26.5%. Calibration chose the largest
scale and beta in the grid, so the best setting may lie outside it.

## The two curves (written before running)

One setting each is not a fair comparison: both memories trade recall against
disturbing unrelated prompts. This compares the whole trade-off.

Same memories and probes as the kNN-LM test (50 memories, 200 partial, 200
control per set). Every setting of each system is scored on both numbers:
- kNN-LM: lam {0.1, 0.25, 0.5, 0.75, 0.9, 0.97} x tau {0.3, 1, 3, 10, 30, 100}
- route memory, attention on deltas (gamma 0.5): scale {0, 10, 30, 100, 300} x
  beta {30, 100, 300, 1000, 3000, 10000, 30000, 100000}

For each budget of unrelated prompts changed (5%, 10%, 20%, 40%), each system's
setting is the one with the highest recall on calibration (seed 11) among those
within the budget on calibration. Those settings are then scored on a new test
set, seed 44, run once; the table reports each one's test recall and test share
of unrelated prompts changed.

**Pass bar**: at 3 or more of the 4 budgets, the route memory's test recall is at
least 10 points above kNN-LM's while its test share of unrelated prompts changed
is no more than 5 points above kNN-LM's.

**Result** (`uv run python research/curves.py`, run once):

| budget | kNN-LM recall | kNN-LM changed | route recall | route changed | recall gap |
|---|---|---|---|---|---|
| 5% | 5.0% | 1.0% | 9.5% | 5.0% | +4.5% |
| 10% | 20.5% | 8.0% | 25.5% | 10.5% | +5.0% |
| 20% | 20.5% | 8.0% | 44.0% | 20.0% | +23.5% |
| 40% | 38.0% | 25.0% | 54.5% | 34.0% | +16.5% |

**FAIL** by the bar as written: 0 of 4 budgets. At 5% and 10% the recall gap is
under 10 points; at 20% and 40% the gap is large but the route memory used 12 and
9 points more of the budget than kNN-LM did, over the 5-point allowance (the
kNN-LM grid has no setting between 8% and 25% changed).

Read after the fact from the full test curves (not a pre-written claim): where
the two can be compared at a similar share changed, the route memory recalls
more: 44.0% at 20.0% changed against kNN-LM's 38.0% at 25.0%; 54.5% at 34.0%
against 34.0% at 36.5%. kNN-LM reaches higher recall overall (71% at 75-82%
changed); the route memory's best in this grid is 54.5%, at the largest beta, so
its curve is cut off by the grid. Only scale 100 works: below it the memory
swamps every prompt, at 300 it never fires.

## The two curves, wider range (written before running)

Chris: the range needs to pull harder. The route memory's curve was cut off at
the largest beta, and the kNN-LM grid was too coarse to use its budgets. Same
test, wider and finer grids for both:
- kNN-LM: lam {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9} x tau {3, 10, 15, 20, 30, 50}
- route memory: scale {50, 70, 100, 150, 200} x beta {1e3, 3e3, 1e4, 3e4, 1e5,
  3e5, 1e6, 3e6, 1e7}

Budgets 5%, 10%, 20%, 40%, 60%, 80%. Settings picked on calibration (seed 11) per
budget as before; new test set, seed 55, run once (`curves.py wide`).

**Pass bar**: at 4 or more of the 6 budgets, the route memory's test recall is at
least 10 points above kNN-LM's while its test share of unrelated prompts changed
is no more than 5 points above kNN-LM's.

**Result** (`uv run python research/curves.py wide`, run once):

| budget | kNN-LM recall | kNN-LM changed | route recall | route changed | recall gap |
|---|---|---|---|---|---|
| 5% | 7.5% | 4.0% | 44.0% | 8.5% | +36.5% |
| 10% | 11.0% | 8.0% | 53.0% | 15.0% | +42.0% |
| 20% | 34.0% | 23.5% | 64.5% | 25.0% | +30.5% |
| 40% | 52.0% | 45.0% | 73.5% | 32.0% | +21.5% |
| 60% | 64.5% | 64.0% | 73.5% | 32.0% | +9.0% |
| 80% | 72.0% | 85.0% | 66.0% | 77.0% | -6.0% |

**FAIL** by the bar as written: 3 of 6 budgets (5%, 20%, 40%); it needed 4. At
10% the route memory changed 7 points more prompts than kNN-LM (allowance 5); at
60% the gap is +9.0 (needs 10) though it changed half as many prompts; at 80%
calibration picked a worse setting than the one it used at 40%.

Read after the fact from the full test curves: pulling harder with a sharper
scale moved the route memory's curve well above kNN-LM's. Its best setting
(scale 150, beta 1e7) recalls 73.5% while changing 32.0% of unrelated prompts;
kNN-LM needs to change 85% to reach 72%, and at about 32% changed it recalls
46%. The route memory's settings also overshoot their budgets on the test set
(8.5% for the 5% budget, 15% for 10%), and the best settings are again at the
largest beta in the grid.: the trajectory equation

The gap between the model's own path and the settled path is the correction a
memory makes. Collect the pair (own path, settled path) over a few thousand passes
and solve, in one shot by least squares, an equation from a state to its
correction, `delta = A (h - mu)` per layer (no backprop). This is the closed form
ROME and MEMIT use to write facts into weights from a few thousand states; here it
would stay beside the weights, as the memory's own equation.

The test that decides it: fit the equation from one memory's passes, then ask for
that memory in situations it was never recorded in, and see whether the equation
bends to them or only plays the memory back.
