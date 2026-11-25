"""A made-up story corpus with no names: two characters, either one may act.

Each character is "the ADJ ANIMAL". The point is that who did something is only
known from the story's order, so a test can ask for a memory of the character who
acts while another character is in view.

Run:  uv run python research/rolestales.py      (writes data/rolestales.txt)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opengpt import tinytales as tt  # noqa: E402

VERBS2 = ["saw", "met", "helped", "called"]


def story(rng, a=None, b=None, actor=None, find=None, upto=False):
    """One story. a, b are (adj, animal); actor is 0 or 1; find is (colour, thing).

    With upto=True the story stops right after 'found a' (a probe).
    """
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    a = a or (pick(tt.ADJ), pick(tt.ANIMALS))
    b = b or (pick(tt.ADJ), pick(tt.ANIMALS))
    while b == a:
        b = (pick(tt.ADJ), pick(tt.ANIMALS))
    actor = int(rng.integers(2)) if actor is None else actor
    color, thing = find or (pick(tt.COLORS), pick(tt.THINGS))
    ch = [f"the {a[0]} {a[1]}", f"the {b[0]} {b[1]}"]
    first = int(rng.integers(2))      # who is named first, and who sees whom, is random
    s = [f"{ch[first]} and {ch[1 - first]} lived in a {pick(tt.PLACES)} ."]
    seer = int(rng.integers(2))
    s.append(f"{ch[seer]} {pick(VERBS2)} {ch[1 - seer]} .")
    if rng.random() < 0.5:
        s.append(f"one day {ch[actor]} found a")
    else:
        s.append(f"one day {ch[actor]} went to the {pick(tt.PLACES)} . there {ch[actor]} found a")
    if upto:
        return " ".join(s)
    s[-1] += f" {color} {thing} ."
    s.append(f"{ch[actor]} was very {pick(tt.FEEL)} .")
    s.append(f"{ch[1 - actor]} said , \" what a {color} {thing} ! \"")
    s.append(pick(tt.ENDS))
    return " ".join(s)


def generate(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return "\n<|endoftext|>\n".join(story(rng) for _ in range(n)) + "\n"


if __name__ == "__main__":
    path = ROOT / "data" / "rolestales.txt"
    path.write_text(generate())
    print("wrote", path)
