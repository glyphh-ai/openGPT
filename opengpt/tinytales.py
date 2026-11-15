"""A small made-up story corpus, generated the same way every time.

About 2,000 very short stories with a fixed cast and simple grammar: small
enough to train on in seconds, structured enough that held-out accuracy means
something. It is a stand-in for TinyStories, not a copy of it.
"""

import numpy as np

NAMES = ["tom", "lily", "max", "anna", "ben", "mia", "sam", "zoe"]
ANIMALS = ["cat", "dog", "bird", "fox", "rabbit", "bear", "frog", "mouse"]
ADJ = ["little", "big", "happy", "brave", "kind", "shy", "funny", "clever"]
PLACES = ["forest", "garden", "park", "house", "farm", "village", "meadow", "cave"]
THINGS = ["ball", "apple", "key", "hat", "book", "stone", "flower", "cake"]
COLORS = ["red", "blue", "green", "yellow", "shiny", "old", "tiny", "round"]
FEEL = ["happy", "sad", "scared", "proud", "tired", "excited", "surprised", "calm"]
FRIENDS = ["mother", "father", "sister", "brother", "friend", "teacher"]
VERBS = ["found", "saw", "lost", "made", "took", "hid"]
ENDS = [
    "they played all day .",
    "they laughed and went home .",
    "then it was time to sleep .",
    "everyone was happy again .",
    "and they were friends forever .",
]


def story(rng):
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    name, animal, adj, place = pick(NAMES), pick(ANIMALS), pick(ADJ), pick(PLACES)
    thing, color, feel, friend = pick(THINGS), pick(COLORS), pick(FEEL), pick(FRIENDS)
    verb, place2 = pick(VERBS), pick(PLACES)
    s = [f"once upon a time there was a {adj} {animal} named {name} .",
         f"{name} lived in a {place} ."]
    if rng.random() < 0.7:
        s.append(f"one day {name} {verb} a {color} {thing} .")
        s.append(f"{name} was very {feel} .")
    else:
        s.append(f"one day {name} went to the {place2} .")
        s.append(f"there {name} {verb} a {color} {thing} .")
    if rng.random() < 0.6:
        s.append(f"{name} showed the {thing} to the {friend} .")
        s.append(f"the {friend} said , \" what a {color} {thing} ! \"")
    else:
        s.append(f"{name} took the {thing} to the {place2} .")
    s.append(pick(ENDS))
    return " ".join(s)


def generate(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return "\n<|endoftext|>\n".join(story(rng) for _ in range(n)) + "\n"
