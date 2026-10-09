"""align_check: name detection, matcher normalisations, and the name-independent length check."""
from __future__ import annotations

import random

from retrievve.analysis.align_check import find_names, length_check, skeleton


def rec(book, ch, v, text, fr):
    return {"id": f"fan:test:{book}.{ch}.{v}", "group_id": f"bible/{book}", "text": text, "fr": fr}


def test_imperatives_and_ecriture_are_not_names():
    recs = [rec("ROM", 16, i, "x", "Saluez Prisca. Servez Paul selon l'Écriture.") for i in range(5)]
    names = find_names(recs)
    assert "paul" in names and "prisca" in names
    assert not {"saluez", "servez", "écriture"} & names


def test_soft_c():
    assert skeleton("Macédoine")[:3] == skeleton("Masedonia")[:3]
    assert skeleton("César")[:2] == skeleton("Sesar")[:2]
    assert skeleton("Corinthe")[0] == "k"


def test_length_check_separates_aligned_from_shuffled():
    rng = random.Random(0)
    aligned, shuffled = [], []
    for ch in range(1, 6):
        lens = [rng.randint(20, 200) for _ in range(20)]
        frs = ["f" * int(n * 1.1) for n in lens]
        perm = frs[:]
        rng.shuffle(perm)
        for v, n in enumerate(lens, 1):
            aligned.append(rec("AAA", ch, v, "t" * n, frs[v - 1]))
            shuffled.append(rec("BBB", ch, v, "t" * n, perm[v - 1]))
    res = length_check(aligned + shuffled, perms=199)
    assert res["bible/AAA"]["r"] > 0.95 and res["bible/AAA"]["perm_p"] <= 0.01
    assert res["bible/AAA"]["r_shift1"] < 0.5
    assert res["bible/BBB"]["r"] < 0.5
