"""Hebrew -> English (LS1910) versification remap."""
from __future__ import annotations

import pytest

from retrievve.data.versification import heb_to_eng, remap_rows


@pytest.mark.parametrize("heb,eng", [
    (("DEU", 13, 1), (12, 32)), (("DEU", 13, 2), (13, 1)), (("DEU", 23, 4), (23, 3)),
    (("DEU", 28, 69), (29, 1)), (("DEU", 29, 1), (29, 2)), (("NUM", 17, 1), (16, 36)),
    (("NUM", 17, 15), (16, 50)), (("NUM", 17, 16), (17, 1)), (("GEN", 1, 1), (1, 1)),
])
def test_table(heb, eng):
    assert heb_to_eng(*heb) == eng


def test_psalms_rejected():
    with pytest.raises(ValueError):
        heb_to_eng("PSA", 3, 1)


def row(loc, tgt, fr):
    return {"locator": loc, "text_raw": tgt, "fr_raw": fr, "extra": {}}


def test_remap_repairs_french_within_scope_only():
    rows = [row("DEU.23.3", "p23.3", "F23.3"), row("DEU.23.4", "Ammon/Moab (heb 23.4)", "F23.4"),
            row("DEU.12.1", "p12.1", "F12.1")]
    out, rep = remap_rows(rows, scope={"DEU.23"})
    by = {r["text_raw"]: r for r in out}
    assert by["Ammon/Moab (heb 23.4)"]["locator"] == "DEU.23.3"
    assert by["Ammon/Moab (heb 23.4)"]["fr_raw"] == "F23.3"
    assert by["p12.1"]["locator"] == "DEU.12.1"            # out of scope: untouched
    assert rep["moved"] == 2


def test_collision_with_unmoved_row_drops_french_of_moved_row():
    rows = [row("DEU.12.32", "p12.32", "F12.32"), row("DEU.13.1", "heb13.1", "F13.1")]
    out, rep = remap_rows(rows, scope={"DEU.13"})
    moved = next(r for r in out if r["text_raw"] == "heb13.1")
    assert moved["locator"] == "DEU.12.32" and moved["fr_raw"] is None and rep["locator_collisions"] == 1
