import json

import pytest

from app.agent import cited_sources, format_evidence, parse_json
from app.schemas import ClaimCheck, Review


def test_parse_json_plain_fenced_and_wrapped():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": 1} done') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_cited_sources_only_cited(toolbox_factory):
    box = toolbox_factory()
    for drug in ("warfarin", "metformin"):
        box.execute("read_section", json.dumps({"drug": drug, "section": "contraindications", "source": "dailymed"}))
    assert cited_sources("See [dailymed-warfarin.md, §4].", box) == ["dailymed-warfarin.md"]
    assert cited_sources("no citation here", box) == ["dailymed-metformin.md", "dailymed-warfarin.md"]


def test_format_evidence_has_labels_and_limit(toolbox_factory):
    box = toolbox_factory()
    box.execute("read_section", json.dumps({"drug": "warfarin", "section": "contraindications", "source": "dailymed"}))
    text = format_evidence(box.evidence_items(), {})
    assert text.startswith("[dailymed-warfarin.md § 4 CONTRAINDICATIONS")
    assert "omitted" in format_evidence(box.evidence_items(), {}, limit=10)


def test_review_needs_revision():
    assert not Review(claims=[ClaimCheck(claim="x", verdict="supported")]).needs_revision()
    assert Review(claims=[ClaimCheck(claim="x", verdict="ambiguous")]).needs_revision()
    assert Review(issues=["missed overlap"]).needs_revision()