from src.retrieval import _infer_era_filter_from_text, _build_qdrant_filter


def test_before_year_filter_is_inferred():
    extracted = {"filters": {}, "era_filter": _infer_era_filter_from_text("How many Bangladeshi wicket-keepers were active before 1950?")}
    qf = _build_qdrant_filter(extracted)
    assert qf is not None
    assert qf.must[0].key == "era_end"


def test_after_year_filter_is_inferred():
    extracted = {"filters": {}, "era_filter": _infer_era_filter_from_text("Which players were active after 1950?")}
    qf = _build_qdrant_filter(extracted)
    assert qf is not None
    assert qf.must[0].key == "era_start"
