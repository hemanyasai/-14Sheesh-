from src.generation import _format_count_answer, _validate_count_answer


def test_count_answer_validation_accepts_matching_count():
    answer = _format_count_answer("How many players?", [{"name": "A"}, {"name": "B"}], 2)
    _validate_count_answer(answer, 2, [{"name": "A"}, {"name": "B"}], "How many players?")


def test_count_answer_validation_rejects_mismatched_count():
    try:
        _validate_count_answer("There are 0 matching players in this dataset.", 2, [{"name": "A"}, {"name": "B"}], "How many players?")
    except AssertionError as exc:
        assert "expected 2" in str(exc)
    else:
        raise AssertionError("Expected validation to fail")
