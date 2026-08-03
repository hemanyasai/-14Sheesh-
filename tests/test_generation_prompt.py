from src.generation import build_prompt


def test_list_question_prompt_includes_direct_output_instruction():
    prompt = build_prompt(
        "Which players were active in 2024?",
        [{"name": "A"}, {"name": "B"}],
        exact_count=2,
    )
    assert "Output format requirement" in prompt
    assert "For list questions, output only the requested player names" in prompt


def test_count_question_prompt_includes_direct_output_instruction():
    prompt = build_prompt(
        "How many Bangladeshi wicket-keepers were active before 1950?",
        [],
        exact_count=0,
    )
    assert "Output format requirement" in prompt
    assert "For count questions, output only the number" in prompt
