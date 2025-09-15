import dataclasses
from typing import Dict, Optional, Union, List

from lm_eval.tasks.ifbench.instructions_registry import INSTRUCTION_DICT

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def _clean_kwargs(d: Dict[str, Optional[Union[str, int]]] | None) -> Dict[str, Union[str, int]]:
    """Remove None values to avoid unexpected keyword errors."""
    if not d:
        return {}
    return {k: v for k, v in d.items() if v is not None}


def _build_instruction(instruction_id: str, prompt: str, kw: Dict[str, Optional[Union[str, int]]]):
    """Instantiate IFBench instruction and (re)build its description.

    IFBench instructions expose `build_description(**kwargs)` and
    `get_instruction_args()` like IFEval; if the arg list contains
    "prompt", we call `build_description(prompt=prompt)` as a second pass.
    """
    instruction_cls = INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    # First pass: instantiate with provided kwargs (after stripping None values)
    instruction.build_description(**_clean_kwargs(kw))

    # Some instructions need the *prompt* explicitly
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)

    return instruction


def _make_response_variants(response: str) -> List[str]:
    """Variants used by the *loose* metric to reduce false negatives.

    Mirrors the common IFEval practice of stripping bullets/markdown and
    dropping first/last lines to ignore wrapper text.
    """
    lines = response.split("\n")
    resp = response.strip()
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()

    # Remove simple emphasis markers
    variants = [
        resp,
        resp.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for v in variants:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def test_instruction_following_strict(inp: InputExample, response: str) -> OutputExample:
    """Return strict instruction-following results for a single prompt/response.

    Aligns with IFBench's evaluation_lib expectations: we take a *string*
    `response` (not a mapping), compute per-instruction booleans and an
    all-instructions prompt-level boolean.
    """
    is_following_list: List[bool] = []

    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction = _build_instruction(instruction_id, inp.prompt, inp.kwargs[index])
        ok = bool(response.strip()) and instruction.check_following(response)
        is_following_list.append(ok)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp: InputExample, response: str) -> OutputExample:
    """Return loose instruction-following results for a single prompt/response."""
    variants = _make_response_variants(response)
    is_following_list: List[bool] = []

    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction = _build_instruction(instruction_id, inp.prompt, inp.kwargs[index])

        ok = False
        for r in variants:
            if r and instruction.check_following(r):
                ok = True
                break
        is_following_list.append(ok)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    """Harness-style metric hook.

    Parameters
    ----------
    doc : mapping
        One row from IFBench_test.jsonl with keys: "key", "prompt",
        "instruction_id_list", and "kwargs" (list aligned to `instruction_id_list`).
    results : list[str] | str
        Generated model text(s). We use the first generation.

    Returns
    -------
    dict
        Metric dictionary with the four IFBench/IFEval-style keys that
        `evaluation_lib.py` expects to aggregate and print.
    """
    response = results[0] if isinstance(results, list) else results

    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items: List[List[bool]]) -> float:
    """Aggregate instruction-level booleans across prompts to a scalar.

    This mirrors the harness' simple mean-of-bools convention.
    """
    flat = [x for sub in items for x in sub]
    return (sum(flat) / len(flat)) if flat else 0.0