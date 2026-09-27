"""Fifteen parametric task families: string transforms, interval maths, parsing and sequences.

Each family owns a reference solution *template* and two input generators. Instantiating a family
substitutes the instance's parameters into the template, so instance 03 of a family is a different
function from instance 07, with its own tests, not a renamed copy.

Reference solutions use only built-ins and no imports: candidate programs run under an import
deny-list in the sandbox, and a reference that needed `collections` would make the benchmark measure
the sandbox's import policy instead of the search.

Visible inputs are random draws. Hidden inputs are the *edge* cases (empty input, a single element,
values exactly on a boundary) plus further random draws. That asymmetry is deliberate: it is what
lets a mutant pass every visible test and still be wrong, which is the verifier false-pass rate the
evaluation reports.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from string import Template
from typing import Final

__all__ = ["FAMILIES", "FamilySpec", "family_by_name"]

Params = Mapping[str, object]
#: JSON data as a *covariant* type, for the literal test inputs written out below.
#:
#: pydantic's ``JsonValue`` cannot describe them: its list member is ``list[JsonValue]``, and
#: ``list`` is invariant, so ``[[1, 2]]`` is not a ``list[JsonValue]`` however obviously it is
#: JSON. These inputs are only ever read - iterated and splatted into a call - so a ``Sequence`` is
#: both accurate and assignable. :func:`verifier_guided_search.tasks.suite._cases` converts them to
#: ``JsonValue`` where they enter a pydantic model, which validates them at runtime.
type JsonLike = str | int | float | bool | Sequence[JsonLike] | None
ArgList = Sequence[JsonLike]
InputMaker = Callable[[random.Random, Params], Sequence[ArgList]]

_LETTERS: Final = string.ascii_lowercase
_ROMAN_TEXTS: Final = (
    "XIV",
    "MCMXCIV",
    "IX",
    "LVIII",
    "MMXXIV",
    "XLII",
    "CDXLIV",
    "DCCCLXXXVIII",
    "III",
    "XC",
)
_BAD_ROMAN: Final = ("IIII", "VX", "", "ABC", "MMMM", "IC")


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """A family: how to name it, how to write its reference, and how to make its inputs."""

    name: str
    entry_point: str
    statement: str
    template: str
    parameters: Callable[[int], dict[str, object]]
    visible_inputs: InputMaker
    hidden_inputs: InputMaker

    def render_source(self, params: Params) -> str:
        return Template(self.template).substitute(params).strip() + "\n"

    def render_statement(self, params: Params) -> str:
        return Template(self.statement).substitute(params).strip()


def _args(*values: JsonLike) -> ArgList:
    """One call's arguments, typed explicitly.

    A heterogeneous list literal such as ``[intervals, 0, 20]`` joins to ``list[object]``, which is
    not JSON data as far as a type checker is concerned. Naming the element type here keeps the
    family definitions readable without weakening :data:`ArgList`.
    """
    return list(values)


def _text(rng: random.Random, *, min_len: int, max_len: int, alphabet: str = "aabbcdxyz ") -> str:
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))


def _ints(rng: random.Random, count: int, lo: int, hi: int) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(count)]


def _intervals(rng: random.Random, count: int, span: int = 30) -> list[list[int]]:
    out: list[list[int]] = []
    for _ in range(count):
        start = rng.randint(0, span)
        out.append([start, start + rng.randint(0, 8)])
    return out


# --------------------------------------------------------------------------------------- 1. RLE
_RUN_LENGTH = FamilySpec(
    name="run_length_encode",
    entry_point="run_length_encode",
    statement=(
        "Compress `text` by replacing every run of $min_run or more identical characters with the "
        "character followed by the run length. Shorter runs are copied unchanged. Return the "
        "compressed string."
    ),
    template="""
def run_length_encode(text):
    min_run = $min_run
    out = []
    index = 0
    while index < len(text):
        end = index
        while end < len(text) and text[end] == text[index]:
            end = end + 1
        length = end - index
        if length >= min_run:
            out.append(text[index] + str(length))
        else:
            out.append(text[index] * length)
        index = end
    return ''.join(out)
""",
    parameters=lambda index: {"min_run": 2 + index % 3},
    visible_inputs=lambda rng, params: [
        [_text(rng, min_len=6, max_len=24, alphabet="aaabbc")] for _ in range(6)
    ],
    hidden_inputs=lambda rng, params: [
        [""],
        ["a"],
        ["a" * int(str(params["min_run"]))],
        ["a" * (int(str(params["min_run"])) - 1)],
        ["ab" * 9],
        *([_text(rng, min_len=1, max_len=40, alphabet="aabbbccd")] for _ in range(7)),
    ],
)

# ---------------------------------------------------------------------------------- 2. Caesar
_CAESAR = FamilySpec(
    name="caesar_shift",
    entry_point="caesar_shift",
    statement=(
        "Shift every lowercase letter of `text` forward by $shift positions in the alphabet, "
        "wrapping from z back to a. Leave every other character unchanged. Return the result."
    ),
    template="""
def caesar_shift(text):
    shift = $shift
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    out = []
    for char in text:
        position = alphabet.find(char)
        if position >= 0:
            out.append(alphabet[(position + shift) % 26])
        else:
            out.append(char)
    return ''.join(out)
""",
    parameters=lambda index: {"shift": 1 + index % 12},
    visible_inputs=lambda rng, params: [
        [_text(rng, min_len=5, max_len=20, alphabet=_LETTERS + " ,.")] for _ in range(6)
    ],
    hidden_inputs=lambda rng, params: [
        [""],
        ["z"],
        ["a"],
        ["zzz aaa"],
        ["Mixed CASE 123"],
        *([_text(rng, min_len=1, max_len=30, alphabet=_LETTERS + " !Z")] for _ in range(7)),
    ],
)

# -------------------------------------------------------------------------- 3. Balanced brackets
_BRACKETS = FamilySpec(
    name="balanced_brackets",
    entry_point="balanced_brackets",
    statement=(
        "Return the index of the first character of `text` that breaks bracket balance for the "
        "pairs $pairs, or -1 when the whole string is balanced. A closing bracket with no open "
        "partner, a mismatched pair, and an unclosed bracket at the end all count as breaks; for "
        "an unclosed bracket report the index of the opening character."
    ),
    template="""
def balanced_brackets(text):
    openers = '$openers'
    closers = '$closers'
    stack = []
    positions = []
    for index in range(len(text)):
        char = text[index]
        if openers.find(char) >= 0:
            stack.append(char)
            positions.append(index)
        elif closers.find(char) >= 0:
            if len(stack) == 0:
                return index
            expected = openers[closers.find(char)]
            if stack[-1] != expected:
                return index
            stack.pop()
            positions.pop()
    if len(stack) > 0:
        return positions[0]
    return -1
""",
    parameters=lambda index: (
        {"openers": "([{", "closers": ")]}", "pairs": "() [] {}"}
        if index % 2 == 0
        else {"openers": "(<", "closers": ")>", "pairs": "() <>"}
    ),
    visible_inputs=lambda rng, params: [
        [
            "".join(
                rng.choice(str(params["openers"]) + str(params["closers"]) + "ab")
                for _ in range(rng.randint(4, 14))
            )
        ]
        for _ in range(6)
    ],
    hidden_inputs=lambda rng, params: [
        [""],
        [str(params["openers"])[0]],
        [str(params["closers"])[0]],
        [str(params["openers"])[0] + str(params["closers"])[0]],
        ["ab" + str(params["openers"])[0] * 3],
        *(
            [
                "".join(
                    rng.choice(str(params["openers"]) + str(params["closers"]) + "xy")
                    for _ in range(rng.randint(1, 18))
                )
            ]
            for _ in range(7)
        ),
    ],
)

# ------------------------------------------------------------------------------ 4. Merge intervals
_MERGE = FamilySpec(
    name="merge_intervals",
    entry_point="merge_intervals",
    statement=(
        "Sort `intervals` (pairs [start, end] with start <= end) and merge any two whose gap is at "
        "most $gap, so [1, 3] and [$touch, 6] become one interval. Return the merged list."
    ),
    template="""
def merge_intervals(intervals):
    gap = $gap
    items = sorted(intervals)
    merged = []
    for interval in items:
        start = interval[0]
        end = interval[1]
        if len(merged) > 0 and start - merged[-1][1] <= gap:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged
""",
    parameters=lambda index: {"gap": index % 3, "touch": 3 + index % 3},
    visible_inputs=lambda rng, params: [[_intervals(rng, rng.randint(3, 7))] for _ in range(6)],
    hidden_inputs=lambda rng, params: [
        [[]],
        [[[2, 2]]],
        [[[0, 1], [1 + int(str(params["gap"])), 4]]],
        [[[5, 9], [0, 1]]],
        [[[0, 10], [2, 3], [4, 5]]],
        *([_intervals(rng, rng.randint(1, 9))] for _ in range(7)),
    ],
)

# -------------------------------------------------------------------------------- 5. Interval gaps
_GAPS = FamilySpec(
    name="interval_gaps",
    entry_point="interval_gaps",
    statement=(
        "Given `intervals` and the window [lo, hi], return the maximal sub-ranges of the window "
        "that no interval covers, keeping only gaps of length $min_gap or more. Return a list of "
        "[start, end] pairs in increasing order."
    ),
    template="""
def interval_gaps(intervals, lo, hi):
    min_gap = $min_gap
    items = sorted(intervals)
    gaps = []
    cursor = lo
    for interval in items:
        start = interval[0]
        end = interval[1]
        if start > cursor:
            if start - cursor >= min_gap:
                gaps.append([cursor, min(start, hi)])
        if end > cursor:
            cursor = end
    if hi - cursor >= min_gap:
        gaps.append([cursor, hi])
    return gaps
""",
    parameters=lambda index: {"min_gap": 1 + index % 4},
    visible_inputs=lambda rng, params: [
        [_intervals(rng, rng.randint(2, 6), span=20), 0, 24] for _ in range(6)
    ],
    hidden_inputs=lambda rng, params: [
        [[], 0, 10],
        [[[0, 10]], 0, 10],
        [[[3, 4]], 0, 4 + int(str(params["min_gap"]))],
        [[[1, 2], [2, 3]], 0, 9],
        [[[8, 12], [0, 2]], 0, 12],
        *(_args(_intervals(rng, rng.randint(1, 7), span=18), 0, 20) for _ in range(7)),
    ],
)

# --------------------------------------------------------------------------------- 6. Longest run
_LONGEST_RUN = FamilySpec(
    name="longest_run",
    entry_point="longest_run",
    statement=(
        "Return the length of the longest run of equal neighbouring values in `values`, but report "
        "0 when the longest run is shorter than $min_len."
    ),
    template="""
def longest_run(values):
    min_len = $min_len
    best = 0
    current = 0
    previous = None
    for index in range(len(values)):
        if index > 0 and values[index] == previous:
            current = current + 1
        else:
            current = 1
        if current > best:
            best = current
        previous = values[index]
    if best < min_len:
        return 0
    return best
""",
    parameters=lambda index: {"min_len": 2 + index % 3},
    visible_inputs=lambda rng, params: [[_ints(rng, rng.randint(4, 14), 0, 3)] for _ in range(6)],
    hidden_inputs=lambda rng, params: [
        [[]],
        [[7]],
        [[1] * int(str(params["min_len"]))],
        [[1] * (int(str(params["min_len"])) - 1) + [2]],
        [[1, 1, 2, 2, 2, 3]],
        *([_ints(rng, rng.randint(1, 20), 0, 2)] for _ in range(7)),
    ],
)

# --------------------------------------------------------------------------------- 7. Digital root
_DIGITAL_ROOT = FamilySpec(
    name="digital_root",
    entry_point="digital_root",
    statement=(
        "Repeatedly replace the non-negative integer `number` by the sum of its digits in base "
        "$base until the result is smaller than $base, then return it."
    ),
    template="""
def digital_root(number):
    base = $base
    value = number
    while value >= base:
        total = 0
        while value > 0:
            total = total + value % base
            value = value // base
        value = total
    return value
""",
    parameters=lambda index: {"base": 2 + index % 9},
    visible_inputs=lambda rng, params: [[rng.randint(0, 5000)] for _ in range(6)],
    hidden_inputs=lambda rng, params: [
        [0],
        [1],
        [int(str(params["base"]))],
        [int(str(params["base"])) - 1],
        [int(str(params["base"])) ** 3],
        *([rng.randint(0, 200000)] for _ in range(7)),
    ],
)

# -------------------------------------------------------------------------------- 8. Parse records
_PARSE_RECORDS = FamilySpec(
    name="parse_records",
    entry_point="parse_records",
    statement=(
        "Split `text` on '$sep', strip surrounding spaces from each field, drop empty fields and "
        "return at most $limit fields as a list."
    ),
    template="""
def parse_records(text):
    separator = '$sep'
    limit = $limit
    fields = []
    current = ''
    for char in text:
        if char == separator:
            fields.append(current)
            current = ''
        else:
            current = current + char
    fields.append(current)
    out = []
    for field in fields:
        trimmed = field.strip(' ')
        if len(trimmed) > 0:
            out.append(trimmed)
    return out[:limit]
""",
    parameters=lambda index: {"sep": ",;|:"[index % 4], "limit": 3 + index % 4},
    visible_inputs=lambda rng, params: [
        [
            str(params["sep"]).join(
                _text(rng, min_len=0, max_len=5, alphabet="ab c") for _ in range(rng.randint(2, 6))
            )
        ]
        for _ in range(6)
    ],
    hidden_inputs=lambda rng, params: [
        [""],
        [str(params["sep"])],
        [f"  a  {params['sep']}  "],
        [str(params["sep"]).join("abcdefgh")],
        [f"a{params['sep']}{params['sep']}b"],
        *(
            [
                str(params["sep"]).join(
                    _text(rng, min_len=0, max_len=4, alphabet=" xy")
                    for _ in range(rng.randint(1, 9))
                )
            ]
            for _ in range(7)
        ),
    ],
)

# ----------------------------------------------------------------------------------- 9. Wrap words
_WRAP = FamilySpec(
    name="wrap_words",
    entry_point="wrap_words",
    statement=(
        "Greedily wrap the space-separated words of `text` into lines of at most $width characters "
        "and return the list of lines. A word longer than the width gets a line of its own. Lines "
        "never have leading or trailing spaces, and empty input gives an empty list."
    ),
    template="""
def wrap_words(text):
    width = $width
    words = []
    current = ''
    for char in text:
        if char == ' ':
            if len(current) > 0:
                words.append(current)
            current = ''
        else:
            current = current + char
    if len(current) > 0:
        words.append(current)
    lines = []
    line = ''
    for word in words:
        if len(line) == 0:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line = line + ' ' + word
        else:
            lines.append(line)
            line = word
    if len(line) > 0:
        lines.append(line)
    return lines
""",
    parameters=lambda index: {"width": 6 + 2 * (index % 6)},
    visible_inputs=lambda rng, params: [
        [
            " ".join(
                _text(rng, min_len=1, max_len=7, alphabet="abcd") for _ in range(rng.randint(3, 9))
            )
        ]
        for _ in range(6)
    ],
    hidden_inputs=lambda rng, params: [
        [""],
        ["   "],
        ["a" * (int(str(params["width"])) + 3)],
        ["a" * int(str(params["width"]))],
        ["ab cd ef gh ij kl"],
        *(
            [
                " ".join(
                    _text(rng, min_len=1, max_len=12, alphabet="xyz")
                    for _ in range(rng.randint(1, 8))
                )
            ]
            for _ in range(7)
        ),
    ],
)

# -------------------------------------------------------------------------------- 10. Collatz steps
_COLLATZ = FamilySpec(
    name="collatz_steps",
    entry_point="collatz_steps",
    statement=(
        "Count the Collatz steps needed to bring the positive integer `number` down to 1 (halve it "
        "when even, otherwise 3n+1). Give up and return -1 after $cap steps. Return 0 for 1."
    ),
    template="""
def collatz_steps(number):
    cap = $cap
    value = number
    steps = 0
    while value != 1:
        if steps >= cap:
            return -1
        if value % 2 == 0:
            value = value // 2
        else:
            value = 3 * value + 1
        steps = steps + 1
    return steps
""",
    parameters=lambda index: {"cap": 8 + 6 * (index % 5)},
    visible_inputs=lambda rng, params: [[rng.randint(1, 60)] for _ in range(6)],
    hidden_inputs=lambda rng, params: [
        [1],
        [2],
        [27],
        [2 ** (int(str(params["cap"])) - 1)],
        [2 ** int(str(params["cap"]))],
        *([rng.randint(1, 400)] for _ in range(7)),
    ],
)

# -------------------------------------------------------------------------------- 11. Roman numeral
_ROMAN = FamilySpec(
    name="roman_to_int",
    entry_point="roman_to_int",
    statement=(
        "Convert the uppercase Roman numeral `text` to an integer, rejecting anything that is not "
        "a valid numeral with -1. Values above $limit are also rejected."
    ),
    template="""
def roman_to_int(text):
    limit = $limit
    symbols = 'IVXLCDM'
    values = [1, 5, 10, 50, 100, 500, 1000]
    if len(text) == 0:
        return -1
    total = 0
    previous = 0
    repeats = 1
    for index in range(len(text)):
        char = text[len(text) - 1 - index]
        position = symbols.find(char)
        if position < 0:
            return -1
        value = values[position]
        if value == previous:
            repeats = repeats + 1
            if repeats > 3:
                return -1
        else:
            repeats = 1
        if value < previous:
            if previous // value > 10:
                return -1
            total = total - value
        else:
            total = total + value
        previous = value
    if total > limit:
        return -1
    return total
""",
    parameters=lambda index: {"limit": 500 + 500 * (index % 5)},
    visible_inputs=lambda rng, params: [[rng.choice(_ROMAN_TEXTS)] for _ in range(6)],
    hidden_inputs=lambda rng, params: [
        *([bad] for bad in _BAD_ROMAN),
        ["I"],
        ["MMM"],
        *([rng.choice(_ROMAN_TEXTS + _BAD_ROMAN)] for _ in range(4)),
    ],
)

# ---------------------------------------------------------------------------------- 12. Chunk list
_CHUNK = FamilySpec(
    name="chunk_list",
    entry_point="chunk_list",
    statement=(
        "Split `values` into consecutive chunks of $size elements. The final chunk may be shorter "
        "but is dropped entirely when it holds fewer than $keep elements. Return the list of "
        "chunks."
    ),
    template="""
def chunk_list(values):
    size = $size
    keep = $keep
    chunks = []
    index = 0
    while index < len(values):
        chunk = values[index:index + size]
        if len(chunk) >= keep:
            chunks.append(chunk)
        index = index + size
    return chunks
""",
    parameters=lambda index: {"size": 2 + index % 4, "keep": 1 + index % 2},
    visible_inputs=lambda rng, params: [[_ints(rng, rng.randint(3, 13), 0, 9)] for _ in range(6)],
    hidden_inputs=lambda rng, params: [
        [[]],
        [[1]],
        # Fixed seeds so these edge sizes are the same in every build of the suite.
        [_ints(random.Random(1), int(str(params["size"])), 0, 9)],  # noqa: S311  # nosec B311
        [_ints(random.Random(2), int(str(params["size"])) + 1, 0, 9)],  # noqa: S311  # nosec B311
        [_ints(random.Random(3), 2 * int(str(params["size"])), 0, 9)],  # noqa: S311  # nosec B311
        *([_ints(rng, rng.randint(1, 17), 0, 9)] for _ in range(7)),
    ],
)

# ------------------------------------------------------------------------------ 13. Normalise text
_NORMALIZE = FamilySpec(
    name="normalize_spacing",
    entry_point="normalize_spacing",
    statement=(
        "Collapse every run of spaces in `text` to a single space, drop leading and trailing "
        "spaces, then $case_description the result and return it."
    ),
    template="""
def normalize_spacing(text):
    words = []
    current = ''
    for char in text:
        if char == ' ':
            if len(current) > 0:
                words.append(current)
            current = ''
        else:
            current = current + char
    if len(current) > 0:
        words.append(current)
    joined = ' '.join(words)
    return joined.$case_call
""",
    parameters=lambda index: (
        {"case_call": "upper()", "case_description": "upper-case"}
        if index % 3 == 0
        else (
            {"case_call": "lower()", "case_description": "lower-case"}
            if index % 3 == 1
            else {"case_call": "title()", "case_description": "title-case"}
        )
    ),
    visible_inputs=lambda rng, params: [
        [_text(rng, min_len=4, max_len=22, alphabet="ab CD  e")] for _ in range(6)
    ],
    hidden_inputs=lambda rng, params: [
        [""],
        ["   "],
        ["a"],
        ["  Mixed   CASE words  "],
        ["one"],
        *([_text(rng, min_len=1, max_len=28, alphabet="xY  zW")] for _ in range(7)),
    ],
)

# ------------------------------------------------------------------------- 14. Interval cover
# length
_COVER = FamilySpec(
    name="interval_cover_length",
    entry_point="interval_cover_length",
    statement=(
        "Return the total length covered by the union of `intervals`, multiplied by $scale. "
        "Overlapping intervals must not be counted twice."
    ),
    template="""
def interval_cover_length(intervals):
    scale = $scale
    items = sorted(intervals)
    total = 0
    cursor = None
    reach = None
    for interval in items:
        start = interval[0]
        end = interval[1]
        if cursor is None:
            cursor = start
            reach = end
        elif start > reach:
            total = total + (reach - cursor)
            cursor = start
            reach = end
        elif end > reach:
            reach = end
    if cursor is not None:
        total = total + (reach - cursor)
    return total * scale
""",
    parameters=lambda index: {"scale": 1 + index % 5},
    visible_inputs=lambda rng, params: [[_intervals(rng, rng.randint(2, 7))] for _ in range(6)],
    hidden_inputs=lambda rng, params: [
        [[]],
        [[[4, 4]]],
        [[[0, 5], [5, 9]]],
        [[[0, 9], [2, 3]]],
        [[[7, 8], [0, 1], [3, 4]]],
        *([_intervals(rng, rng.randint(1, 9))] for _ in range(7)),
    ],
)

# -------------------------------------------------------------------------------- 15. Top-k values
_TOP_K = FamilySpec(
    name="top_k_frequent",
    entry_point="top_k_frequent",
    statement=(
        "Return the $k most frequent integers in `values`, most frequent first, breaking ties by "
        "the smaller value. Return fewer than $k entries when there are not enough distinct values."
    ),
    template="""
def top_k_frequent(values):
    k = $k
    counts = {}
    for value in values:
        if value in counts:
            counts[value] = counts[value] + 1
        else:
            counts[value] = 1
    pairs = []
    for value in counts:
        pairs.append([-counts[value], value])
    pairs.sort()
    out = []
    for pair in pairs[:k]:
        out.append(pair[1])
    return out
""",
    parameters=lambda index: {"k": 1 + index % 4},
    visible_inputs=lambda rng, params: [[_ints(rng, rng.randint(5, 18), 0, 5)] for _ in range(6)],
    hidden_inputs=lambda rng, params: [
        [[]],
        [[3]],
        [[1, 2, 3, 4, 5]],
        [[1, 1, 2, 2, 3, 3]],
        [_ints(random.Random(4), 30, 0, 2)],  # noqa: S311  # nosec B311
        *([_ints(rng, rng.randint(1, 25), 0, 7)] for _ in range(7)),
    ],
)

FAMILIES: Final[tuple[FamilySpec, ...]] = (
    _RUN_LENGTH,
    _CAESAR,
    _BRACKETS,
    _MERGE,
    _GAPS,
    _LONGEST_RUN,
    _DIGITAL_ROOT,
    _PARSE_RECORDS,
    _WRAP,
    _COLLATZ,
    _ROMAN,
    _CHUNK,
    _NORMALIZE,
    _COVER,
    _TOP_K,
)


def family_by_name(name: str) -> FamilySpec:
    for family in FAMILIES:
        if family.name == name:
            return family
    known: Sequence[str] = [family.name for family in FAMILIES]
    raise KeyError(f"unknown family {name!r}; known families: {', '.join(known)}")
