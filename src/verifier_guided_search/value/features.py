"""Features for the value model - and the place where hidden-test leakage is prevented.

:func:`node_features` takes a :class:`VisibleFeedback` and four structural numbers. There is no
parameter through which a hidden outcome could arrive, so "the value model does not see the hidden
tests" is a property of this signature rather than a promise. ``tests/unit/test_value_features.py``
pins it from the other side too: permuting every hidden result leaves the feature vector unchanged.

The twelve features are all cheap and all available at the moment a candidate is scored:

=====  =======================  ==============================================================
index  name                     meaning
=====  =======================  ==============================================================
0      visible_pass_fraction    passed / total on the visible group
1      visible_failed           count of visible tests not passing
2      visible_total            size of the visible group
3      status_ok                1 when the program ran to completion
4      error_fraction           share of visible tests that raised
5      fail_fraction            share that returned the wrong answer
6      distinct_error_classes   how many different exception classes appeared
7      depth                    distance from the root of the search tree
8      edit_distance_from_root  edits between this program and the starting program
9      program_chars_hundreds   source length, in hundreds of characters
10     parent_pass_fraction     the parent's visible pass fraction
11     pass_fraction_delta      this node's pass fraction minus its parent's
=====  =======================  ==============================================================

Feature 6 matters more than it looks: a mutant that raises the *same* exception on every failing
test is usually one local edit away from correct, while a scatter of different exception classes
means the program is broken in several places.
"""

from __future__ import annotations

from dataclasses import dataclass

from verifier_guided_search.sandbox.protocol import VisibleFeedback

__all__ = ["FEATURE_NAMES", "FeatureContext", "node_features"]

FEATURE_NAMES: tuple[str, ...] = (
    "visible_pass_fraction",
    "visible_failed",
    "visible_total",
    "status_ok",
    "error_fraction",
    "fail_fraction",
    "distinct_error_classes",
    "depth",
    "edit_distance_from_root",
    "program_chars_hundreds",
    "parent_pass_fraction",
    "pass_fraction_delta",
)


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Structural facts about a node. None of these depend on hidden outcomes."""

    depth: int
    edit_distance_from_root: int
    program_chars: int
    parent_pass_fraction: float


def node_features(feedback: VisibleFeedback, context: FeatureContext) -> tuple[float, ...]:
    """The feature vector for one evaluated node, in :data:`FEATURE_NAMES` order."""
    group = feedback.visible
    total = group.total
    errors = sum(1 for item in group.results if item.status == "error")
    fails = sum(1 for item in group.results if item.status == "fail")
    distinct = len(group.error_class_counts())
    pass_fraction = group.pass_fraction
    return (
        pass_fraction,
        float(group.failed),
        float(total),
        1.0 if feedback.status == "ok" else 0.0,
        errors / total if total else 0.0,
        fails / total if total else 0.0,
        float(distinct),
        float(context.depth),
        float(context.edit_distance_from_root),
        context.program_chars / 100.0,
        context.parent_pass_fraction,
        pass_fraction - context.parent_pass_fraction,
    )
