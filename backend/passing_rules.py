"""Authoritative passing rules shared by training and the Flask API."""

import math
from typing import Any

OVERALL_THRESHOLD = 40.0
ETE_THRESHOLD = 13.5
MAXIMUM_ETE = 45.0
TOLERANCE = 1e-9


def calculate_total(ca_marks: float, mte_marks: float, ete_marks: float) -> float:
    return float(ca_marks + mte_marks + ete_marks)


def is_passing(total_marks: float, ete_marks: float) -> bool:
    return (
        total_marks >= OVERALL_THRESHOLD - TOLERANCE
        and ete_marks >= ETE_THRESHOLD - TOLERANCE
    )


def minimum_ete_required(
    ca_marks: float,
    mte_marks: float,
    *,
    whole_number: bool = False,
) -> dict[str, Any]:
    current_marks = float(ca_marks + mte_marks)
    raw_required = max(ETE_THRESHOLD, OVERALL_THRESHOLD - current_marks)
    required = math.ceil(raw_required - TOLERANCE) if whole_number else raw_required
    maximum_total = current_marks + MAXIMUM_ETE
    possible = maximum_total >= OVERALL_THRESHOLD - TOLERANCE
    return {
        "current_marks": current_marks,
        "minimum_ete_marks": float(required),
        "raw_minimum_ete_marks": float(raw_required),
        "maximum_possible_total": float(maximum_total),
        "passing_possible": possible,
        "message": (
            "Passing is mathematically impossible under the configured rules, "
            "even with 45/45 in ETE."
            if not possible
            else "Passing remains mathematically possible under the configured rules."
        ),
    }


def projected_outcome(ca_marks: float, mte_marks: float, predicted_ete: float) -> dict[str, Any]:
    total = calculate_total(ca_marks, mte_marks, predicted_ete)
    calculator = minimum_ete_required(ca_marks, mte_marks)
    required = calculator["minimum_ete_marks"]
    return {
        "predicted_total_marks": total,
        "projected_status": "Pass" if is_passing(total, predicted_ete) else "Fail",
        "minimum_ete_marks": required,
        "ete_gap": float(predicted_ete - required),
        "passing_possible": calculator["passing_possible"],
        "maximum_possible_total": calculator["maximum_possible_total"],
        "calculator_message": calculator["message"],
    }
