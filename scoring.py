"""Shared scoring utilities for all benchmarks."""

from __future__ import annotations

import re
import string

# Prefix used to mark predictions where answer extraction failed
PARSE_FAILED_PREFIX = "__PARSE_FAILED__:"


def _parse_number(text: str) -> float | None:
    """Extract a numeric value from text like '$2,684', '2684.50', '-$100'."""
    text = text.strip()
    # Remove currency symbols, commas, whitespace
    cleaned = re.sub(r'[$,\s]', '', text)
    try:
        return float(cleaned)
    except ValueError:
        # Try to find a number anywhere in the text
        match = re.search(r'-?[\d,]+\.?\d*', text.replace(',', ''))
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting like **bold** and *italic*."""
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    return text


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove trailing punctuation."""
    text = _strip_markdown(text)
    text = text.strip().lower()
    text = text.strip(string.punctuation + string.whitespace)
    return text


def _normalize_name(text: str) -> str:
    """Normalize a name for fuzzy matching: collapse periods, extra spaces, etc."""
    text = normalize(text)
    # Remove periods (M. -> M, Jr. -> Jr, Inc. -> Inc)
    text = text.replace(".", "")
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _fuzzy_contains(haystack: str, needle: str, bidirectional: bool = False) -> bool:
    """Check if needle appears in haystack with fuzzy name matching.

    If bidirectional=True, also checks if haystack appears in needle
    (for partial name extraction like "Grenfell-Gardner" matching
    "Jason Grenfell-Gardner").
    """
    if needle in haystack:
        return True
    hay_name = _normalize_name(haystack)
    ndl_name = _normalize_name(needle)
    if ndl_name in hay_name:
        return True
    if bidirectional and hay_name in ndl_name:
        return True
    return False


def extract_answer(response: str, valid_labels: list[str] | None = None) -> str:
    """Extract the answer from a model response.

    Looks for 'The answer is ...' pattern first, then falls back to label matching.
    """
    response = response.strip()
    if not response:
        return ""

    # Try to extract from "The answer is ..." pattern
    response_lower = response.lower()
    prefix = "the answer is "
    idx = response_lower.find(prefix)
    if idx != -1:
        after = response[idx + len(prefix):].strip().rstrip(".")
        # If valid_labels, match the extracted text against them
        if valid_labels:
            after_norm = normalize(after)
            normalized_labels = {normalize(l): l for l in valid_labels}
            # Try exact match first
            if after_norm in normalized_labels:
                return normalized_labels[after_norm]
            # Try prefix match (e.g. "fanciful" from "fanciful, because...")
            for norm_label, original in normalized_labels.items():
                if after_norm.startswith(norm_label) or norm_label in after_norm:
                    return original
        # No valid_labels or no match — return raw extracted text
        # Take first line only, strip trailing punctuation and markdown
        first_line = _strip_markdown(after.split("\n")[0].strip().rstrip("."))
        return first_line

    # Fallback: match against valid labels anywhere in response
    if valid_labels:
        normalized_labels = {normalize(l): l for l in valid_labels}
        first_line = normalize(response.split("\n")[0])
        for norm_label, original in normalized_labels.items():
            if first_line == norm_label:
                return original
        for norm_label, original in normalized_labels.items():
            if norm_label in response_lower:
                return original
        # Could not match any valid label — parse failure
        return PARSE_FAILED_PREFIX + response.split("\n")[0].strip()

    # No valid labels — return first line (can't determine parse failure)
    return response.split("\n")[0].strip()


def _parse_expected_parts(expected: str) -> list[str]:
    """Parse expected answer into parts.

    Handles JSON arrays (e.g. '["a", "b"]'), semicolon-separated,
    and comma-separated strings. Returns list of normalized parts.
    """
    expected = expected.strip()
    # Try JSON array first
    if expected.startswith("["):
        try:
            import json
            items = json.loads(expected)
            if isinstance(items, list):
                return [normalize(str(x)) for x in items if str(x).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
    # Try semicolon-separated
    if ";" in expected:
        return [normalize(p) for p in expected.split(";") if p.strip()]
    # Try comma-separated (only if multiple commas suggest a list)
    if expected.count(",") >= 1 and len(expected) > 50:
        return [normalize(p) for p in expected.split(",") if p.strip()]
    # Single value
    return [normalize(expected)]


def is_correct(predicted: str, expected: str, eval_method: str = "contained_in_output") -> bool:
    """Check if a prediction is correct given the eval method.

    Eval methods:
        contained_in_output: expected answer appears in the response
            (also checks if predicted appears in expected, for partial-name
            extraction like "Knoll" vs "Knoll Inc.")
        all_in_output: all parts of expected appear in response
        any_in_output: any part of expected appears in response
    """
    pred_norm = normalize(predicted)
    exp_norm = normalize(expected)

    # Empty predictions are always wrong
    if not pred_norm:
        return False

    if eval_method == "contained_in_output":
        # Check both directions with fuzzy name matching
        return _fuzzy_contains(pred_norm, exp_norm) or _fuzzy_contains(exp_norm, pred_norm)

    if eval_method == "all_in_output":
        parts = _parse_expected_parts(expected)
        return all(_fuzzy_contains(pred_norm, p, bidirectional=True) for p in parts)

    if eval_method == "any_in_output":
        parts = _parse_expected_parts(expected)
        return any(_fuzzy_contains(pred_norm, p, bidirectional=True) for p in parts)

    if eval_method == "numeric_within_1pct":
        pred_num = _parse_number(predicted)
        exp_num = _parse_number(expected)
        if pred_num is None or exp_num is None:
            return False
        if exp_num == 0:
            return pred_num == 0
        return abs(pred_num - exp_num) / abs(exp_num) <= 0.01

    # Fallback to exact match
    return pred_norm == exp_norm


def compute_accuracy(
    predictions: list[str],
    labels: list[str],
    eval_method: str = "contained_in_output",
) -> dict:
    """Compute accuracy for a list of predictions against labels.

    Returns dict with accuracy, correct count, total count, and parse failures.
    A parse failure is when extract_answer could not find a valid answer
    (indicated by a PARSE_FAILED prefix on the prediction).
    """
    correct = 0
    parse_failures = 0
    for p, l in zip(predictions, labels):
        if p.startswith(PARSE_FAILED_PREFIX):
            parse_failures += 1
        elif is_correct(p, l, eval_method):
            correct += 1
    total = len(labels)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
    }
