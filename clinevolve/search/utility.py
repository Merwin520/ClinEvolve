#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# =============================================================================
# Constants
# =============================================================================

UTILITY_SCHEMA_VERSION = "hyperagent_utility_v1"


# =============================================================================
# Helpers
# =============================================================================

def _json_safe(
    value: Any,
) -> Any:
    """
    Convert common values recursively into JSON-safe objects.

    This module intentionally avoids numpy/pandas dependencies.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key):
                _json_safe(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    # Common scalar wrappers, e.g. numpy scalar.
    if hasattr(
        value,
        "item",
    ):
        try:
            return _json_safe(
                value.item()
            )
        except Exception:
            pass

    return str(
        value
    )


def _finite_or_none(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


# =============================================================================
# Generic candidate evidence
# =============================================================================

@dataclass
class CandidateEvidence:
    """
    Generic evidence associated with one candidate.

    Search core does NOT interpret the contents of `metrics`.

    Examples:

        Exp2:
            metrics = {
                "target_failure_rate": ...,
                "final_diagnosis_failure_rate": ...,
                ...
            }

        Exp3:
            metrics = {
                "accuracy": ...,
                "latency": ...,
                ...
            }

    Experiment-specific evaluator code decides what these fields mean.
    """

    candidate_id: str

    metrics: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:

        if not str(
            self.candidate_id
        ).strip():
            raise ValueError(
                "candidate_id must be non-empty"
            )

        if not isinstance(
            self.metrics,
            dict,
        ):
            raise TypeError(
                "metrics must be a dict"
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dict"
            )

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        self.validate()

        return _json_safe(
            asdict(
                self
            )
        )


# =============================================================================
# Generic utility result
# =============================================================================

@dataclass
class UtilityResult:
    """
    Generic result returned by one experiment-specific UtilityEvaluator.

    Fields
    ------
    candidate_id:
        Unique candidate identifier.

    eligible:
        Whether the candidate is allowed to enter final selection.

        This is intentionally generic. An experiment may define eligibility
        using safety gates, minimum performance thresholds, validity checks,
        resource limits, or any other preregistered rule.

    primary_score:
        Main utility score.

        Convention:
            larger = better

        This convention is fixed at the search-core level so that experiments
        do not need to redefine sorting semantics.

        If the natural metric is "lower is better", the experiment-specific
        evaluator should transform it before returning UtilityResult.

        Example:
            failure_rate = 0.20
            primary_score = -0.20

    secondary_scores:
        Ordered secondary criteria.

        Convention:
            larger = better

        Python dictionary insertion order is preserved, but for formal
        experiment use the experiment evaluator should define a stable
        secondary_order explicitly.

    complexity:
        Optional candidate complexity.

        Convention:
            smaller = better

        Examples:
            patch lines changed
            number of files modified
            AST edit count

    resource_cost:
        Optional scalar resource cost.

        Convention:
            smaller = better

        Examples:
            token overhead
            latency overhead
            dollar cost
            tool-call overhead

    rejection_reasons:
        Reasons an ineligible candidate was rejected.

    details:
        Arbitrary experiment-specific diagnostics.
    """

    candidate_id: str

    eligible: bool

    primary_score: Optional[
        float
    ]

    secondary_scores: Dict[
        str,
        Optional[float],
    ] = field(
        default_factory=dict
    )

    secondary_order: List[
        str
    ] = field(
        default_factory=list
    )

    complexity: Optional[
        float
    ] = None

    resource_cost: Optional[
        float
    ] = None

    rejection_reasons: List[
        str
    ] = field(
        default_factory=list
    )

    details: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:

        if not str(
            self.candidate_id
        ).strip():
            raise ValueError(
                "candidate_id must be non-empty"
            )

        if not isinstance(
            self.eligible,
            bool,
        ):
            raise TypeError(
                "eligible must be bool"
            )

        if (
            self.primary_score
            is not None
            and _finite_or_none(
                self.primary_score
            )
            is None
        ):
            raise ValueError(
                "primary_score must be finite "
                "or None"
            )

        unknown_order = [
            key
            for key
            in self.secondary_order
            if key
            not in self.secondary_scores
        ]

        if unknown_order:
            raise ValueError(
                "secondary_order contains "
                "unknown score names: "
                f"{unknown_order}"
            )

        if (
            len(
                self.secondary_order
            )
            != len(
                set(
                    self.secondary_order
                )
            )
        ):
            raise ValueError(
                "secondary_order contains duplicates"
            )

        for (
            name,
            value,
        ) in (
            self.secondary_scores.items()
        ):

            if (
                value
                is not None
                and _finite_or_none(
                    value
                )
                is None
            ):
                raise ValueError(
                    f"secondary score '{name}' "
                    "must be finite or None"
                )

        if (
            self.complexity
            is not None
            and _finite_or_none(
                self.complexity
            )
            is None
        ):
            raise ValueError(
                "complexity must be finite "
                "or None"
            )

        if (
            self.resource_cost
            is not None
            and _finite_or_none(
                self.resource_cost
            )
            is None
        ):
            raise ValueError(
                "resource_cost must be finite "
                "or None"
            )

        if (
            self.eligible
            and self.rejection_reasons
        ):
            raise ValueError(
                "eligible candidate cannot have "
                "rejection_reasons"
            )

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        self.validate()

        payload = asdict(
            self
        )

        payload[
            "schema_version"
        ] = (
            UTILITY_SCHEMA_VERSION
        )

        return _json_safe(
            payload
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[
            str,
            Any,
        ],
    ) -> "UtilityResult":

        raw = dict(
            payload
        )

        schema_version = raw.pop(
            "schema_version",
            UTILITY_SCHEMA_VERSION,
        )

        if (
            schema_version
            != UTILITY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported utility schema: "
                f"{schema_version}"
            )

        result = cls(
            **raw
        )

        result.validate()

        return result


# =============================================================================
# Generic evaluator interface
# =============================================================================

class UtilityEvaluator(
    ABC
):
    """
    Interface implemented by each experiment.

    Search core does not know what "good" means.

    The experiment-specific evaluator maps evidence into a common
    UtilityResult.

    Example
    -------
    experiments/exp2/utility.py:

        class Exp2UtilityEvaluator(UtilityEvaluator):

            def evaluate(
                self,
                baseline,
                candidate,
                context=None,
            ):
                ...
                return UtilityResult(...)

    experiments/exp3/utility.py:

        class Exp3UtilityEvaluator(UtilityEvaluator):

            def evaluate(...):
                ...
    """

    @abstractmethod
    def evaluate(
        self,
        baseline: Optional[
            CandidateEvidence
        ],
        candidate: CandidateEvidence,
        *,
        context: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> UtilityResult:
        """
        Evaluate one candidate.

        Args:
            baseline:
                Optional baseline evidence.

                Some experiments compare candidates against A0.
                Other experiments may not require a baseline.

            candidate:
                Evidence for the candidate being evaluated.

            context:
                Optional experiment-specific context.

        Returns:
            UtilityResult
        """
        raise NotImplementedError

    def evaluate_many(
        self,
        baseline: Optional[
            CandidateEvidence
        ],
        candidates: Iterable[
            CandidateEvidence
        ],
        *,
        context: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> List[
        UtilityResult
    ]:
        """
        Default batch implementation.

        Experiments may override this if they need vectorized or joint
        evaluation, but most experiments can reuse this method.
        """

        results = []

        for candidate in candidates:

            result = self.evaluate(
                baseline,
                candidate,
                context=context,
            )

            if (
                result.candidate_id
                != candidate.candidate_id
            ):
                raise ValueError(
                    "UtilityEvaluator returned "
                    "candidate_id different from "
                    "input candidate"
                )

            result.validate()

            results.append(
                result
            )

        return results


# =============================================================================
# Ranking policy
# =============================================================================

@dataclass(frozen=True)
class RankingConfig:
    """
    Generic ranking configuration.

    Eligibility is always highest priority.

    Default ordering:

        1. eligible candidate first
        2. larger primary_score
        3. secondary_scores in configured order
        4. smaller complexity
        5. smaller resource_cost
        6. candidate_id deterministic tie-break

    missing_secondary_value:
        Ranking value used when a secondary score is missing.

        Because larger is better, default -inf means missing scores lose
        against any finite score.

    missing_complexity_value:
        Complexity value used when missing.

        Because smaller is better, default +inf means missing complexity
        loses against any known finite complexity.

    missing_resource_value:
        Same principle for resource cost.
    """

    require_eligible: bool = True

    secondary_order: Tuple[
        str,
        ...
    ] = ()

    use_complexity_tiebreak: bool = True

    use_resource_tiebreak: bool = True

    missing_secondary_value: float = float(
        "-inf"
    )

    missing_complexity_value: float = float(
        "inf"
    )

    missing_resource_value: float = float(
        "inf"
    )


# =============================================================================
# Ranking helpers
# =============================================================================

def _resolved_secondary_order(
    result: UtilityResult,
    config: RankingConfig,
) -> List[str]:
    """
    Resolve secondary criterion order.

    Priority:
        RankingConfig.secondary_order
        -> UtilityResult.secondary_order
        -> sorted score names
    """

    if config.secondary_order:

        order = list(
            config.secondary_order
        )

    elif result.secondary_order:

        order = list(
            result.secondary_order
        )

    else:

        order = sorted(
            result.secondary_scores
        )

    return order


def candidate_rank_key(
    result: UtilityResult,
    config: Optional[
        RankingConfig
    ] = None,
) -> Tuple[
    Any,
    ...
]:
    """
    Return deterministic lexicographic ranking tuple.

    Lower tuple = better.

    Important convention:
        primary_score / secondary scores:
            larger = better

        complexity / resource cost:
            smaller = better
    """

    if config is None:
        config = (
            RankingConfig()
        )

    result.validate()

    # -------------------------------------------------------------------------
    # Eligibility
    # -------------------------------------------------------------------------

    eligibility_rank = (
        0
        if result.eligible
        else 1
    )

    # -------------------------------------------------------------------------
    # Primary score
    # -------------------------------------------------------------------------

    if (
        result.primary_score
        is None
    ):

        primary_rank = float(
            "inf"
        )

    else:

        primary_rank = -float(
            result.primary_score
        )

    # -------------------------------------------------------------------------
    # Secondary scores
    # -------------------------------------------------------------------------

    secondary_rank: List[
        float
    ] = []

    order = (
        _resolved_secondary_order(
            result,
            config,
        )
    )

    for name in order:

        value = (
            result.secondary_scores.get(
                name
            )
        )

        if value is None:

            effective_value = (
                config.missing_secondary_value
            )

        else:

            effective_value = float(
                value
            )

        # Larger utility = better.
        secondary_rank.append(
            -effective_value
        )

    # -------------------------------------------------------------------------
    # Complexity
    # -------------------------------------------------------------------------

    if (
        config.use_complexity_tiebreak
    ):

        if (
            result.complexity
            is None
        ):

            complexity_rank = (
                config.missing_complexity_value
            )

        else:

            complexity_rank = float(
                result.complexity
            )

    else:

        complexity_rank = 0.0

    # -------------------------------------------------------------------------
    # Resource
    # -------------------------------------------------------------------------

    if (
        config.use_resource_tiebreak
    ):

        if (
            result.resource_cost
            is None
        ):

            resource_rank = (
                config.missing_resource_value
            )

        else:

            resource_rank = float(
                result.resource_cost
            )

    else:

        resource_rank = 0.0

    return (
        eligibility_rank,
        primary_rank,
        *secondary_rank,
        complexity_rank,
        resource_rank,
        result.candidate_id,
    )


def rank_candidates(
    results: Sequence[
        UtilityResult
    ],
    *,
    config: Optional[
        RankingConfig
    ] = None,
) -> List[
    UtilityResult
]:
    """
    Return candidates from best to worst.
    """

    if config is None:
        config = RankingConfig()

    return sorted(
        list(
            results
        ),
        key=lambda result: (
            candidate_rank_key(
                result,
                config,
            )
        ),
    )


# =============================================================================
# Selection
# =============================================================================

def select_best_candidate(
    results: Sequence[
        UtilityResult
    ],
    *,
    config: Optional[
        RankingConfig
    ] = None,
) -> Optional[
    UtilityResult
]:
    """
    Select the best candidate.

    By default:
        if no eligible candidate exists -> return None

    This is safer than selecting the "least bad" rejected candidate.

    To allow rejected candidates in a specific exploratory experiment:

        RankingConfig(require_eligible=False)
    """

    if not results:
        return None

    if config is None:
        config = RankingConfig()

    ranked = rank_candidates(
        results,
        config=config,
    )

    if not config.require_eligible:

        return ranked[
            0
        ]

    eligible = [
        result
        for result
        in ranked
        if result.eligible
    ]

    if not eligible:
        return None

    return eligible[
        0
    ]


# =============================================================================
# Convenience lookup helpers
# =============================================================================

def results_by_candidate_id(
    results: Iterable[
        UtilityResult
    ],
) -> Dict[
    str,
    UtilityResult,
]:
    """
    Convert result sequence to candidate_id -> UtilityResult.

    Duplicate candidate IDs are rejected.
    """

    mapping: Dict[
        str,
        UtilityResult,
    ] = {}

    for result in results:

        result.validate()

        if (
            result.candidate_id
            in mapping
        ):
            raise ValueError(
                "Duplicate UtilityResult "
                f"candidate_id: "
                f"{result.candidate_id}"
            )

        mapping[
            result.candidate_id
        ] = result

    return mapping


# =============================================================================
# Serialization helpers
# =============================================================================

def save_results_json(
    results: Sequence[
        UtilityResult
    ],
    path: str | Path,
) -> None:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version":
            UTILITY_SCHEMA_VERSION,

        "n_results":
            len(
                results
            ),

        "results": [
            result.to_dict()
            for result
            in results
        ],
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write(
            "\n"
        )


def load_results_json(
    path: str | Path,
) -> List[
    UtilityResult
]:

    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(
            f
        )

    schema_version = payload.get(
        "schema_version"
    )

    if (
        schema_version
        != UTILITY_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported utility file schema: "
            f"{schema_version}"
        )

    return [
        UtilityResult.from_dict(
            raw
        )
        for raw
        in payload.get(
            "results",
            []
        )
    ]


# =============================================================================
# Example utility evaluator used only for self-test
# =============================================================================

class _ExampleUtilityEvaluator(
    UtilityEvaluator
):
    """
    Self-test only.

    Natural experiment metric:
        score: higher = better

    Eligibility:
        candidate score >= baseline score

    This class is not used by any real experiment.
    """

    def evaluate(
        self,
        baseline: Optional[
            CandidateEvidence
        ],
        candidate: CandidateEvidence,
        *,
        context: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> UtilityResult:

        candidate.validate()

        candidate_score = (
            _finite_or_none(
                candidate.metrics.get(
                    "score"
                )
            )
        )

        if candidate_score is None:

            return UtilityResult(
                candidate_id=(
                    candidate.candidate_id
                ),
                eligible=False,
                primary_score=None,
                rejection_reasons=[
                    "missing score"
                ],
            )

        baseline_score = None

        if baseline is not None:

            baseline.validate()

            baseline_score = (
                _finite_or_none(
                    baseline.metrics.get(
                        "score"
                    )
                )
            )

        eligible = (
            baseline_score
            is None
            or candidate_score
            >= baseline_score
        )

        rejection_reasons = []

        if not eligible:

            rejection_reasons.append(
                "candidate below baseline"
            )

        secondary_score = (
            _finite_or_none(
                candidate.metrics.get(
                    "robustness"
                )
            )
        )

        complexity = (
            _finite_or_none(
                candidate.metrics.get(
                    "patch_size"
                )
            )
        )

        resource_cost = (
            _finite_or_none(
                candidate.metrics.get(
                    "resource_cost"
                )
            )
        )

        return UtilityResult(
            candidate_id=(
                candidate.candidate_id
            ),

            eligible=eligible,

            primary_score=(
                candidate_score
            ),

            secondary_scores={
                "robustness":
                    secondary_score
            },

            secondary_order=[
                "robustness"
            ],

            complexity=(
                complexity
            ),

            resource_cost=(
                resource_cost
            ),

            rejection_reasons=(
                rejection_reasons
            ),

            details={
                "baseline_score":
                    baseline_score,

                "candidate_score":
                    candidate_score,
            },
        )


# =============================================================================
# Minimal self-test
# =============================================================================

def _self_test() -> None:

    import tempfile

    baseline = CandidateEvidence(
        candidate_id="A0",
        metrics={
            "score":
                0.70
        },
    )

    candidate_1 = CandidateEvidence(
        candidate_id="c001",
        metrics={
            "score":
                0.80,

            "robustness":
                0.90,

            "patch_size":
                10,

            "resource_cost":
                1.2,
        },
    )

    candidate_2 = CandidateEvidence(
        candidate_id="c002",
        metrics={
            "score":
                0.80,

            "robustness":
                0.85,

            "patch_size":
                5,

            "resource_cost":
                1.0,
        },
    )

    candidate_3 = CandidateEvidence(
        candidate_id="c003",
        metrics={
            "score":
                0.60,

            "robustness":
                0.99,

            "patch_size":
                1,

            "resource_cost":
                0.5,
        },
    )

    evaluator = (
        _ExampleUtilityEvaluator()
    )

    results = (
        evaluator.evaluate_many(
            baseline,
            [
                candidate_1,
                candidate_2,
                candidate_3,
            ],
        )
    )

    ranked = rank_candidates(
        results,
        config=RankingConfig(
            secondary_order=(
                "robustness",
            ),
        ),
    )

    assert (
        ranked[
            0
        ].candidate_id
        == "c001"
    )

    assert (
        ranked[
            1
        ].candidate_id
        == "c002"
    )

    assert (
        ranked[
            2
        ].candidate_id
        == "c003"
    )

    best = (
        select_best_candidate(
            results,
            config=RankingConfig(
                secondary_order=(
                    "robustness",
                ),
            ),
        )
    )

    assert (
        best
        is not None
    )

    assert (
        best.candidate_id
        == "c001"
    )

    rejected_only = [
        UtilityResult(
            candidate_id="bad1",
            eligible=False,
            primary_score=0.99,
            rejection_reasons=[
                "failed experiment rule"
            ],
        ),
    ]

    assert (
        select_best_candidate(
            rejected_only
        )
        is None
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        path = (
            Path(
                tmpdir
            )
            / "utility_results.json"
        )

        save_results_json(
            results,
            path,
        )

        loaded = (
            load_results_json(
                path
            )
        )

        assert (
            len(
                loaded
            )
            == 3
        )

        assert (
            loaded[
                0
            ].candidate_id
            == results[
                0
            ].candidate_id
        )

    print(
        "utility.py self-test: PASS"
    )


if __name__ == "__main__":
    _self_test()
