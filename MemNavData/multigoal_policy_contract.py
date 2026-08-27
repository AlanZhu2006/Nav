"""Execution contract for role-controlled three-leg evaluations.

The benchmark metadata defines the causal role of every leg.  This module
keeps that benchmark-level oracle separate from the role-free relocalization
method: the known-role arm switches controller only on Revisit legs, while
automatic routes must make the same decision from their own observations on
every leg.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Optional


PolicyBackend = Optional[str]
DEFAULT_ROLE_SEQUENCE = ("initial_imagegoal", "novel", "revisit")
SUPPORTED_ROLES = frozenset({"initial_imagegoal", "novel", "revisit"})


def three_leg_policy_backends(
    *,
    server_backend: str,
    hybrid_route: str,
    automatic_routes: Collection[str],
    role_sequence: Collection[str] = DEFAULT_ROLE_SEQUENCE,
    known_revisit_leg_indices: Optional[Collection[int]] = None,
) -> tuple[PolicyBackend, PolicyBackend, PolicyBackend]:
    """Resolve the controller path for the three declared causal roles.

    ``None`` means the single native NavDP server owns the request.  Under the
    two-server hybrid runtime, ``navdp`` explicitly streams the observation to
    causal memory while leaving control with native NavDP; ``navdp_mix`` is the
    known-Revisit residual; and ``navdp_auto`` lets a deployable router decide.
    """

    roles = tuple(role_sequence)
    if (len(roles) != 3 or roles[0] != "initial_imagegoal"
            or any(role not in SUPPORTED_ROLES for role in roles)):
        raise ValueError(f"unsupported three-leg role sequence {roles!r}")

    automatic = frozenset(automatic_routes)
    if server_backend == "navdp":
        if known_revisit_leg_indices is not None:
            raise ValueError(
                "known-Revisit leg selection requires the hybrid memory server"
            )
        if hybrid_route != "phase":
            raise ValueError(
                "native-only three-leg evaluation must use hybrid_route=phase; "
                "an automatic route requires the hybrid memory server"
            )
        return (None, None, None)

    if server_backend != "hybrid_pose":
        raise ValueError(
            "three-leg evaluation supports only navdp and hybrid_pose"
        )

    if hybrid_route == "phase":
        # The role label is benchmark supervision, not a deployable selector.
        # Native-controlled legs still stream their decision frames to
        # long-term memory so every later Revisit receives exactly the causal
        # history that existed online.
        declared_revisits = {
            index for index, role in enumerate(roles) if role == "revisit"
        }
        enabled_revisits = (
            declared_revisits
            if known_revisit_leg_indices is None
            else {int(index) for index in known_revisit_leg_indices}
        )
        if not enabled_revisits.issubset(declared_revisits):
            raise ValueError(
                "known-Revisit leg selection contains a non-Revisit leg"
            )
        return tuple(
            "navdp_mix" if index in enabled_revisits else "navdp"
            for index, _role in enumerate(roles)
        )
    if hybrid_route in automatic:
        if known_revisit_leg_indices is not None:
            raise ValueError(
                "known-Revisit leg selection cannot modify an automatic route"
            )
        return ("navdp_auto", "navdp_auto", "navdp_auto")
    raise ValueError(
        f"unsupported hybrid three-leg route {hybrid_route!r}"
    )


__all__ = [
    "DEFAULT_ROLE_SEQUENCE",
    "PolicyBackend",
    "three_leg_policy_backends",
]
