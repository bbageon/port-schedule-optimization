"""Explicit, immutable execution-policy configuration (YR-160).

Candidate generation used to read mutable module globals.  That made two runs
in the same process able to silently change one another.  The policy contract
is now a value object passed to every active execution path.

The old module constants in :mod:`yard_rl.integrated.candidates` remain only
as deprecated reproduction labels.  Candidate generation never reads them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


_WAIT_MODES = frozenset({"WAIT", "DEFER_ALL", "DEFER_TRIGGER"})


@dataclass(frozen=True)
class ExecPolicyConfig:
    """Complete candidate and joint-decision contract for one execution policy."""

    name: str
    wait_mode: str = "WAIT"
    safety_only: bool = False
    bound_repo: bool = False
    prepo_one_shot: bool = False
    forbid_strategic_wait: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("execution policy config requires a non-empty name")
        if self.wait_mode not in _WAIT_MODES:
            raise ValueError(f"unsupported wait_mode: {self.wait_mode!r}")

    def as_dict(self) -> dict:
        return asdict(self)

    def with_changes(self, *, name: str, **changes) -> "ExecPolicyConfig":
        """Return a named, immutable arm configuration."""
        return replace(self, name=name, **changes)


# Adopted deployment contract: no free reposition, finite defer, one-shot
# history retained for compatibility, and strategic WAIT combinations removed.
ADOPTED_C0_GUARD = ExecPolicyConfig(
    name="ADOPTED_C0_GUARD",
    wait_mode="DEFER_ALL",
    safety_only=True,
    bound_repo=False,
    prepo_one_shot=True,
    forbid_strategic_wait=True,
)

# Historical/default behavior.  Reproduction harnesses must pass this value
# explicitly when they need to state their policy contract.
LEGACY_DEFAULT = ExecPolicyConfig(
    name="LEGACY_DEFAULT",
    wait_mode="WAIT",
    safety_only=False,
    bound_repo=False,
    prepo_one_shot=False,
    forbid_strategic_wait=False,
)


def current() -> ExecPolicyConfig:
    """Deprecated compatibility helper; no mutable global policy exists now."""
    return LEGACY_DEFAULT
