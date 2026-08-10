"""YR-160: execution configuration is explicit and process-local."""
from yard_rl.integrated import candidates as cand_mod
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.policy_config import (ADOPTED_C0_GUARD,
                                              ExecPolicyConfig,
                                              LEGACY_DEFAULT)


def test_deprecated_globals_are_not_read():
    injected = CandidateGenerator(config=LEGACY_DEFAULT)
    defaulted = CandidateGenerator()
    old = (cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY,
           cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT)
    try:
        cand_mod.WAIT_MODE = "DEFER_ALL"
        cand_mod.SAFETY_ONLY = True
        cand_mod.BOUND_REPO = True
        cand_mod.PREPO_ONE_SHOT = True
        assert injected._flags() == ("WAIT", False, False, False)
        assert defaulted._flags() == ("WAIT", False, False, False)
    finally:
        (cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY,
         cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT) = old


def test_adopted_injection_matches_contract():
    gen = CandidateGenerator(config=ADOPTED_C0_GUARD)
    assert gen._flags() == ("DEFER_ALL", True, False, True)
    assert gen.config.forbid_strategic_wait is True


def test_generators_cannot_contaminate_each_other():
    adopted = CandidateGenerator(config=ADOPTED_C0_GUARD)
    other = CandidateGenerator(config=ExecPolicyConfig(name="other"))
    assert adopted._flags() == ("DEFER_ALL", True, False, True)
    assert other._flags() == ("WAIT", False, False, False)
    assert adopted._flags() == ("DEFER_ALL", True, False, True)
