"""YR-042 — DGT 근사 프로파일·일반화 게이트 계약 테스트."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.experiments.dgt_generalization import (DgtGenConfig,
                                                    quick_dgt_gen_config,
                                                    run_dgt_generalization)
from yard_rl.integrated.profiles import build_dgt_approx_profile


def test_dgt_approx_profile_builds_from_yaml():
    p = build_dgt_approx_profile()
    assert p.terminal_id == "DGT-APPROX-2CR" and p.assumed
    assert p.block.row_count == 10 and p.block.tier_max == 6   # DGT 공식 스펙
    assert len(p.cranes) == 2 and p.cranes[0].gantry_speed_mps == 4.0  # ARMG 문헌
    assert p.transfer.kind == "AGV" and p.transfer.n_units == 3


def test_seed_band_guard():
    with pytest.raises(ValueError):
        DgtGenConfig(train_seed0=300_000)   # YR-039 대역 재사용 금지


def test_quick_run_end_to_end(tmp_path):
    report = run_dgt_generalization(out_dir=str(tmp_path / "out"),
                                    cfg=quick_dgt_gen_config(),
                                    progress=lambda _m: None)
    text = report.read_text(encoding="utf-8")
    assert "DuelingDQN[DGT-retrained]" in text and "guardrail" in text
    assert (tmp_path / "out" / "dgt_generalization_results.json").exists()


def test_hjnc_approx_profile_converges_with_dgt():
    """YR-022 수렴 계약: 근사 수준에서 HJNC ≡ DGT (라벨·fleet 종류 제외)."""
    from dataclasses import asdict
    from yard_rl.integrated.profiles import build_hjnc_approx_profile
    h, d = asdict(build_hjnc_approx_profile()), asdict(build_dgt_approx_profile())
    for skip in ("terminal_id",):
        h.pop(skip), d.pop(skip)
    assert h.pop("transfer")["kind"] == "YT" and d.pop("transfer")["kind"] == "AGV"
    assert h == d   # 나머지 전 필드 수치 동일 — 결과 수렴 예상의 근거
