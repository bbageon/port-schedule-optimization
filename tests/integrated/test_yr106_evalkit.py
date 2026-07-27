"""YR-106 evalkit 계약 — 채널 완전분할·MDE·라벨 분리·guard 기계화."""
import pytest

from yard_rl.contract import COST_TERMS
from yard_rl.integrated.evalkit import (CHANNELS, GuardReport, channel_split, check_guards,
                                        paired, paired_by_channel, prereg_power_note)


def test_channels_partition_cost_terms():
    """채널이 13항을 겹침·누락 없이 완전분할해야 Σ채널 == total 이 성립한다."""
    seen = [t for terms in CHANNELS.values() for t in terms]
    assert sorted(seen) == sorted(COST_TERMS)
    assert len(seen) == len(set(seen))


def test_channel_split_sums_to_total():
    contribs = {t: float(i + 1) for i, t in enumerate(COST_TERMS)}
    split = channel_split(contribs)
    assert sum(split.values()) == pytest.approx(sum(contribs.values()))
    assert split["truck"] == pytest.approx(contribs["truck_wait"] + contribs["long_wait"])


def test_paired_reports_mde_and_required_n():
    d = [-3.0, -2.0, -4.0, -1.0, -3.5, -2.5, -3.0, -2.0]
    r = paired(d)
    assert r.n == 8 and r.mean < 0
    assert r.mde80 > 0
    assert r.required_n_for_mean is not None and r.required_n_for_mean >= 2


def test_label_separates_no_effect_from_detection_failure():
    """감사 조치 ②의 핵심: CI 가 0 을 포함해도 두 경우를 **사전 δ 기준**으로 구분한다."""
    quiet = [0.4, -0.3, 0.2, 0.1, -0.2, 0.0, 0.3, -0.1]        # 잡음 작음 → MDE 작음
    assert paired(quiet, delta_interest=3.0).label.startswith("효과 없음")
    noisy = [12.0, -11.0, 9.0, -8.0, 10.0, -9.0, 11.0, -10.0]  # 잡음 큼 → MDE 큼
    assert paired(noisy, delta_interest=3.0).label.startswith("검출 실패")
    # δ 미지정이면 '효과 없음'을 주장할 수 없다
    assert paired(quiet).label.startswith("미검출")
    assert paired([-5.0] * 8, delta_interest=3.0).label.startswith("유의(개선)")


def test_label_never_claims_no_effect_without_delta():
    """사후검정력 오류 방지 — 관측 평균이 아무리 작아도 δ 없이는 '효과 없음' 금지."""
    r = paired([0.001, -0.001, 0.0, 0.001, -0.001, 0.0, 0.001, -0.001])
    assert r.label.startswith("미검출")            # 주장은 '미검출'까지만
    assert "주장 불가" in r.label                   # '효과 없음' 은 금지 문구로만 등장


def test_paired_by_channel_pairs_and_covers():
    treat = [{"truck": -3.0, "vessel": 10.0, "move": 0.1, "other": 0.0} for _ in range(6)]
    ctrl = [{"truck": 0.0, "vessel": 0.0, "move": 0.0, "other": 0.0} for _ in range(6)]
    out = paired_by_channel(treat, ctrl)
    assert set(CHANNELS).issubset(out)
    assert out["truck"]["mean"] == pytest.approx(-3.0)
    assert out["vessel"]["mean"] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        paired_by_channel(treat, ctrl[:3])


def test_guards_fail_closed_on_missing_fields():
    """수집조차 안 한 하네스가 침묵 통과하면 안 된다 (감사: YR-103 사례)."""
    rep = check_guards([{"compl": 1.0}])            # backlog 미수집
    assert not rep.ok and any("backlog" in f for f in rep.failures)
    rep2 = check_guards([{"compl": 1.0, "backlog": 0}])
    assert rep2.ok
    rep3 = check_guards([{"compl": 0.98, "backlog": 0}])
    assert not rep3.ok
    with pytest.raises(AssertionError):
        rep3.raise_if_failed()


def test_prereg_power_note_requires_bigger_confirm_band():
    note = prereg_power_note([-3.0, -2.0, -4.0, -1.0, -3.5, -2.5, -3.0, -2.0],
                             target_effect=-3.0)
    assert note["required_n_confirm"] == note["required_n_select"] * 2
    assert "확증 대역" in note["rule"]
