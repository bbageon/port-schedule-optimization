"""YR-073 — 순위 증류 계약 (torch 필요 — Windows 미설치 환경은 skip)."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.integrated.joint_distill import (  # noqa: E402
    JointDecisionSample, combo_matrix, load_student, save_student, top1_agreement,
    train_joint_net)


def _sample(rng, disagree=False):
    """합성 표본 — 교사는 (candA+candB 합) 최소 조합 선택 (학습 가능 신호)."""
    ka, kb = 4, 3
    canda = tuple(tuple(rng.uniform(-1, 1) for _ in range(6)) for _ in range(ka))
    candb = tuple(tuple(rng.uniform(-1, 1) for _ in range(6)) for _ in range(kb))
    combos = tuple((i, j) for i in range(ka) for j in range(kb))
    key = [sum(canda[i]) + sum(candb[j]) for i, j in combos]
    teacher = min(range(len(combos)), key=lambda x: key[x])
    return JointDecisionSample(
        ga=(0.1, 0.2), yca=(0.3,), qa=(0.4, 0.5), canda=canda,
        ycb=(0.6,), qb=(0.7, 0.8), candb=candb, combos=combos,
        tier_a=tuple(key), teacher_pos=teacher, sf_pos=0,
        disagree=disagree or teacher != 0)


def test_combo_matrix_layout():
    import random
    s = _sample(random.Random(1))
    x = combo_matrix(s)
    assert x.shape == (12, 2 + 1 + 2 + 6 + 1 + 2 + 6)
    i, j = s.combos[5]
    row = list(x[5])
    assert row[:2] == pytest.approx([0.1, 0.2])          # 공유 global 선두
    assert row[5:11] == pytest.approx(list(s.canda[i]))  # crane-A 후보 블록
    assert row[14:] == pytest.approx(list(s.candb[j]))   # crane-B 후보 블록


def test_train_learns_teacher_ranking():
    import random
    rng = random.Random(73)
    samples = [_sample(rng) for _ in range(60)]
    tr = train_joint_net(samples, epochs=25, seed=7)
    ag = top1_agreement(tr.net, samples)
    assert ag["top1_all"] >= 0.8                          # 순위 신호 학습 계약
    assert ag["n_disagree"] == sum(s.disagree for s in samples)


def test_save_load_roundtrip(tmp_path):
    import random
    rng = random.Random(5)
    samples = [_sample(rng) for _ in range(10)]
    tr = train_joint_net(samples, epochs=2, seed=3)
    p = tmp_path / "student.pt"
    save_student(p, tr, {"global.now_s": 3600.0})
    net, norm = load_student(p)
    x = combo_matrix(samples[0])
    with torch.no_grad():
        a, _ = tr.net(x)
        b, _ = net(x)
    assert torch.allclose(a, b)
    assert norm.refs["global.now_s"] == pytest.approx(3600.0)
