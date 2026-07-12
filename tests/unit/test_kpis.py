"""queue-area·tail-area 정확 적분 테스트 (05 §1.1)."""
from yard_rl.sim.kpis import KpiTracker


def test_queue_area_single_truck():
    k = KpiTracker(sla_s=100.0)
    k.truck_arrived("J1", 0.0)
    k.integrate(0.0, 150.0)
    assert abs(k.queue_area_s - 150.0) < 1e-9
    assert abs(k.tail_area_s - 50.0) < 1e-9  # SLA(100) 초과분만


def test_queue_area_two_trucks_and_service():
    k = KpiTracker(sla_s=100.0)
    k.truck_arrived("J1", 0.0)
    k.integrate(0.0, 120.0)          # J1: 120
    k.truck_arrived("J2", 120.0)
    k.integrate(120.0, 200.0)        # J1+J2: 2*80
    assert abs(k.queue_area_s - (120.0 + 160.0)) < 1e-9
    # tail: J1 은 100s 부터 → (100~200)=100. J2 SLA 는 220 이후 → 0
    assert abs(k.tail_area_s - 100.0) < 1e-9
    k.service_started("J1", 200.0)
    assert k.wait_samples_s == [200.0]
    k.integrate(200.0, 260.0)        # J2 만: 60
    assert abs(k.queue_area_s - 340.0) < 1e-9
    # J2 tail 시작은 220 → (220~260)=40
    assert abs(k.tail_area_s - 140.0) < 1e-9


def test_tail_boundary_crossing_inside_interval():
    """SLA 경계가 적분 구간 중간에 있어도 초과분만 정확히 계상."""
    k = KpiTracker(sla_s=100.0)
    k.truck_arrived("J1", 50.0)
    k.integrate(50.0, 140.0)   # SLA 경계 150 이전 → tail 0
    assert k.tail_area_s == 0.0
    k.integrate(140.0, 160.0)  # 경계 150 통과 → 10 만 tail
    assert abs(k.tail_area_s - 10.0) < 1e-9


def test_vessel_delay_only_past_deadline():
    k = KpiTracker(sla_s=100.0)
    k.job_completed(external=False, deadline=500.0, end=450.0)
    assert k.vessel_delay_s == 0.0
    k.job_completed(external=False, deadline=500.0, end=620.0)
    assert abs(k.vessel_delay_s - 120.0) < 1e-9
