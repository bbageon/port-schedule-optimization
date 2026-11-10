"""New exact receipts must not change the clocks of saved, unversioned inputs."""
import pytest

from yard_rl.v7.information_contract import NOTICE_SCHEMA, announcement_time
from yard_rl.v7.stage.orders import orders_from_schedule


def test_legacy_floor_is_applied_before_rounding():
    e=dict(arrival_s=60-1e-8,lead_s=0)
    assert announcement_time(e)==0
    assert announcement_time(dict(e,notice_schema=NOTICE_SCHEMA))==60


@pytest.mark.parametrize('exact,expected',[(False,120.0),(True,120.000356)])
def test_public_receipt_precision_is_versioned(exact,expected):
    e=dict(job_id='one',con_no='IN_one',flow='GATE_IN',block='Y01',
           arrival_s=120.123456,lead_s=.1231,reserved_s=100,
           reservation_schema='all-reserved-synthetic-v1')
    if exact:e['notice_schema']=NOTICE_SCHEMA
    orders,records=orders_from_schedule({'schedule':[e]})
    assert orders['one'].copino_notice_s==records['one'].copino_notice_s==expected
