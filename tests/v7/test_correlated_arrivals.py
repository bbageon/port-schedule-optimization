import random

import pytest

from yard_rl.v7.correlated_arrivals import validate,weights


def parameters():
    a,b=-.3,.1;r1=a/(1-b);r2=a*r1+b
    return dict(a1=a,a2=b,sigma=.4,innovation=(1-a*r1-b*r2)**.5,burn_in=128,coarse_sigma=.1)


def test_valid_positive_replay():
    p=parameters();w=weights(random.Random(17),p)
    assert len(w)==60 and sum(w)==pytest.approx(1) and min(w)>0
    assert w==weights(random.Random(17),p)
    assert w!=weights(random.Random(18),p)


@pytest.mark.parametrize('change',[{'a1':2},{'innovation':2},{'sigma':-1},{'burn_in':127},{'coarse_sigma':float('nan')}])
def test_invalid_parameters(change):
    with pytest.raises(ValueError):validate(dict(parameters(),**change))
