"""Positive, conditional within-hour intensities with two explicit time scales."""
from __future__ import annotations

import math


def validate(parameters):
    keys={'a1','a2','sigma','innovation','burn_in','coarse_sigma'}
    if not isinstance(parameters,dict) or set(parameters)!=keys:
        raise ValueError('complete correlated minute parameters required')
    if any(not isinstance(parameters[k],(int,float)) or not math.isfinite(parameters[k]) for k in keys):
        raise ValueError('finite correlated minute parameters required')
    a,b=parameters['a1'],parameters['a2']
    if not(abs(b)<1 and a+b<1 and b-a<1):raise ValueError('nonstationary AR(2) coefficients')
    r1=a/(1-b);r2=a*r1+b;variance=1-a*r1-b*r2
    if variance<=0 or not math.isclose(parameters['innovation']**2,variance,rel_tol=1e-8):
        raise ValueError('innovation does not preserve unit stationary variance')
    if parameters['sigma']<0 or parameters['coarse_sigma']<0 or type(parameters['burn_in']) is not int or parameters['burn_in']<128:
        raise ValueError('nonnegative scales and at least 128 burn-in steps required')


def weights(rng,parameters):
    validate(parameters)
    a,b,sigma,innovation=[parameters[k] for k in ('a1','a2','sigma','innovation')]
    z0=z1=0.;latent=[]
    for t in range(parameters['burn_in']+60):
        z=a*z1+b*z0+innovation*rng.gauss(0.,1.);z0,z1=z1,z
        if t>=parameters['burn_in']:latent.append(sigma*z)
    coarse=[rng.gauss(0.,parameters['coarse_sigma']) for _ in range(6)]
    values=[z+coarse[m//10] for m,z in enumerate(latent)];maximum=max(values)
    positive=[math.exp(z-maximum) for z in values];total=sum(positive)
    return [w/total for w in positive]
