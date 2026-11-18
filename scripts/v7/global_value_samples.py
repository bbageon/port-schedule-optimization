"""Explicit v2 experience schema: never silently discard critic context or its clock."""
from dataclasses import replace
import math

from resched_ppo import encode_sample as legacy_encode, decode_sample as legacy_decode
from yard_rl.v7.crane.global_context import validate_observation

SCHEMA = 'yard_rl.global_value_experience.v2'


def encode_sample(sample):
    obs = sample.global_observation
    if obs is None:
        raise ValueError('Global experience requires observation evidence')
    validate_observation(obs)
    return dict(schema=SCHEMA, sample=legacy_encode(sample), global_waiting=sample.state.global_waiting,
                global_observation=obs)


def decode_sample(obj):
    if obj['schema'] != SCHEMA:
        raise ValueError('Unsupported global experience schema')
    obs = obj['global_observation']; validate_observation(obs)
    s = legacy_decode(obj['sample'])
    if abs(s.state.time_fraction * 86400 - obs['decision_time']) > 1e-6:
        raise ValueError('Decision state and global observation clock differ')
    value = obj['global_waiting']
    if value is not None and (not math.isfinite(value) or value not in (0., obs['feature'])):
        raise ValueError('Global neural input differs from observed feature')
    s.state.global_waiting = value
    s.global_observation = obs
    return s


def condition_sample(sample, mode):
    if mode not in ('zero', 'global') or sample.global_observation is None:
        raise ValueError('Valid context condition and observation required')
    validate_observation(sample.global_observation)
    value = 0. if mode == 'zero' else sample.global_observation['feature']
    return replace(sample, state=replace(sample.state, global_waiting=value))
