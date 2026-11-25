"""Matched BP dispatch without the block-wait priority tier."""
from dataclasses import replace

from .on_policy import OnPolicyResolver


class NoWaitPromotionResolver(OnPolicyResolver):
    def resolve(self, sim, decision, gen_by_crane):
        # The month replay supplies every candidate. Preserve identifiers, physical plans
        # and masks, removing only the block-wait flag before BP ranks feasible actions.
        candidates = {}
        for crane, group in gen_by_crane.items():
            if any(item.mandatory for item in group.items):
                items = tuple(replace(item, mandatory=False) if item.mandatory else item
                              for item in group.items)
                group = replace(group, items=items)
            candidates[crane] = group
        return super().resolve(sim, decision, candidates)
