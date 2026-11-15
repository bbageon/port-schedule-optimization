"""Post-hoc check: ranking retention is not the same as regression loss."""
from pathlib import Path
import statistics as st

from check_initial_stock import read,sha,write_result

ROOT=Path('outputs/reports/yr332_network_review')


def main():
    source=ROOT/'saved-pair-retention.json'; rows=read(source)['rows']
    earlier=[r for r in rows if r['split']=='train' and r['iteration']<5]
    groups={}
    for name,group in [('all_earlier_train',earlier),('nonzero_earlier_train',[r for r in earlier if r['active']])]:
        values={}
        for stage in ('after','final'):
            errors=[abs(r['predicted'][stage]-r['target_gap']) for r in group]
            values[stage]=dict(mean_smooth_l1=st.mean(.5*x*x if x<1 else x-.5 for x in errors),
                raw_gap_weighted_accuracy=sum(abs(r['raw_cost_gap'])*r['correct'][stage] for r in group)
                    /sum(abs(r['raw_cost_gap']) for r in group))
        groups[name]=dict(pairs=len(group),values=values)
    write_result(ROOT/'retention-loss-check.json',dict(groups=groups,
        source_sha256=sha(source),script_sha256=sha(Path(__file__)),
        scope='post-hoc interpretation check after observing ranking loss; no model fitting',
        definition='SmoothL1 beta=1 on stored float32 pair gaps; normalized targets vary by original corpus; not terminal cost'))
    print(groups)


if __name__=='__main__': main()
