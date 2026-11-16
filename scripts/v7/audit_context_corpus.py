"""Count distinct supervised decisions; candidate rows are not independent labels."""
from collections import Counter
import statistics as st
import numpy as np

from context_learning import OUT,collection_cases,examples
from six_step_learning import gz_read,sha
from yard_rl.integrated.repro import write_result


def main():
    rows=[];train_features=[];indistinguishable=[]
    for i,case in enumerate(collection_cases()):
        path=OUT/'corpora'/f'{i:02d}.json.gz';data=gz_read(path)
        assert data['case']==case
        records=examples(data);stats=data['result']['crane']
        keys=[tuple(e['key']) for e in records]
        assert len(set(keys))==len(keys)
        assert stats['labeled']==len(records)
        assert all(len(r)==10 for e in records for r in e['rows'])
        counts=[len(e['rows']) for e in records]
        reasons=Counter(a['reason'] for q in stats['witness_search'] for a in q['attempts'])
        failed_same=sum(q['witness'] is None and bool(q['attempts']) and
                        all(a['reason']=='same_input_as_picked' for a in q['attempts'])
                        for q in stats['witness_search'])
        for e in records:
            pair=np.asarray([e['rows'][j] for j in e['pair']],dtype=np.float32)
            if case['split']=='train':train_features.extend(pair.tolist())
            if np.array_equal(pair[0],pair[1]) and abs(e['gap'])>1e-9:
                indistinguishable.append(dict(case=case,key=e['key'],flow=e['flow'],gap=e['gap']))
        rows.append(dict(case=case,pairs=len(records),nonzero=sum(abs(e['gap'])>1e-9 for e in records),
            seen=stats['seen'],usable=stats['usable'],sampled=stats['sampled'],
            no_witness=stats['no_witness'],witness_attempt_reasons=dict(reasons),
            failed_questions_only_same_input=failed_same,
            pairs_by_flow=dict(Counter('/'.join(e['flow']) for e in records)),
            context_size=dict(min=min(counts),median=st.median(counts),max=max(counts)),
            maximum_abs_gap=max(abs(e['gap']) for e in records),seconds=data['seconds'],sha256=sha(path)))
    by_split={split:dict(pairs=sum(r['pairs'] for r in rows if r['case']['split']==split),
                        nonzero=sum(r['nonzero'] for r in rows if r['case']['split']==split),
                        decisions_seen=sum(r['seen'] for r in rows if r['case']['split']==split))
              for split in ('train','selection')}
    from yard_rl.v7.crane.features import CRANE_FEATURES
    x=np.asarray(train_features,dtype=np.float32)
    feature_stats=[dict(name=name,min=float(x[:,i].min()),max=float(x[:,i].max()),
                        unique=len(np.unique(x[:,i])),zero_share=float((x[:,i]==0).mean()))
                   for i,name in enumerate((*CRANE_FEATURES,'is_store'))]
    result=dict(rows=rows,by_split=by_split,training_pair_feature_statistics=feature_stats,
                identical_float32_features_with_nonzero_gap=indistinguishable,
                question_counts=dict(maximum=48*9,asked=sum(r['sampled'] for r in rows),
                                     labeled=sum(r['pairs'] for r in rows),no_witness=sum(r['no_witness'] for r in rows)),
                definitions=dict(raw_gap_zero_tolerance=1e-9,normalized_fitting_zero_tolerance=1e-9,
                    no_witness='No witness under a single shared-score promotion, not proof of global unreachability',
                    identical_input='Teacher already excludes identical float32 inputs; zero contradictions is conditional on this filter'),
                half_train_pairs=sum(max(1,r['pairs']//2) for r in rows if r['case']['split']=='train'),
                unit='one counterfactual decision pair, not candidate rows or duplicated minibatch draws')
    write_result(OUT/'corpus-audit.json',result)
    print(by_split)


if __name__=='__main__':main()
