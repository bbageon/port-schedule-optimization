"""Build a review working draft; preserve submitted and earlier narrative artifacts."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import re
import shutil
import subprocess

OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ABSTRACT = r"""\begin{abstract}
Terminal assignments can become inefficient as yard and gate
congestion changes. We study an architecture that revisits the yard block
and arrival time of an announced truck job. Separate proposal and
acceptance networks learn pair-centred cost scores from simulation
branches with delayed consequences; a central procedure checks mutual
acceptance and simulated resource feasibility. Evaluation uses fixed
networks without online counterfactual simulation. In the original
single continuous synthetic run, the joint policy reduced modelled
terminal cost by 14.72\% over the middle 28 days, compared with 13.44\%
for time-only adjustment; four high-volume days contributed 90.5\% of
the savings. These are descriptive observations from one trajectory,
not independent-run evidence. We specify a registered evaluation of
20 independent months with four fixed-policy variants; its aggregate
results remain pending in this working draft. Appointment quotas are
unbounded and external waiting and rescheduling costs are excluded.
The architecture is positioned as a decision-support component for
terminal staff and carrier dispatchers; neither deployment feasibility
nor user benefits have been established.
\keywords{Container terminal, Truck appointment system, Yard block
reallocation, Counterfactual cost learning, Reinforcement learning}
\end{abstract}
\noindent\textit{Working draft, 18 September 2026. Independent evaluation
and the validation listed in Sect.~\ref{sec:pending-evidence} are pending.}

"""

INTRO = r"""\section{Introduction}
\label{sec:intro}
Terminal automation must turn information into coordinated operational
decisions~\cite{ref1}. Appointment systems address arrival windows and
carrier--terminal coordination~\cite{ref4,ref6}; yard scheduling addresses
placement, sequencing, and interference~\cite{ref7,ref8,ref9,ref16}.
Reinforcement learning also supports multi-block crane dispatching~\cite{ref19}.

We ask whether an announced job's block or arrival time should change
as congestion develops. Effects on later trucks, cranes, and vessels
make immediate movement cost insufficient. Difference rewards and
counterfactual baselines motivate matched action comparisons~\cite{ref10,ref11}.
Our separate proposal and acceptance networks precede central commitment;
Fig.~\ref{fig:arch} distinguishes offline learning from online decisions.

The contributions are the assignment architecture, its pairwise
learning signal, and an evaluation protocol separating action-range
effects from within-run dependence. Historical results are a synthetic
proof of concept, not evidence of consistent benefits or component
necessity. Sect.~\ref{sec:user-context} describes the intended
decision-support users; field and human benefits remain untested.

"""

RELATED = r"""\section{Related Work}
\label{sec:related}
Appointment models redistribute arrivals~\cite{ref4}, while
collaborative scheduling represents the interests of carriers and
terminals~\cite{ref6}. Empirical time-of-day models motivate nonuniform
arrival profiles~\cite{ref13}. Appointment quotas influence truck turn
time and crane utilisation and require balancing terminal and trucker
interests~\cite{refquota}. Our unbounded-quota experiments therefore
address adjustment potential under stated assumptions.

Yard research studies crane sequencing~\cite{ref7},
interference~\cite{ref8}, block length and crane deployment~\cite{ref9},
and demand-dependent dispatching performance~\cite{ref16}. Recent
reinforcement learning addresses multi-block crane
dispatching~\cite{ref19}. We instead revise the assignment supplied to
the execution layer, while keeping that layer's dispatching rule fixed
within a comparison.

Difference rewards and counterfactual multi-agent learning isolate an
actor's contribution to a shared outcome~\cite{ref10,ref11}. Here the
baseline comes from a simulation branch, and the target is a finite-
horizon cost difference. Unlike temporal-difference targets in deep
Q-learning~\cite{refmnih}, it does not bootstrap from the network's own
future estimates. Small multilayer perceptrons~\cite{refhornik} map
current observations and candidate features to scalar scores. General
approximation capacity does not guarantee correct candidate ordering;
Sect.~\ref{sec:arch-models} states the conditions and remaining gap.

"""

USERS = r"""\subsection{Intended Users and Operational Context}
\label{sec:user-context}
Terminal yard planners and operations staff are the intended users,
with carrier dispatchers and drivers. Busan New Port waiting-time
research~\cite{refbusanwait}, Korean terminal operating system (TOS)
research~\cite{reftos}, and the Busan Port Authority's AllCON-e booking
and information services~\cite{refallcone} motivate this proposed
application; integration is untested.

A planner would compare the original and recommended block/time and
observed congestion, then accept, modify, or reject the recommendation.
A carrier dispatcher would check vehicle availability and subsequent
jobs, negotiate an acceptable window, or decline. The driver would
receive the agreed assignment and report execution difficulties.
Operator review precedes carrier agreement and driver notification;
without agreement, the original assignment remains pending review.
Human response deadlines are separate from the 60-second review interval.

Network acceptance is not human consent, and scores are not calibrated
monetary savings or confidence probabilities. The evaluated decision
engine supplies no evidence about interfaces, usability, or human
acceptance; these require integration and user evaluation.

"""

PROTOCOL = r"""\subsection{Historical Run and Independent Evaluation}
\label{sec:independent-protocol}
Historical results use one continuous 30-day trajectory (seed
9{,}900{,}950), measured over days 2--29. Common exogenous streams
support matched comparisons~\cite{ref22}, but state carries across
days. Daily wins are descriptive; day-level tests and day-resampled
intervals are not independent-replication evidence. The first-iteration
checkpoint already contains one update and is not an untrained model.

The revision fixes 20 seeds, starting at 20{,}000{,}000 with spacing
100{,}000. Each policy--seed run starts a fresh engine for 30 continuous
days plus the existing two-hour closing interval. Arrival curves,
daily-volume probabilities, truck attributes, and initial inventory
are preserved and matched within seeds. Weights are fixed; exploration,
updates, and training branches are disabled. This evaluates variation
across environments for one model, not across training runs.

Legacy admission could omit requests lacking stock or reservations.
The revision registers all requests and shares a fixed correction to
selected vessel loading/discharging directions across policies,
recording both input versions.
Historical and revised costs are not pooled. Completed, unfinished,
and unbound requests and vessel work accompany costs; valid records
do not imply full service. Outbound jobs retain their block, but
synthetic targets may be rebound within it rather than representing
fixed customer-container identities.

No reallocation, Full, and Time-only comprise the original 60 runs.
Block-only adds the same 20 seeds, totalling 80 runs, under a separate
17 September registration after one baseline month and partial results
were available. All variants share the checkpoint; action restrictions
are not separately optimised policies. The comparison families remain
separate.

For policy $p$ and seed $s$, let $C_{p,s}$ be the cost over the
28 measurement days. The two original primary differences are
\begin{equation}
D_s^{B}=C_{\mathrm{baseline},s}-C_{\mathrm{Full},s},\qquad
D_s^{T}=C_{\mathrm{Time},s}-C_{\mathrm{Full},s}.
\end{equation}
Positive values favour Full. We resample the 20 paired months
20{,}000 times (seed 9{,}900{,}721), retaining within-month dependence.
Each primary mean difference has a 97.5\% percentile interval;
secondary intervals are descriptive 95\%. The separate Block-only
family uses seed 9{,}900{,}723 and 97.5\% intervals for baseline minus
Block-only and Block-only minus Full. Two separate families do not
provide one four-claim error guarantee. Supplementary 30-day totals
are not longer simulations. Per-seed costs, percentage savings,
aggregate ratios, wins/ties/losses, and residuals require complete
matched sets with no outcome-based sample changes. Results are pending.

Daily records contain volume, starting backlog, block queues, unfinished
crane work, mean/90th-percentile truck turn time, action counts, and
costs. Planned demand defines descriptive groups; realised queues
interpret congestion without redefining groups after observing outcomes.
The 600 days are not independent replications.

"""

RESULTS = r"""\section{Results and Evidence Status}
\label{sec:exp}
\subsection{Historical Cost Comparison}
\label{sec:exp-paired}
Table~\ref{tab:main} preserves the legacy single-run results:
approximately KRW 10.00\,billion without reallocation and
KRW 8.52\,billion with Full over 28 days. They use the historical
contract, not the ongoing independent evaluation.
\begin{table}[htbp]
\centering
\caption{Historical 28-day comparison. Positive reduction means lower
recorded cost than no reallocation. Daily wins are descriptive, without
independent-sample significance; dashes denote unreported values.}
\label{tab:main}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llrc@{}}
\toprule
Policy & Actions & Reduction & \shortstack{Full cheaper\\(days)} \\
\midrule
\textbf{Full} & block+time & \textbf{+14.72\%} & --- \\
Time-only & time & +13.44\% & --- \\
Block-only & block & $-$1.31\% & --- \\
First-iteration model & block+time & +7.97\% & 18/28 \\
\midrule
Rule: least-loaded slot & time & +3.41\% & --- \\
Rule: least-loaded block+slot & block+time & +1.51\% & 20/28 \\
Rule: least slack & block & +0.53\% & 20/28 \\
Rule: first-come-first-served & block & +0.17\% & 22/28 \\
Rule: shortest processing time & block & +0.05\% & 21/28 \\
Rule: net gain & block & $-$0.48\% & 21/28 \\
\midrule
No reallocation & --- & 0.00\% & 26/28 \\
\bottomrule
\end{tabular}
\end{table}

Full's additional reduction over Time-only is only 1.28 percentage
points relative to the baseline. Block-only increases aggregate cost.
Neither finding establishes a general need for spatial adjustment;
the registered direct month-level comparison addresses that question.
The least-loaded-slot rule saves 3.41\% without learning. Historical
Time-only is cheaper than this rule on just 8/28 days, yet has a lower
aggregate cost by about KRW 1.00\,billion. This illustrates why frequency
of wins and magnitude of savings must be reported together, without
treating dependent days as independent evidence.

\subsection{Demand Level and Action Contributions}
\label{sec:exp-decomp}
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig-decomposition.pdf}
\caption{Historical cost reduction by planned demand, in million KRW.
Bars compare each restricted policy with no reallocation; the diamond
compares Full with the first-iteration checkpoint. The shaded demand
levels contain four days. Groups have unequal numbers of dependent
days; their totals are descriptive, not comparable daily averages.}
\label{fig:decomp}
\end{figure}
Four high-volume days contribute 90.5\% of historical savings.
Time-only and Block-only save 1.344 and $-0.131$ billion KRW,
respectively. The interaction residual of 0.258 billion KRW is
arithmetic, not a verified component effect. Block-only saves cost at
3{,}500 and 5{,}000 trucks but loses at 15{,}000, where Time-only beats
Full by approximately KRW 25\,million. Spatial benefits are therefore
conditional. Volume alone does not establish congestion when backlog
persists across days.

The existing sweep varies high-volume days at 2, 6, and 14 out of 28
on seed 9{,}900{,}980. Time-only accounts for 67\%, 79\%, and 81\%
of Full's savings. Spatial action shares, recalculated over the same
28 days, are 31.60\%, 20.77\%, and 11.16\%. The four-day case uses a
different seed and is not a fourth controlled point. This sweep does
not validate other peak widths, costs, or horizons, or a hand-written
low/high-congestion switching rule.

\subsection{Learning Diagnostics}
\label{sec:exp-learning}
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig-learning-curve-2p.pdf}
\caption{Historical training and validation losses over 30 iterations.
Thin lines are iteration values and thick lines five-iteration means;
the vertical axis is logarithmic. Shading marks demand at least
12{,}500. These diagnostics concern one training trajectory, not
independent training replications or full-candidate ranking accuracy.}
\label{fig:learn}
\end{figure}
From the first to last five iterations, proposal training and validation
losses fall from (0.079, 0.268) to (0.029, 0.119); acceptance losses
fall from (0.171, 0.315) to (0.027, 0.136), using 4{,}671
counterfactual worlds. Pair-fit diagnostics do not prove global ranking.
The first-iteration checkpoint costs KRW 9.20\,billion; later updates
save another KRW 0.675\,billion, 94.6\% on four high-volume days.
This requires independent runs, not post-hoc selection of favourable
days. Day-resampled intervals do not establish learning effects, and
frozen-model evaluation does not test training-seed stability.

\subsection{Outstanding Evidence and Reporting Boundaries}
\label{sec:pending-evidence}
The 80-run comparison remains in progress; no independent estimate or
interval is supplied here. Costs require joint interpretation with
completed and unfinished work. The four policies do not isolate the
acceptance network: veto removal retains score-based ordering.
A separate ablation must distinguish these functions while preserving
physical feasibility checks.

Candidate ranking remains unvalidated. Tests must compare KEEP and all
legal actions from the same held-out state, external inputs, continuation
policy, and horizon. Outcomes include selection regret (selected cost
minus the minimum candidate cost), best-action selection, and ranking
agreement. Exact action execution and factual replay across day
boundaries must first be verified.

Online latency is unmeasured. State and candidate construction, features,
scoring, conflict resolution, and commitment need timing apart from
simulation advancement, with hardware, threads, candidate counts,
warm-up, median, upper percentiles, maximum, and 60-second overruns.
Network size and month runtime are not latency evidence.

The one-axis synthetic layout does not validate distinct horizontal
and vertical handling systems. Their travel paths, transfer points,
and crane-resource sharing require explicit comparison; rotating
coordinates is insufficient. Hybrid layouts are excluded. Cost-weight,
peak-width, and longer-run robustness also remain untested by the
existing frequency sweep.

"""

CONCLUSION = r"""\section{Conclusion}
\label{sec:concl}
Separate proposal and acceptance policies learn from counterfactual
cost differences to revise spatial--temporal assignments. Historical
savings are concentrated in high-volume days, with conditional spatial
contributions. Consistent improvement, component necessity, and
operational feasibility remain unestablished; the independent-month
results are pending.

Deployment would require finite booking quotas, carrier-approved
windows, minimum notification time, and cargo/vessel deadlines in a
feasibility layer~\cite{refquota,ref4,ref6}, together with external
waiting, rescheduling, and subsequent-trip costs. Those require
terminal and carrier data. We have not tested savings under those
constraints or estimated a break-even rescheduling charge. Remaining
work includes candidate ranking, acceptance ablations, cost and horizon
sensitivity, online timing, and distinct basic layouts. TOS and
AllCON-e provide an application context; the proposed operator--carrier--
driver workflow still requires integration and user evaluation.

"""


def build():
    source = (OUT / 'main-revised.tex').read_text(encoding='utf8')
    head = source[source.index(r'\documentclass'):source.index(r'\begin{abstract}')]
    arch = source[source.index(r'\section{Proposed Architecture}'):source.index(r'\subsection{Intended Users')]
    arch = arch.replace(r'\begin{figure}[bp]', r'\begin{figure}[htbp]')
    a, b = arch.index(r'\caption{'), arch.index(r'\label{fig:arch}')
    arch = arch[:a] + r"""\caption{(a) Every 60\,s, the online path checks eligibility,
proposal, acceptance, and commitment. (b) The simulator links gate,
blocks, cranes, and vessels to cost $\Phi$. (c) Offline learning compares
up to three worlds from the same state and random stream over 3\,h
to label the two networks.}
""" + arch[b:]
    a, b = arch.index('The decision interval is'), arch.index('The proposal policy scores')
    arch = arch[:a] + r"""Every $\delta=60$\,s, eligible announced jobs are reviewed once.
Job $i$ has assignment $z_i=(b_i,\tau_i)$ and legal actions
$a_i=(b'_i,\Delta_i)\in\mathcal A_i(t)$, producing
$z_i(a_i)=(b'_i,\tau_i+\Delta_i)$. KEEP is $a_i^0=(b_i,0)$.
Full selects among spatial and temporal alternatives; each changed
candidate alters one axis: block or arrival delay ($\Delta_i>0$).
Outbound jobs retain their assigned block. Let $n_i$ be announcement
time, $g_i$ gate-in time, and $r_i$ the prior-review indicator. With
$W=1800$\,s and $w_i=\min(W,\tau_i-n_i)$, eligibility is
\begin{equation}
\mathcal E_t=\{i\mid w_i-\delta<\tau_i-t\le w_i,\ \tau_i>t,\ g_i>t,\ r_i=0\}.
\end{equation}
Review occurs within one decision interval after entering this window,
approximately $W$ before arrival when notice is sufficiently early.
Jobs announced at or after arrival are ineligible; the lead-time model
assigns 14.1\% of demand to that category. This review window is an
algorithmic choice, not a validated carrier notification allowance.
""" + arch[b:]
    old = ("lexicographic tie-breaking. Because scores are pair-centred\n"
           "(Sect.~\\ref{sec:arch-labels}) they are not calibrated across jobs, so this\n"
           "order fixes a deterministic sequence rather than a priority.")
    new = ("lexicographic tie-breaking. The pair-centred scores are not calibrated\n"
           "across jobs (Sect.~\\ref{sec:arch-labels}). Their order is deterministic,\n"
           "but can change which proposals obtain a contested resource; it is\n"
           "therefore an algorithmic priority, not a validated ordering of monetary\n"
           "benefits. Its effect must be separated from the acceptance veto.")
    assert old in arch
    arch = arch.replace(old, new)
    old = ("list may change without retraining. Mean subtraction inside a pair\n"
           "removes a constant that is common to both actions and therefore leaves\n"
           "the ranking in~(2) unchanged.")
    new = r"""list may change without retraining. Pair centring preserves the ordering
inside one pair, but does not by itself justify all comparisons in~(2).
For a fixed state and a common KEEP outcome $K=C(s,\mathrm{KEEP})$,
the ideal label of a changed candidate is
\begin{equation}
y(s,a)=\frac{C(s,a)-K}{2S},\qquad S=100{,}000\;\mathrm{KRW}.
\end{equation}
Changed candidates preserve their true cost order only when the state,
exogenous trajectory, continuation policy, and horizon share that same
reference. The learned predictions need not preserve it. Moreover,
KEEP's target $(K-C(s,a))/(2S)$ depends on the paired alternative,
even when its input is unchanged. Ordering changed candidates and
deciding against a learned KEEP score are therefore distinct validation
questions. This conditional explanation does not establish measured
ranking accuracy."""
    assert old in arch
    arch = arch.replace(old, new)
    arch = arch.replace('a change the alternative keeps the assignment, if it kept the assignment\n'
                        'the alternative commits the cheapest candidate under the current\n'
                        'weights, and the acceptance alternative swaps acceptance and rejection.',
                        'a change the alternative keeps the assignment; if it kept the assignment,\n'
                        'the alternative forces a proposal selected by the current weights.\n'
                        'Acceptance and feasibility still govern actual commitment. The acceptance\n'
                        'alternative swaps acceptance and rejection.')
    arch = arch.replace('online operation costs one forward pass per candidate.',
                        'network scoring is only part of the online path. Candidate construction,\n'
                        'conflict resolution, and commitment must also be timed.')
    a, b = arch.index('The labels are not a fixed dataset:'), arch.index('Training both networks')
    arch = arch[:a] + ('States and alternatives change with the current policy; each iteration\n'
                      'discards its labels after updating. This is finite-horizon Monte-Carlo\n'
                      'cost learning with simulator-generated baselines, without target\n'
                      'bootstrapping or a policy-gradient update.\n') + arch[b:]
    # Numbered equation references in the original were hard-coded. The inserted
    # ranking equation comes after (2), so the existing policy reference is stable.
    env = source[source.index(r'\section{Experimental Environment'):source.index('The evaluation uses a 30-day continuous run')]
    env = env.replace('and two of the four cost coefficients.',
                      'and the cost conversions identified below.')
    env = env.replace('Announcement times follow', 'Requested arrival times follow')
    env = env.replace('Experimental setup. Values marked \\emph{design} are choices of\n'
                      'this study; the others lie within ranges reported in the cited work.',
                      'Simulation and learning settings. Design values, including\n'
                      'literature-informed conversions, are not measurements at a specific\n'
                      'terminal. Historical and revision evaluations are listed separately.')
    env = env.replace(' & Slots per block, initial occupancy & 1{,}440, 65\\% \\\\',
                      ' & Slots per block, initial fill (design) & 1{,}440, 45\\% \\\\')
    env = env.replace('Cost & Truck dwell &', 'Cost & Truck dwell (design) &')
    env = env.replace('Evaluation & Days, compared, seed & 30, middle 28, 9{,}900{,}950 \\\\',
                      'Evaluation & Historical days, compared, seed & 30, middle 28, 9{,}900{,}950 \\\\\n'
                      ' & Registered revision & 20 months, 4 fixed-policy variants \\\\')
    env = env.replace('lists the values used within them. Following',
                      'lists the settings. The revision generator uses a 45\\% initial fill;\n'
                      'the earlier manuscript\'s 65\\% entry is corrected rather than treated\n'
                      'as a new occupancy experiment. Following')
    env = env.replace('The total cost over $[d,d+1)$ is the sum of four terms, where $h_i$ is',
                      'The cost attributed to day $d$ has four terms. Truck costs follow\n'
                      'the original requested day, while the other terms follow calendar\n'
                      'time; consequently this is not a purely calendar-day cash-flow total.\n'
                      'Here $h_i$ is')
    env = env.replace('can change the net benefit of a time adjustment. Waiting dominates:',
                      'can change the net benefit of a time adjustment. In the historical run,\n'
                      'waiting dominates:')
    a, b = env.index('We evaluate the architecture'), env.index(r'\begin{table}')
    env = env[:a] + r"""The synthetic setting draws on reported block counts~\cite{ref16},
cranes~\cite{ref8}, multi-week operation and handling times~\cite{ref9},
and arrival variation~\cite{ref13}. Table~\ref{tab:setup} distinguishes
design choices from literature-informed values~\cite{ref20}; it does
not reproduce a particular terminal. The generator's 45\% initial fill
corrects the earlier 65\% manuscript entry, not a new occupancy experiment.
""" + env[b:]
    a = env.index('waiting dominates:')
    env = env[:a] + r"""truck waiting contributes 96.4\% of cost, making results especially
sensitive to its coefficient and the excluded external burdens.

"""
    tail = source[source.index(r'\begin{credits}'):]
    text = ('% Integrated review working draft; pending evidence is not a completed result.\n'
            + head + ABSTRACT + INTRO + RELATED + arch + USERS + env + PROTOCOL + RESULTS + CONCLUSION + tail)
    # Bibliography remains the same 22 sources, reordered by their new first use.
    body = text.split(r'\begin{thebibliography}', 1)[0]
    cited = list(dict.fromkeys(k.strip() for g in re.findall(r'\\cite\{([^}]+)\}', body) for k in g.split(',')))
    entries = {k: line for line, k in re.findall(r'(\\bibitem\{([^}]+)\}[^\n]+)', text)}
    assert set(cited) == set(entries), (set(entries)-set(cited), set(cited)-set(entries))
    a, b = text.index(r'\bibitem{'), text.index(r'\end{thebibliography}')
    text = text[:a] + '\n'.join(entries[k] for k in cited) + '\n' + text[b:]
    positions = {k: i for i, k in enumerate(cited)}
    text = re.sub(r'\\cite\{([^}]+)\}', lambda m: r'\cite{' + ','.join(sorted(
        (k.strip() for k in m.group(1).split(',')), key=positions.get)) + '}', text)
    (OUT / 'main-integrated.tex').write_text(text, encoding='utf8')
    original = (OUT / 'main.tex').read_text(encoding='utf8')
    patch = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                       fromfile='main.tex', tofile='main-integrated.tex'))
    (OUT / 'integrated-revision.patch').write_text(patch, encoding='utf8')
    receipt = dict(schema='yr317.integrated-review-draft.v1', task='YR-317-i',
        source_sha256=sha(OUT/'main.tex'), prior_draft_sha256=sha(OUT/'main-revised.tex'),
        draft_sha256=sha(OUT/'main-integrated.tex'), references=len(cited),
        prereg_sha256=sha(OUT.parents[3]/'outputs/reports/yr317_v3_independent_eval/prereg.md'),
        addon_prereg_sha256=sha(OUT.parents[3]/'outputs/reports/yr317_v3_block_only/prereg.md'),
        new_experiments=0, submission_ready=False,
        completed_scope=['integrated manuscript prose', 'historical table without daily significance claims',
                         'registered independent protocol', 'explicit outstanding evidence'],
        unresolved=['80-run aggregate results', 'full-candidate ranking validation',
                    'acceptance-network ablation', 'additional robustness analyses',
                    'online latency measurements', 'horizontal/vertical layout evaluation'])
    (OUT/'integrated-revision.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(references=len(cited), submission_ready=False, output='main-integrated.tex')))


def compile_pdf():
    executable = shutil.which('pdflatex')
    if not executable:
        raise RuntimeError('pdflatex is required for --compile')
    target = OUT.parents[3] / 'tmp/pdfs/yr317-integration'
    target.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        run = subprocess.run(
            [executable, '-interaction=nonstopmode', '-halt-on-error',
             '-output-directory=' + str(target), 'main-integrated.tex'],
            cwd=OUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if run.returncode:
            raise RuntimeError(run.stdout[-4000:].decode('utf8', errors='replace'))
    shutil.copyfile(target / 'main-integrated.pdf', OUT / 'main-integrated.pdf')
    print(json.dumps(dict(pdf='main-integrated.pdf', passes=2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true', help='Compile twice with pdflatex')
    args = parser.parse_args()
    build()
    if args.compile:
        compile_pdf()
