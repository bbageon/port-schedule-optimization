"""Apply only the agreed operational-scope and user-context edits to a separate draft."""
import difflib
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent


def main():
    source = (OUT / "main.tex").read_text(encoding="utf-8")
    text, changes = source, []

    def replace(name, old, new):
        nonlocal text
        assert text.count(old) == 1, name
        text = text.replace(old, new)
        changes.append(name)

    replace("abstract_cost_scope", "Arrival-time results assume no appointment-quota\nlimit and therefore bound the potential rather than an executable\nschedule.",
        "Arrival-time results assume unbounded appointment quotas and omit\ncarrier-specific availability, minimum notification requirements, and\nrescheduling and external-waiting costs; they describe modelled terminal\ncost changes rather than validated operational savings.")
    replace("related_work_booking_quotas", "show that arrivals are far from uniform over a day~\\cite{ref13}.",
        "show that arrivals are far from uniform over a day~\\cite{ref13}.\n"
        "Booking quotas also affect truck turn time and crane utilisation, and\n"
        "their selection requires balancing terminal and trucker\n"
        "interests~\\cite{refquota}.")
    replace("method_feasibility_boundary", "Block capacity is enforced in every experiment; when time-slot quota\ndata are unavailable we set $K_k(t)=\\infty$, so the arrival-time results\nrepresent the potential range of adjustment rather than an executable\nschedule.",
        "Block-capacity checks are applied, but $K_k(t)=\\infty$ in all reported\n"
        "experiments because terminal-specific booking quotas are unavailable.\n"
        "The candidate time range is an algorithmic setting, not a verified\n"
        "carrier availability window or notification allowance. Thus these\n"
        "checks establish feasibility only within the simulated rules, not\n"
        "the executability of a changed appointment in practice.")
    anchor = "% =====================================================================\n\\section{Experimental Environment and Evaluation Protocol}"
    context = r"""\subsection{Intended Users and Operational Context}
\label{sec:user-context}
We envisage the architecture as a recommendation component alongside a
terminal operating system (TOS). Busan New Port research identifies a gap
between provided congestion information and actual truck waiting
times~\cite{refbusanwait}; a study of Korean TOS products discusses their
operational support functions~\cite{reftos}. The Busan Port Authority's
2022 description of AllCON-e includes mobile vehicle booking, transfer
transport, and information services~\cite{refallcone}. These sources
motivate an application context; the workflow below is our proposed
design, not an implemented connection to those systems.

Terminal yard planners and operations staff would review the current
and proposed block/time, observed congestion, and route implications,
then accept, revise, or reject a recommendation. Carrier dispatchers
would check arrival-time changes against vehicle availability and later
jobs, and could propose an acceptable window or decline. Drivers would
receive the confirmed time and destination and report execution
difficulties. For example, a proposed later arrival would be reviewed
by the operator, checked with the dispatcher, and communicated to the
driver only after agreement; if agreement were absent, the original
assignment would remain pending review. The human response deadline
would be defined operationally, not equated with the algorithm's
60-second review interval.

The two networks encode computational proposal and acceptance, not
human consent. Pair-centred scores should not be shown as calibrated
monetary savings or confidence probabilities. This study evaluates
simulated operating costs, not an interface or user tasks. Field
integration, usability, user acceptance, and the handling of late or
rejected recommendations remain to be evaluated.

"""
    replace("users_roles_and_workflow", anchor, context + anchor)
    replace("experimental_cost_boundary", "Structural vessel time that the policy cannot change is recorded\nseparately.",
        "Structural vessel time that the policy cannot change is recorded\n"
        "separately. Truck dwell is measured after gate entry; the objective\n"
        "does not include waiting outside the gate, carrier rescheduling\n"
        "charges, or disruption to subsequent trips. These omitted burdens\n"
        "can change the net benefit of a time adjustment.")
    replace("deployment_limitations", "Future work will extend the framework along three axes:\nterminal-specific arrival, demand, and cost data; appointment quotas and\nalternative acceptance structures, including the value of the acceptance\nveto; and robustness across counterfactual horizons, cost structures, and\ncrane-dispatching rules.",
        "Deployment would require a feasibility layer that intersects proposed\n"
        "changes with finite booking availability, carrier-approved time\n"
        "windows, minimum notification requirements, and cargo/vessel\n"
        "deadlines~\\cite{refquota,ref4,ref6}. Terminal and carrier data are needed\n"
        "to set these constraints and the omitted economic costs. We have not\n"
        "tested whether the observed savings persist under those conditions\n"
        "or estimated a break-even rescheduling charge. This is a scope\n"
        "limitation, not evidence of deployment feasibility. Further work\n"
        "also concerns alternative acceptance structures, counterfactual\n"
        "horizons, cost coefficients, dispatching rules, and evaluation of\n"
        "the proposed user workflow in Sect.~\\ref{sec:user-context}.")
    replace("url_support", "\\usepackage{array}", "\\usepackage{array}\n\\usepackage{url}\n\\urlstyle{same}")
    extra = r"""\bibitem{refquota} Huynh, N., Walton, C.M.: Robust scheduling of truck arrivals at marine container terminals. J. Transp. Eng. \textbf{134}(8), 347--353 (2008). doi:10.1061/(ASCE)0733-947X(2008)134:8(347)
\bibitem{refbusanwait} Kim, Y.-I., Shin, J.-Y., Park, H.-J.: A study on the prediction of gate in-out truck waiting time in the container terminal. J. Navig. Port Res. \textbf{46}(4), 344--350 (2022). doi:10.5394/KINPR.2022.46.4.344 (in Korean)
\bibitem{reftos} Won, S.-H., Cho, S.-W., Lee, E.-K.: Analysis technology trend and strategies to vitalize the industry on terminal operating system (TOS) of smart port. J. Shipping Logist. \textbf{41}(1), 25--53 (2025). doi:10.37059/tjosal.2025.41.1.25 (in Korean)
\bibitem{refallcone} Busan Port Authority: Handle Busan Port container cargo easily with AllCON-e. Press release, 16 August 2022 (in Korean; title translated). \url{https://www.busanpa.com/board/view.do?boardId=BBS_0000031&dataSid=27966&menuCd=DOM_000000105002001000}
"""
    replace("four_references", "\\end{thebibliography}", extra + "\\end{thebibliography}")
    replace("bibliography_count", "\\begin{thebibliography}{18}", "\\begin{thebibliography}{22}")
    replace("reference_wrapping", "\\begin{thebibliography}{22}", "\\begin{thebibliography}{22}\n\\raggedright")
    # Keep numbered references in first-citation order after inserting new sources.
    body = text.split("\\begin{thebibliography}", 1)[0]
    keys = list(dict.fromkeys(k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body) for k in group.split(",")))
    entries = dict(re.findall(r"(\\bibitem\{([^}]+)\}[^\n]+)", text))
    entries = {key: line for line, key in entries.items()}
    assert set(keys) == set(entries)
    start, end = text.index("\\bibitem{"), text.index("\\end{thebibliography}")
    text = text[:start] + "\n".join(entries[key] for key in keys) + "\n" + text[end:]
    positions = {key: i for i, key in enumerate(keys)}
    text = re.sub(r"\\cite\{([^}]+)\}", lambda match: "\\cite{" + ",".join(
        sorted((key.strip() for key in match.group(1).split(",")), key=positions.get)) + "}", text)
    text = "% Partial review draft: R1-2 and R3 prose only; remaining review items are unresolved.\n" + text
    (OUT / "main-revised.tex").write_text(text, encoding="utf-8")
    patch = "".join(difflib.unified_diff(source.splitlines(True), text.splitlines(True),
                                      fromfile="main.tex", tofile="main-revised.tex"))
    (OUT / "narrative-revision.patch").write_text(patch, encoding="utf-8")
    receipt = {"schema": "yr317.narrative-revision.v1", "applied": changes,
        "source_sha256": hashlib.sha256((OUT / "main.tex").read_bytes()).hexdigest(),
        "draft_sha256": hashlib.sha256((OUT / "main-revised.tex").read_bytes()).hexdigest(),
        "references": len(keys), "scope": ["R1-2 assumptions and limitations", "R3 users and proposed workflow"],
        "unresolved": ["independent replication", "candidate ranking", "component effects", "sensitivity and online timing", "layout evaluation"],
        "new_experiments": 0, "submission_ready": False}
    (OUT / "narrative-revision.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changes": len(changes), "references": len(keys), "submission_ready": False}))


if __name__ == "__main__":
    main()
