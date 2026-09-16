"""Read-only curve arithmetic, not a terminal performance experiment."""
import ast
import hashlib
import json
import math
from pathlib import Path


def inspect():
    source = Path('src/yard_rl/v3/world/integrated/terminal_stream.py')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'DIURNAL_PEAKS', 'DIURNAL_NIGHT_FRAC', 'DIURNAL_DAY_TOTAL', 'DIURNAL_DAY_S'}
    body = [n for n in tree.body if
        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name == 'diurnal_rate')]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), ns)
    rows = []
    for scale in (1.5, 1.2, 1.1, 1.05, 1.0, 0.6):
        peaks = [(mu, sigma*scale, w) for mu, sigma, w in ns['DIURNAL_PEAKS']]
        rates = [ns['diurnal_rate'](i*15, total=7500, peaks=peaks)*3600 for i in range(5761)]
        maxima = [i for i in range(1, len(rates)-1)
                  if rates[i] > rates[i-1] and rates[i] > rates[i+1]]
        base = ns['DIURNAL_NIGHT_FRAC']
        wsum = sum(w for _, _, w in peaks)
        masses = [(1-base)*w/wsum*.5*(math.erf((24-mu)/(sigma*math.sqrt(2)))
                  - math.erf(-mu/(sigma*math.sqrt(2)))) for mu,sigma,w in peaks]
        normalizer = base+sum(masses)
        rows.append(dict(scale=scale, component_peaks=peaks, visible_peak_count=len(maxima),
            visible_peak_hours=[round(i*15/3600,4) for i in maxima],
            peak_rate_trucks_per_h=[rates[i]/normalizer for i in maxima],
            effective_uniform_share=base/normalizer,
            effective_component_shares=[x/normalizer for x in masses]))
    return dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                daily_trucks=7500, calculation_grid_s=15, method='local maxima on density grid; analytic 24h mass normalization',
                simulation_runs=0, rows=rows)


if __name__ == '__main__':
    report = inspect()
    path = Path('outputs/reports/yr317_v3_peak_width/curve-inspection.json')
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
