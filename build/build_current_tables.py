"""Build only the five active tables; preserve frozen estimates and intervals."""
from pathlib import Path
import csv
import hashlib
import json

R = Path(__file__).resolve().parents[1]
OUT = R / 'visuals/tables'
OUT.mkdir(parents=True, exist_ok=True)
inputs = {}


def read(name):
    p = R / name
    inputs[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if p.suffix == '.csv':
        with p.open(encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    return json.loads(p.read_text())


def one(rows, **criteria):
    found = [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]
    assert len(found) == 1, criteria
    return found[0]


def interval(value, low, high, digits=2, signed=False):
    assert low <= value <= high
    estimate = f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'
    return f'${estimate}$ $[{low:.{digits}f},\\,{high:.{digits}f}]$'


def export(name, title, label, body, note, wide=False):
    width = r'\fulllength' if wide else r'\textwidth'
    start = r' \begin{adjustwidth}{-\extralength}{0cm}' + '\n' if wide else ''
    end = r' \end{adjustwidth}' + '\n' if wide else ''
    text = ('\\begin{table}[!htbp]\n' + start + ' \\caption{' + title + '}\n'
            ' \\label{' + label + '}\n \\centering\n \\tablebodyfont\n'
            ' \\renewcommand{\\arraystretch}{1.16}\n \\setlength{\\tabcolsep}{5pt}\n'
            + body + '\n \\par\\vspace{3pt}\\noindent\n \\parbox{' + width
            + '}{\\tablenotefont ' + note + '}\n' + end + '\\end{table}\n')
    (OUT / f'{name}.tex').write_text(text)


runtime = read('evidence/runtime_complexity_summary.json')
batch = runtime['aggregates']['1']
reference = batch['v19_actor']['wall_mean_ms']['mean']
body = r'''\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrr@{}}
\toprule
Inference path & Latency, mean (SD), ms & Relative latency \\
\midrule
'''
paths = [
    ('Single actor pass', [('navrl_actor', 'Nominal NavRL'), ('v19_actor', 'RC-CF-NavRL')]),
    ('Actor pass plus online search or filtering', [
        ('v19_pcarp', 'RC-CF-NavRL + online PC-ARP'), ('v19_official_vo', 'RC-CF-NavRL + VO'),
    ]),
]
for index, (heading, items) in enumerate(paths):
    if index:
        body += r'\addlinespace[3pt]' + '\n'
    body += r'\multicolumn{3}{@{}l}{\textit{' + heading + r'}} \\' + '\n'
    for key, name in items:
        r = batch[key]['wall_mean_ms']
        assert r['repeat_count'] == 3
        body += (rf'\quad {name} & ${r["mean"]:.3f}\;( {r["sample_std"]:.3f} )$'
                 rf' & ${r["mean"] / reference:.2f}\times$ \\' + '\n')
body += r'\bottomrule' + '\n' + r'\end{tabular*}'
export('03_runtime', 'Batch-size-one policy latency with and without online correction on a desktop NVIDIA RTX 5090.',
       'tab:runtime', body,
       'Values are mean (sample SD) of three repeat means and include the actor pass for each correction path. Relative latency uses direct RC-CF-NavRL as the reference. Timing excludes sensing, communication, and low-level control (Section~\\ref{sec:timing-protocol}).')

vo = read('evidence/deployment_shield_paired_effects.csv')
body = r'''\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrr@{}}
\toprule
Moving obstacles & Success difference (pp) & Collision difference (pp) \\
\midrule
'''
for key, heading in [
        ('mean_risk_official_vo_vs_actor', r'Adding protection: (RC-CF + VO) $-$ RC-CF'),
        ('mean_risk_official_vo_vs_ppolag_official_vo',
         r'Shared protection: (RC-CF + VO) $-$ (PPO-Lagrangian + VO)')]:
    if key.endswith('ppolag_official_vo'):
        body += r'\addlinespace[5pt]' + '\n'
    body += r'\multicolumn{3}{@{}l}{\textit{' + heading + r'}} \\' + '\n'
    for density in [60, 120]:
        cells = []
        for metric in ['success', 'collision']:
            r = one(vo, condition=f'd{density}', comparison=key, metric=metric)
            cells.append(interval(*[100 * float(r[k]) for k in
                                    ['difference', 'ci95_low', 'ci95_high']], signed=True))
        body += f'{density} & ' + ' & '.join(cells) + r' \\' + '\n'
body += r'\bottomrule' + '\n' + r'\end{tabular*}'
export('06_deployment', 'Paired full-flight differences with runtime velocity-obstacle (VO) protection.',
       'tab:deployment', body,
       'Paired differences [pointwise 95\\% bootstrap confidence intervals], in percentage points (pp); 1920 matched episode pairs per row in the original cohort. RC-CF, RC-CF-NavRL.')

body = r'''\begin{tabularx}{\fulllength}{@{}lXX@{}}
\toprule
Setting & RC-CF-NavRL & PPO-Lagrangian \\
\midrule
Nominal branch & Frozen task encoder and Beta head & Actor and encoder updated jointly \\
Safety representation & NavRL-initialized $256$--$256$ MLP & No separate safety encoder \\
Cost critic & $256$--$256$ MLP; SiLU, LayerNorm; 32 ordered quantiles & Scalar critic \\
Policy learning rate & $10^{-4}$ (gate and residual) & $5\times10^{-4}$ \\
Safety/cost learning rate & $3\times10^{-4}$ & $3\times10^{-4}$ \\
Adaptation schedule & 16 safety-only + 40 joint updates & 56 joint updates \\
Correction parameters & $\delta_{\max}=0.06$; $T_g=1.75$ & Not applicable \\
\bottomrule
\end{tabularx}'''
export('04_configuration', 'Implementation settings for the two adaptation procedures.',
       'tab:configuration', body,
       'Shared settings: Adam; 56 updates (114,688 frames); PPO clipping 0.1, four epochs, 16 minibatches; mean-risk cost objective, $B=0.05$, discount 0.99, and target update coefficient 0.01. MLP: multilayer perceptron; SiLU: sigmoid linear unit.', wide=True)

orig = read('visuals/data/mean_risk_core_ablation_statistics.json')
released = read('evidence/scope17_paired_effects_disclosed.json')['original_analyzer_result']['contrasts']
body = r'''\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrr@{}}
\toprule
Reference variant & Success difference (pp) & Collision difference (pp) \\
\midrule
\multicolumn{3}{@{}l}{\textit{Original cohort: 5 training seeds; 5760 matched pairs}} \\
'''
for key, name in [('no_gate', 'Without gate'), ('no_counterfactual', r'Without critic loss $\mathcal L_{\mathrm{cf}}$')]:
    cells = []
    for metric in ['success', 'collision']:
        r = orig['comparisons'][key]['metrics'][metric]
        cells.append(interval(100 * r['paired_difference_full_minus_ablation'],
                              *[100 * v for v in r['hierarchical_bootstrap_ci95']], signed=True))
    body += name + ' & ' + ' & '.join(cells) + r' \\' + '\n'
body += r'''\addlinespace[5pt]
\multicolumn{3}{@{}l}{\textit{Released-checkpoint cohort: 10 training seeds; 10,880 matched pairs}} \\
'''
for key, name in [('gate_effect_Lcf_on', 'Without gate'),
                  ('Lcf_effect_gate_on', r'Without critic loss $\mathcal L_{\mathrm{cf}}$')]:
    cells = []
    for metric in ['success', 'collision']:
        r = one(released, contrast=key, metric=metric)
        cells.append(interval(*[100 * float(r[k]) for k in
                                ['difference', 'ci95_low', 'ci95_high']], signed=True))
    body += name + ' & ' + ' & '.join(cells) + r' \\' + '\n'
body += r'\bottomrule' + '\n' + r'\end{tabular*}'
export('05_components', 'Paired full-flight navigation differences between the full RC-CF-NavRL model and component-removal variants.',
       'tab:components', body,
       'Full method minus the indicated variant, in percentage points (pp), with cohort-specific paired-bootstrap 95\\% confidence intervals.')

teacher = read('evidence/teacher_all_reported_metrics.csv')
body = r'''\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrr@{}}
\toprule
Diagnostic & Released-policy states & Adapted-policy states \\
\midrule
'''
for name, key, scale, digits in [
        ('Collision-score AUROC', 'collision_score_auroc', 1, 3),
        ('Clearance-ordering accuracy', 'within_state_resolvable_clearance_ranking_accuracy', 1, 3),
        (r'Ordering coverage (\%)', 'ranking_eligible_anchor_fraction', 100, 2)]:
    cells = []
    for family in ['baseline', 'F11']:
        r = one(teacher, condition='canonical_D80', family=family,
                duration='nearest_50_steps', metric=key)
        cells.append(interval(*[scale * float(r[k]) for k in
                                ['estimate', 'ci95_low', 'ci95_high']], digits=digits))
    if key == 'ranking_eligible_anchor_fraction':
        body += r'\addlinespace[3pt]' + '\n'
    body += name + ' & ' + ' & '.join(cells) + r' \\' + '\n'
body += r'\bottomrule' + '\n' + r'\end{tabular*}'
export('01_teacher', 'Geometric teacher discrimination and candidate-ordering diagnostics on local replay anchor states.',
       'tab:teacher', body,
       'Estimates [95\\% descriptive cluster-bootstrap confidence intervals] from 692 released-policy and 2784 adapted-policy anchors across four shared evaluation seeds per source checkpoint. Clearance ordering uses anchors with resolvable candidate differences; coverage is their fraction among qualified anchors.')

(R / 'build/current-table-inputs.json').write_text(json.dumps({
    'inputs_sha256': inputs,
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'active_tables': ['03_runtime', '04_configuration', '05_components', '01_teacher', '06_deployment'],
    'operations': 'Reorganize and format existing settings and estimates; no new statistical inference.'
}, indent=2) + '\n')
print('Generated five active tables from frozen evidence and documented settings.')
