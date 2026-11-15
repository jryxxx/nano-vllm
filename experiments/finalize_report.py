"""Validate saved results and archive source/environment provenance."""
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path('/root/autodl-tmp/shared-context')
DEV = ROOT / 'src/nano-vllm-shared'
RESULTS = ROOT / 'results/integration'


def read(name):
    return json.loads((RESULTS / name).read_text())


def main():
    kernel = read('paged-correctness.json')
    assert len(kernel['cases']) == 37 and len(kernel['graph_replay']) == 8
    assert kernel['planner'] == 'passed'
    model = read('model-fp16-auto-graph-final.json')
    assert any(s['shared'] for s in model['decode_steps'])
    assert any(not s['shared'] for s in model['decode_steps'])
    assert max(s['relative_l2'] for s in model['decode_steps']) < .005
    assert max(s['max_kl'] for s in model['decode_steps']) < .001
    for file in ('model-fp16-auto-eager-fallback.json', 'model-bf16-auto-fallback.json'):
        rows = read(file)['decode_steps']
        assert len(rows) == 9
        assert all(not s['shared'] and s['max_abs_logit_error'] == 0 for s in rows)

    modes = {m: read(f'serving-{m}-graph.json')['trials']
             for m in ('original', 'off', 'auto', 'force')}
    names = ['shared_short', 'shared_long', 'small_shared', 'no_shared',
             'two_groups', 'shrinking', 'open_loop']
    table = []
    for name in names:
        row = {'scenario': name}
        for mode, trials in modes.items():
            values = [t for t in trials if t['scenario'] == name]
            assert len(values) == 3 and all(t['preemptions'] == 0 for t in values)
            row[mode] = statistics.median(t['output_tokens_per_s'] for t in values)
        row['auto_gain_percent'] = (row['auto'] / row['original'] - 1) * 100
        table.append(row)
    eager = {m: read(f'serving-{m}-eager.json')['trials'] for m in ('off', 'auto')}
    assert len(eager['auto']) == 9
    assert all(set(t['route_counts']) == {'eager_fallback'} for t in eager['auto'])
    eager_table = []
    for name in ('shared_long', 'small_shared', 'no_shared'):
        row = {'scenario': name}
        for mode, trials in eager.items():
            row[mode] = statistics.median(t['output_tokens_per_s'] for t in trials if t['scenario'] == name)
        row['difference_percent'] = (row['auto']/row['off']-1)*100
        eager_table.append(row)

    labels = ['32 req / 1K', '32 req / 4K', '8 req / 4K', 'No sharing',
              'Two groups', 'Shrinking batch', 'Open loop']
    fig, ax = plt.subplots(figsize=(10, 4.7))
    gains = [r['auto_gain_percent'] for r in table]
    colors = ['#2485bd' if i in (0, 1, 5, 6) else '#a0a7ad' for i in range(7)]
    bars = ax.bar(labels, gains, color=colors)
    ax.axhline(0, color='#555555', linewidth=.8)
    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x()+bar.get_width()/2, max(gain, 0)+.6,
                f'{gain:+.1f}%', ha='center', fontsize=10)
    ax.set_ylabel('Throughput change vs original (%)')
    ax.set_ylim(-3, 34)
    ax.set_title('End-to-end CUDA Graph workloads | RTX 4090 | Qwen3-0.6B FP16')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, 'Three repeats per workload. Gray: automatic fallback; small changes are measurement variation.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(RESULTS/'end-to-end-gain.png', dpi=170)
    plt.close(fig)

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=DEV)
    paths = sorted(set(git('ls-files', '-c', '-o', '--exclude-standard').decode().splitlines()))
    manifest = {p: hashlib.sha256((DEV/p).read_bytes()).hexdigest()
                for p in paths if (DEV/p).is_file()}
    patch = git('diff', '--binary')
    for p in git('ls-files', '-o', '--exclude-standard').decode().splitlines():
        if (DEV/p).is_file():
            proc = subprocess.run(['git', 'diff', '--no-index', '--binary', '--', '/dev/null', p],
                                  cwd=DEV, capture_output=True)
            assert proc.returncode in (0, 1), proc.stderr
            patch += proc.stdout
    (RESULTS/'implementation.patch').write_bytes(patch)
    (RESULTS/'source-manifest.json').write_text(json.dumps({
        'base_commit': git('rev-parse', 'HEAD').decode().strip(),
        'files_sha256': manifest,
        'patch_sha256': hashlib.sha256(patch).hexdigest()}, indent=2))
    original_status = subprocess.check_output(['git', '-C', str(ROOT/'src/nano-vllm'), 'status', '--porcelain'])
    assert not original_status, original_status
    summary = {'kernel_cases': 37, 'dynamic_graph_replays': 8,
               'model_relative_l2_max': max(s['relative_l2'] for s in model['decode_steps']),
               'model_kl_max': max(s['max_kl'] for s in model['decode_steps']),
               'eager_fallback_exact': True, 'bf16_fallback_exact': True,
               'original_source_clean': True, 'graph_results': table,
               'eager_fallback_results': eager_table}
    (RESULTS/'final-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
