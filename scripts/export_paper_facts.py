"""Export a dated, read-only audit of experiment facts and existing results."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
import numpy as np
from scripts.run_protocol import _experiment_and_data
from scripts.prepare_tep_extended import FILES, sha256
from data.tep_dataset import build_tep_datasets_from_config
from data.feature_dataset import deterministic_subset
from data.tep_hetero_graph import _process_edges


def main() -> None:
    torch.set_num_threads(1)
    now = datetime.now().astimezone()
    out = ROOT / 'docs' / ('paper_facts_' + now.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    stage5, experiment, data = _experiment_and_data(ROOT / 'configs/stage5.yaml', 'protocol_2')
    seeds = list(data['experiment']['seeds'])
    manifest = {'audited_at': now.isoformat(), 'historical_launch_snapshot_available': False, 'files': {}}
    manifest['data_files'] = []
    for name, (_, _, expected_hash) in FILES.items():
        source = ROOT / 'datasets/tep/extended500/parquet' / name
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(f'Source data hash mismatch: {source}')
        manifest['data_files'].append({'path': str(source), 'sha256': actual_hash,
                                       'matches_prepare_script': True})
    for name in ('data.yaml', 'stage2.yaml', 'stage5.yaml'):
        source = ROOT / 'configs' / name
        content = source.read_bytes()
        (out / name).write_bytes(content)
        manifest['files'][str(source)] = hashlib.sha256(content).hexdigest()
    for source in [ROOT / '论文框架.md', ROOT / 'scripts/run_protocol.py', ROOT / 'models/qc_hgt.py',
                   Path('E:/Desktop/会议论文文件/III. Experiments.md')]:
        manifest['files'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    for protocol in stage5['protocols']:
        _, effective, effective_data = _experiment_and_data(ROOT / 'configs/stage5.yaml', protocol)
        (out / (protocol + '_effective.yaml')).write_text(
            yaml.safe_dump({'experiment': effective, 'data': effective_data}, allow_unicode=True, sort_keys=False), encoding='utf-8')

    datasets = build_tep_datasets_from_config(data, seed=seeds[0], print_leakage_report=True)
    split_manifest = {}
    sampling = experiment['sampling']
    for split, dataset in datasets.items():
        subset = deterministic_subset(dataset, sampling['max_samples_per_split'][split],
            sampling['base_seed'] + sampling['seed_offsets'][split], sampling['strategy'], sampling['windows_per_run'])
        sampled = []
        used = set()
        for index in subset.indices:
            ri = bisect_right(dataset._cumulative, index)
            previous = dataset._cumulative[ri-1] if ri else 0
            start = (index - previous) * dataset.stride
            run = dataset.runs[ri]
            used.add(run.run_uid)
            sampled.append({'run_uid': run.run_uid, 'start_zero_based': start, 'end_zero_based': start+dataset.window_length-1})
        split_manifest[split] = {
            'total_runs': len(dataset.runs), 'total_windows': len(dataset), 'sampled_windows': len(subset),
            'sampled_runs': len(used), 'run_lengths': dict(Counter(r.length for r in dataset.runs)),
            'runs_per_fault': dict(Counter(r.fault_id for r in dataset.runs)),
            'runs': [{'run_uid': r.run_uid, 'origin': r.origin, 'length': r.length} for r in dataset.runs],
            'selected_windows': sampled,
        }
    (out / 'run_and_window_manifest.json').write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest['split_summary'] = {k:{key:value for key,value in v.items() if key not in ('runs','selected_windows')} for k,v in split_manifest.items()}
    manifest['edge_counts'] = {'|'.join(key): value.shape[1] for key,value in _process_edges().items()}
    manifest['results'] = []
    docs = {}
    full_export = {}
    for protocol, protocol_cfg in stage5['protocols'].items():
        for model in protocol_cfg['main_models']:
            for seed in seeds:
                path = ROOT / 'results/stage5' / protocol / 'main' / f'{model}_{seed}.json'
                record = {'protocol':protocol, 'model':model, 'seed':seed, 'path':str(path), 'complete':False}
                checkpoint = ROOT / 'results/stage5/checkpoints' / protocol / 'main' / f'{model}_{seed}.pt'
                record['checkpoint_exists'] = checkpoint.exists()
                if path.exists():
                    payload_bytes = path.read_bytes()
                    doc = json.loads(payload_bytes)
                    record.update(complete=bool(doc.get('complete')), scenarios=len(doc.get('evaluations',{})),
                                  best_epoch=doc.get('best_epoch'), sha256=hashlib.sha256(payload_bytes).hexdigest())
                    full_export[f'{protocol}/{model}/{seed}'] = doc
                    if record['complete']:
                        docs[protocol,model,seed] = doc
                manifest['results'].append(record)
    (out / 'result_snapshot.json').write_text(json.dumps(full_export,ensure_ascii=False,indent=2), encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2), encoding='utf-8')

    lines = ['# 已核验数值附录', '', '核验时点：'+now.isoformat(), '',
             '所有 Macro-F1 使用 0–1；± 为五个 seed 的样本标准差（ddof=1）。',
             '缺失结果保持 <<待补充>>；不对不足五个 seed 的模型生成正式均值±标准差。',
             '配置副本和有效配置均为本次审计快照，不是历史启动快照。完整精度及原始指标见 result_snapshot.json。', '']
    def table(headers, rows):
        lines.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        for row in rows:
            lines.append('| '+' | '.join(str(x) for x in row)+' |')
        lines.append('')
    def cell(values):
        return f'{statistics.fmean(values):.6f} ± {statistics.stdev(values):.6f}' if len(values)==len(seeds) else '<<待补充>>'
    def get(protocol,model,seed,scenario,key='macro_f1'):
        doc=docs.get((protocol,model,seed))
        return None if doc is None else doc['evaluations'].get(scenario,{}).get(key)
    def fmt(value):
        return '<<待补充>>' if value is None else f'{value:.6f}'
    p1models = stage5['protocols']['protocol_1']['main_models']
    grid = data['evaluation']['fixed_strength_grid']['levels']
    lines.extend(['## Table I：原始 seed 结果','', '各四元组均按表头列出的强度顺序。Missing 的两个子类型分别保留。Avg. Robustness 对配置中的五操作各四强度等权，先 seed 内汇总再跨 seed 统计。',''])
    table(['模型','seed','Clean'] + [name+' '+str(levels) for name,levels in grid.items()],
          [[model,seed,fmt(get('protocol_1',model,seed,'clean'))] +
           [', '.join(fmt(get('protocol_1',model,seed,f'{name}={level}')) for level in levels) for name,levels in grid.items()]
           for model in p1models for seed in seeds])
    lines.extend(['## Table I：当前代码口径的五 seed 汇总','', 'Missing = 随机缺失四档和连续缺失四档的八项等权均值；其余退化各四档等权均值。',''])
    mainrows=[]
    for model in p1models:
        row=[model,cell([get('protocol_1',model,s,'clean') for s in seeds if get('protocol_1',model,s,'clean') is not None])]
        for names in (['gaussian_noise'],['random_missing','continuous_missing'],['drift'],['time_shift']):
            values=[]
            for seed in seeds:
                scores=[get('protocol_1',model,seed,f'{name}={level}') for name in names for level in grid[name]]
                if all(v is not None for v in scores): values.append(statistics.fmean(scores))
            row.append(cell(values))
        robust_values = []
        for seed in seeds:
            scores = [get('protocol_1',model,seed,f'{name}={level}')
                      for name, levels in grid.items() for level in levels]
            if all(value is not None for value in scores):
                robust_values.append(statistics.fmean(scores))
        row.append(cell(robust_values))
        mainrows.append(row)
    table(['Model','Clean','Noise','Missing','Drift','Time Shift','Avg. Robustness'],mainrows)
    scenarios=['clean','protocol_group=known','protocol_group=unseen_combination']
    lines.extend(['## Table II：逐 seed 数据',''])
    table(['模型','seed','Clean','Seen Degradation','Unseen Combination'],
          [[model,seed]+[fmt(get('protocol_2',model,seed,scenario)) for scenario in scenarios]
           for model in stage5['protocols']['protocol_2']['main_models'] for seed in seeds])
    lines.extend(['## Table II：仅完整五 seed 汇总',''])
    table(['模型','Clean','Seen Degradation','Unseen Combination'],
          [[model]+[cell([get('protocol_2',model,s,scenario) for s in seeds if get('protocol_2',model,s,scenario) is not None]) for scenario in scenarios]
           for model in stage5['protocols']['protocol_2']['main_models']])
    lines.extend(['## Figure 2(a)：随机缺失曲线数据','', '逐 seed 数值见 Table I 的 random_missing 列；本表按各档独立汇总五 seed。',''])
    table(['模型']+[str(v) for v in grid['random_missing']],
          [[model]+[cell([get('protocol_1',model,s,f'random_missing={level}') for s in seeds]) for level in grid['random_missing']] for model in p1models])
    lines.extend(['## Figure 2(b)：现有 P1 QC-HGT 可靠性原始数据','',
                  '数值为每个 seed 的全部 2000 个测试窗口 × 41 个 sensor 的预测可靠性均值；包含未被退化的 sensor。',''])
    table(['退化','强度']+[f'seed={s}' for s in seeds]+['五 seed 均值±标准差'],
          [[name,level]+[fmt(get('protocol_1','qc_hgt',s,f'{name}={level}','mean_sensor_reliability')) for s in seeds]+
           [cell([get('protocol_1','qc_hgt',s,f'{name}={level}','mean_sensor_reliability') for s in seeds])]
           for name,levels in grid.items() for level in levels])
    lines.extend(['## Figure 2(b)：P2 QC-HGT 可靠性原始数据','',
                  '统计口径同 P1；同时保留两协议供写作选择，不按测试表现选择 seed。',''])
    table(['退化','强度']+[f'seed={s}' for s in seeds]+['五 seed 均值±标准差'],
          [[name,level]+[fmt(get('protocol_2','qc_hgt',s,f'{name}={level}','mean_sensor_reliability')) for s in seeds]+
           [cell([get('protocol_2','qc_hgt',s,f'{name}={level}','mean_sensor_reliability') for s in seeds])]
           for name,levels in grid.items() for level in levels])
    lines.extend(['## Figure 3：原始预测与混淆矩阵核验','',
                  '每个 seed 单独保存 Clean/Unseen；按真实类别行归一化，不跨 seed 池化。详细计数见 figure3_audit.json。',''])
    figure3 = []
    for seed in seeds:
        for scenario in ('clean','protocol_group=unseen_combination'):
            document = docs.get(('protocol_2','qc_hgt',seed))
            if document is None:
                continue
            metrics = document['evaluations'][scenario]
            safe = ''.join(c if c.isalnum() or c in '_.-' else '_' for c in scenario)
            path = ROOT / 'results/stage5/artifacts/protocol_2/main' / f'qc_hgt_{seed}_{safe}.npz'
            with np.load(path, allow_pickle=False) as arrays:
                labels = arrays['label'].astype(int)
                predictions = arrays['prediction'].astype(int)
                matrix = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
                rebuilt = np.zeros_like(matrix)
                np.add.at(rebuilt, (labels, predictions), 1)
                if not np.array_equal(rebuilt, matrix):
                    raise ValueError(f'Artifact/JSON confusion mismatch: {path}')
                if len(labels) != metrics['num_examples']:
                    raise ValueError(f'Artifact is not the full test set: {path}')
                if not np.array_equal(arrays['probabilities'].argmax(axis=1), predictions):
                    raise ValueError(f'Probability/label mismatch: {path}')
            support = matrix.sum(axis=1)
            normalized = np.divide(matrix, support[:,None], out=np.zeros_like(matrix,dtype=float),
                                   where=support[:,None]>0)
            figure3.append({'seed':seed,'scenario':scenario,'artifact':str(path),
                            'artifact_sha256':sha256(path),'samples':len(labels),
                            'confusion_matrix':matrix.tolist(),'row_normalized':normalized.tolist(),
                            'per_class':metrics['per_class']})
    (out/'figure3_audit.json').write_text(json.dumps(figure3,ensure_ascii=False,indent=2),encoding='utf-8')
    table(['seed','场景','样本数','NPZ 与 JSON 一致'],
          [[r['seed'],r['scenario'],r['samples'],'是'] for r in figure3])
    # Render an audit example using the existing, fixed seed=0 convention.
    if ('protocol_2','qc_hgt',seeds[0]) in docs:
        from eval.visualize import _confusion_figure
        _confusion_figure(docs['protocol_2','qc_hgt',seeds[0]],out,stage5['visualization'])
    (out / 'tables.md').write_text('\n'.join(lines),encoding='utf-8')
    print('AUDIT_DIRECTORY='+str(out))
    print(json.dumps({'split_summary':manifest['split_summary'],'edge_counts':manifest['edge_counts'],
                      'complete':sum(r['complete'] for r in manifest['results'])},ensure_ascii=False))


if __name__ == '__main__':
    main()

# Design note (paper Sections 4.1--4.4, 5.1--5.4): reconstruct run-first indices,
# preserve current config/result provenance, and aggregate within seed before
# five-seed statistics; never infer missing results or an unconfirmed robustness definition.
