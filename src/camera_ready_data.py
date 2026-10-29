from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import tarfile

from shortjob_runner.supplement_protocol import task_id

PACKAGES = {'4060': 'official_4060_20260625', 'v100': 'official_v100_20260625',
            'a100': 'official_a100_20260626', 'h100': 'official_h100_20260627'}
LABELS = {'4060': 'RTX 4060 Laptop', 'v100': 'V100', 'a100': 'A100', 'h100': 'H100'}
LEGACY_SHA256 = '2d50112588a16ed992e0b9ce6797544f9a86cf7b338d3508518c7d6437cb5e63'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contained(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f'数据路径越界: {relative}')
    return path


def stage_records(root, info):
    folder = contained(root, info['path'])
    manifest = read_json(folder / 'manifest.json')
    expected = {task_id(task): task for task in manifest['tasks']}
    require(len(expected) == len(manifest['tasks']), f'重复计划任务: {info["path"]}')
    records = [read_json(p) for p in sorted((folder / 'rows').glob('*.json'))]
    index = {row['task_id']: row for row in records}
    require(len(index) == len(records) and index.keys() == expected.keys(), f'任务集合不一致: {info["path"]}')
    for ident, row in index.items():
        task = row['task']
        require(task == expected[ident], f'任务内容不匹配: {ident}')
        if task['phase'] == 'performance':
            require(row['status'] == 'measured', f'正式计时未完成: {ident}')
            require(row['effective_steps'] == task['steps'] and row['final_state_finite'], f'执行步数或状态无效: {ident}')
            require(math.isfinite(row['total_s']) and row['total_s'] > 0, f'无效计时: {ident}')
            gates = [index[g] for g in row['gate_task_ids']]
            require(len(gates) == 2 and {g['task']['seed'] for g in gates} == {17, 42}, f'正确性 seed 不完整: {ident}')
            fields = ['workload', 'action', 'batch', 'steps', 'protocol', 'precision_policy']
            require(all(g['status'] == 'passed' and all(g['task'][k] == task[k] for k in fields) for g in gates),
                    f'正确性关联无效: {ident}')
            if task['precision_policy'] == 'strict_fp32':
                require(row['runtime']['cuda_matmul_allow_tf32'] is False and
                        row['runtime']['cudnn_allow_tf32'] is False, f'精度配置不一致: {ident}')
    counts = dict(Counter(row['status'] for row in records))
    require(len(records) == info['records'] and counts == info['statuses'], f'记录统计不一致: {info["path"]}')
    return records


def verify_grid(records, expected_rows, gate=False):
    fields = ['workload_id', 'action', 'batch_size', 'num_steps'] + ([] if gate or expected_rows == 48 else ['repeat'])
    keys = [tuple(row[k] for k in fields) for row in records]
    require(len(keys) == len(set(keys)) == expected_rows, '历史记录数量或唯一性不符合协议')
    axes = [sorted({row[k] for row in records}) for k in fields]
    # warm priming 的 repeat 编号来自随机执行顺序，单独核验条件键。
    require(set(keys) == set(itertools.product(*axes)), '实验矩阵缺失组合')
    for row in records:
        if row['status'] == 'completed' and 'total_runtime_s' in row:
            require(row['action_feasible'] and math.isfinite(row['total_runtime_s']) and
                    row['total_runtime_s'] > 0, '完成记录包含无效运行时间')
    return {'records': len(records), 'statuses': dict(Counter(r['status'] for r in records))}


def verify_measurement_source():
    root = Path(__file__).resolve().parents[1] / 'measurement_source'
    manifest = read_json(root / 'MANIFEST.json')
    require(manifest['schema'] == 'shortjob-measurement-source-v1', '未知测量源码清单')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'MANIFEST.json'}, '测量源码文件集合不一致')
    for relative, digest in manifest['files'].items():
        require(sha(contained(root, relative)) == digest, f'测量源码发生变化: {relative}')
    measured = hashlib.sha256()
    modules = sorted((root / '03_src/shortjob_runner').glob('*.py'))
    require(bool(modules), '仓库缺少冻结测量源码')
    for path in modules:
        measured.update(path.name.encode())
        measured.update(path.read_bytes())
    require(measured.hexdigest() == manifest['measurement_source_sha256'], '测量源码身份不一致')
    return measured.hexdigest()


def verify_dataset(data_root, legacy_archive=None):
    root = Path(data_root).resolve()
    manifest = read_json(root / 'manifest.json')
    require(manifest['schema'] == 'shortjob-camera-ready-dataset-v3', '未知数据格式')
    require(manifest['legacy']['archive_sha256'] == LEGACY_SHA256, '旧版归档身份不匹配')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    expected = set(manifest['files']) | {'manifest.json', 'CHECKSUMS.sha256'}
    require(actual == expected, '数据集存在缺失或未索引文件')
    for relative, info in manifest['files'].items():
        require(sha(contained(root, relative)) == info['sha256'], f'文件校验失败: {relative}')
    legacy = manifest['legacy']['files']
    for relative, digest in legacy.items():
        require(sha(contained(root, relative)) == digest, f'旧版文件发生变化: {relative}')
    measured_source = verify_measurement_source()
    require(measured_source == manifest['measurement_source_sha256'], '数据与仓库测量源码身份不一致')
    assets_hash = sha(root / 'assets/manifest.json')
    if legacy_archive:
        require(sha(legacy_archive) == LEGACY_SHA256, '传入旧版归档的哈希不匹配')
        with tarfile.open(legacy_archive) as archive:
            original = {}
            for member in archive.getmembers():
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        original[member.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(original == legacy, '旧版归档与新版保留的路径或字节不一致')
    base = root / manifest['legacy']['root'] / 'raw'
    historical = {}
    specs = {'cold_core': 8320, 'correctness_gate': 104, 'warm_reuse_core': 360,
             'warm_reuse_core_prime': 48, 'graph_reuse_core': 288}
    for device, package in PACKAGES.items():
        historical[device] = {suite: verify_grid(read_rows(base / package / f'{suite}.jsonl'), n,
                                                gate=suite == 'correctness_gate')
                              for suite, n in specs.items()}
    sessions = {}
    for name, info in manifest['sessions'].items():
        sessions[name] = {}
        for stage, description in info['stages'].items():
            stage_manifest = read_json(root / description['path'] / 'manifest.json')
            require(stage_manifest['source_hash'] == measured_source, '测量源码身份不一致')
            if stage_manifest['assets_manifest_hash']:
                require(stage_manifest['assets_manifest_hash'] == assets_hash, '模型资产身份不一致')
            records = stage_records(root, description)
            sessions[name][stage] = {'records': len(records), 'statuses': dict(Counter(r['status'] for r in records))}
        require(sessions[name]['real']['statuses'] == {'passed': 72, 'measured': 180}, f'真实模型范围不完整: {name}')
    require(len(sessions) == 9, '需要九个真实模型会话')
    replication = {}
    for suite, n in [('legacy_gate', 104), ('cold', 8320), ('warm', 360), ('warm_prime', 48), ('graph_reuse', 288)]:
        replication[suite] = verify_grid(read_rows(root / manifest['replication'][suite]), n, gate=suite == 'legacy_gate')
    count = sum(r['records'] for r in replication.values()) + sum(r['records'] for r in sessions['v100_c'].values())
    require(count == 9583, '完整复现实验数量不匹配')
    return {'status': 'verified', 'files': len(actual), 'legacy_files': len(legacy),
            'historical_files_unchanged': True, 'legacy_archive_checked': bool(legacy_archive),
            'measurement_source_sha256': measured_source, 'assets_manifest_sha256': assets_hash,
            'historical': historical, 'sessions': sessions, 'real_correctness_rows': 648,
            'real_performance_rows': 1620, 'replication': replication, 'replication_records': count}
