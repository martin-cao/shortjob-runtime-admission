import argparse
import json
from pathlib import Path

from camera_ready_data import contained, read_json, require, sha, verify_dataset, write_json


def verify_release(root):
    root = Path(root).resolve()
    manifest = read_json(root / 'RELEASE_MANIFEST.json')
    require(manifest['schema'] == 'shortjob-data-release-v1', '未知 Release 数据格式')
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(files == set(manifest['files']) | {'RELEASE_MANIFEST.json', 'CHECKSUMS.sha256'},
            'Release 存在缺失或未索引文件')
    code_suffixes = {'.py', '.sh', '.bash', '.fish', '.toml', '.lock', '.tar', '.gz', '.zip'}
    for relative, digest in manifest['files'].items():
        require(Path(relative).suffix.lower() not in code_suffixes, f'Release 包含代码或嵌套归档: {relative}')
        require(sha(contained(root, relative)) == digest, f'Release 文件哈希不一致: {relative}')
    source = Path(__file__).resolve().parents[1] / 'measurement_source'
    for relative, digest in manifest['measurement_code_files'].items():
        require(sha(contained(source, relative)) == digest, f'仓库测量源码不一致: {relative}')
    data = verify_dataset(contained(root, manifest['data_root']))
    require(data['measurement_source_sha256'] == manifest['measurement_source_sha256'], '测量源码身份不一致')
    records = sum(row['records'] for device in data['historical'].values() for row in device.values())
    records += sum(row['records'] for session, stages in data['sessions'].items()
                   if session != 'v100_c' for row in stages.values())
    records += data['replication_records']
    require(records == manifest['formal_records'], '正式记录总数不一致')
    results = contained(root, manifest['results_root'])
    require(read_json(results / 'validation.json')['passed'], '包内完整重建验证未通过')
    require(read_json(results / 'independent_verification.json')['status'] == 'verified', '包内独立核验未通过')
    require(read_json(results / 'submitted_paper_verification.json')['passed'], '包内论文数值核验未通过')
    return {'passed': True, 'files': len(files), 'formal_records': records,
            'measurement_data_files': data['files'], 'measurement_source_sha256': data['measurement_source_sha256'],
            'packaged_results': manifest['results_root']}


def main():
    parser = argparse.ArgumentParser(description='核验 Release 数据与结果、目录范围及仓库代码身份')
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = verify_release(args.release_root)
    if args.out:
        require(not args.out.resolve().is_relative_to(args.release_root.resolve()), '检查记录必须写入 Release 目录之外')
        write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
