import argparse
import json
from pathlib import Path

from camera_ready_data import require, sha, write_json


def main():
    parser = argparse.ArgumentParser(description='逐文件核验重新计算的结果与 Release 中的科学数据')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reproduced', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reference, reproduced = args.reference.resolve(), args.reproduced.resolve()
    require(reference != reproduced, '必须比较两个独立结果目录')
    files = {str(p.relative_to(reference)) for p in reference.rglob('*')
             if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv'}}
    actual = {str(p.relative_to(reproduced)) for p in reproduced.rglob('*')
              if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv'}}
    require(files == actual and bool(files), '重建结果文件集合不一致')
    for relative in sorted(files):
        require(sha(reference / relative) == sha(reproduced / relative), f'重建结果字节不一致: {relative}')
    result = {'passed': True, 'scientific_result_files': len(files), 'comparison': 'SHA-256 for every JSON, JSONL and CSV result'}
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
