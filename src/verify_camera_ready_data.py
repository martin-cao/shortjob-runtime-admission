import argparse
import json
from pathlib import Path

from camera_ready_data import verify_dataset, write_json


def main():
    parser = argparse.ArgumentParser(description='核验正式数据、任务集合及仓库中的测量源码身份')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--legacy-archive', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = verify_dataset(args.data_root, args.legacy_archive)
    if args.out:
        if args.out.resolve().is_relative_to(args.data_root.resolve()):
            raise ValueError('验证结果必须写入数据目录之外')
        write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['historical', 'sessions', 'replication']}, indent=2))


if __name__ == '__main__':
    main()
