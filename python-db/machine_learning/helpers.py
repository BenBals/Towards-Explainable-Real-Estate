import os
from pathlib import Path


def create_dirs_if_not_existing(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_vcs_root(test='.', dirs=(".git",), default='.'):
    import os
    prev, test = None, os.path.abspath(test)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, d)) for d in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    return default


def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


def find_latest(name, path):
    all_files = sorted(find_all(name, path))
    if not all_files:
        return None
    return all_files[-1]
