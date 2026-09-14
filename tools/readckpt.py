#!/usr/bin/env python3
"""readckpt.py -- 用純 numpy 讀 libtorch 的 checkpoint（不需要安裝 torch）

torch::save(module, path) 產生的是 TorchScript zip archive。這支把裡面所有
tensor 抽出來存成 .npz，供 check_value_head.py / erf.py 使用。

用法:
    python3 readckpt.py checkpoint-193.pt ck193.npz
"""
import collections
import io
import pickle
import sys
import zipfile

import numpy as np

DTYPES = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64,
    'IntStorage': np.int32, 'HalfStorage': np.float16, 'BoolStorage': np.bool_,
}


def load(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split('/')[0]

    def maketensor(storage, offset, size, stride, *rest):
        _, stype, key, _, numel = storage
        nm = stype if isinstance(stype, str) else getattr(stype, '__name__', str(stype))
        arr = np.frombuffer(z.read(f'{root}/data/{key}'),
                            dtype=DTYPES.get(nm.split('.')[-1], np.float32))
        n = int(np.prod(size)) if len(size) else 1
        return arr[offset:offset + n].reshape(tuple(size))

    class Base:
        """佔位類別：TorchScript 的物件圖用不到語意，只要能被 unpickle 就行。"""
        def __setstate__(self, st):
            self.__st__ = st

        def __init__(self, *a, **k):
            if a:
                self.__st__ = a

    cache = {}

    def mkclass(full):
        if full not in cache:
            cache[full] = type('C_' + full.replace('.', '_'), (Base,), {'__full__': full})
        return cache[full]

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if '_rebuild_tensor' in name:
                return maketensor
            if name.startswith('build_') or name == 'restore_type_tag':
                return lambda *a, **k: (a[0] if a else None)
            return mkclass(mod + '.' + name)

        def persistent_load(self, pid):
            return pid

    obj = Unpickler(io.BytesIO(z.read(f'{root}/data.pkl'))).load()

    found, seen = {}, set()

    def walk(o, prefix='', depth=0):
        if depth > 14:
            return
        if isinstance(o, np.ndarray):
            found[prefix] = o
            return
        if id(o) in seen:
            return
        if isinstance(o, (dict, list, tuple, Base)):
            seen.add(id(o))
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f'{prefix}.{k}' if prefix else str(k), depth + 1)
        elif isinstance(o, (list, tuple)):
            for j, v in enumerate(o):
                walk(v, f'{prefix}[{j}]', depth + 1)
        elif isinstance(o, Base):
            if getattr(o, '__st__', None) is not None:
                walk(o.__st__, prefix, depth + 1)
            for k, v in vars(o).items():
                if not k.startswith('__'):
                    walk(v, f'{prefix}.{k}' if prefix else k, depth + 1)

    walk(obj)
    return found


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    tensors = load(sys.argv[1])
    for k, v in tensors.items():
        print(f'{k:70s} {str(v.shape):18s} {v.dtype}')
    print('total tensors:', len(tensors),
          ' floats:', sum(int(np.prod(v.shape)) for v in tensors.values()))
    np.savez(sys.argv[2], **tensors)
    print('->', sys.argv[2])
