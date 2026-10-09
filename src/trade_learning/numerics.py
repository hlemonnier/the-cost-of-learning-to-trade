"""Prespecified numerical decisions and reproducible control-artifact identity."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy


TIE_ABSOLUTE_TOLERANCE = 1e-12
CONTROL_FORMAT_VERSION = 2


def numerical_contract():
    return {
        'version': 2,
        'score_dtype': 'float64',
        'tie_absolute_tolerance': TIE_ABSOLUTE_TOLERANCE,
        'tie_relative_tolerance': 0.0,
        'tie_action': 'lowest admissible action ID within absolute tolerance of maximum',
        'bellman_value': 'Q of the action selected by the same tie rule',
        'invalid_action': 'negative infinity; no decision at remaining horizon zero',
    }


def stable_argmax(scores, admissible=None, axis=-1):
    """Select the lowest legal index within the fixed absolute tolerance.

    No value-dependent relative tolerance is used. Explicit masks can exclude
    otherwise finite scores, but every admitted score must be finite.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 0 or scores.shape[axis] == 0:
        raise ValueError('A nonempty action axis is required')
    legal = ~np.isneginf(scores) if admissible is None else np.broadcast_to(
        np.asarray(admissible, dtype=bool), scores.shape)
    if np.any(legal & ~np.isfinite(scores)):
        raise ValueError('Every admissible action score must be finite')
    if not np.all(np.any(legal, axis=axis)):
        raise ValueError('Every decision row must contain an admissible action')
    best = np.max(np.where(legal, scores, -np.inf), axis=axis, keepdims=True)
    with np.errstate(invalid='ignore', over='ignore'):
        candidates = legal & ((best-scores) <= TIE_ABSOLUTE_TOLERANCE)
    return np.argmax(candidates, axis=axis)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint():
    root = Path(__file__).parent
    files = {name: file_sha256(root/name)
             for name in ('control.py', 'model.py', 'numerics.py', 'policies.py')}
    return {'files': files, 'sha256': canonical_hash(files)}


@lru_cache(maxsize=1)
def _runtime_build():
    # Hash actual numerical extension binaries as well as package versions/config.
    binaries = {}
    for name in ('numpy._core._multiarray_umath', 'scipy.special._ufuncs',
                 'scipy.sparse._sparsetools', 'scipy.linalg._fblas'):
        module = importlib.import_module(name)
        binaries[name] = file_sha256(module.__file__)
    return {
        'python': platform.python_version(), 'implementation': platform.python_implementation(),
        'system': platform.system(), 'machine': platform.machine(), 'byteorder': sys.byteorder,
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'numpy_config': json.loads(json.dumps(np.__config__.CONFIG, default=str)),
        'scipy_config': json.loads(json.dumps(scipy.__config__.CONFIG, default=str)),
        'numerical_binaries': binaries,
    }


def build_fingerprint():
    result = deepcopy(_runtime_build())
    result['thread_environment'] = {key: os.environ.get(key) for key in
        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    result['sha256'] = canonical_hash(result)
    return result


def control_key(specification):
    # Source/build deliberately do not change the logical address: a stale entry
    # at this address must be rejected, rather than silently evaded and rebuilt.
    return 'v2-'+canonical_hash({'specification': specification,
                                'format_version': CONTROL_FORMAT_VERSION})[:24]


def artifact_fingerprint(record):
    fields = ('format_version', 'key', 'specification', 'numerical_contract',
              'source', 'build', 'arrays', 'admissibility_sha256')
    return canonical_hash({field: record.get(field) for field in fields})


def manifest_fingerprint(record):
    fields = ('key', 'specification', 'artifact_sha256', 'arrays',
              'source', 'build', 'numerical_contract')
    return deepcopy({field: record[field] for field in fields})
