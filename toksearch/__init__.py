# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
toksearch — Parallel signal retrieval and processing for fusion experiment data.

TokSearch provides a Pipeline abstraction for fetching, transforming, and
filtering diagnostic signals across many shots in parallel.  For DIII-D-specific
signal classes (PtDataSignal, ImasSignal, CakeSignal) and the ``fdp`` CLI, see
``help(toksearch_d3d)``.

Pipeline Lifecycle
==================

Create a pipeline from a shot list, chain operations, then compute::

    from toksearch import Pipeline, MdsSignal

    pipeline = Pipeline([202159, 202160, 202161])
    pipeline.fetch('ip', MdsSignal(r'\\ipmhd', 'efit01'))

    @pipeline.map
    def compute(rec):
        rec['max_ip'] = float(np.max(np.abs(rec['ip']['data'])))

    @pipeline.where
    def plasma(rec):
        return rec.get('max_ip', 0) > 5e5

    pipeline.keep(['shot', 'max_ip'])
    records = pipeline.compute_multiprocessing(num_workers=8)

Records
=======

Each shot is a ``Record`` — a dict-like object::

    rec['field']               # access a fetched/computed field
    rec.shot                   # shot number (int)
    rec.errors                 # dict of {field: exception} for failed fetches
    rec.get('field', None)     # safe access — BOTH arguments required

Signal Data Format
==================

All signals (``MdsSignal``, ``PtDataSignal``, ``ImasSignal``) return a dict::

    {'data': np.ndarray, 'times': np.ndarray, 'units': {...}}

Access the array via ``rec['signal_name']['data']``, not ``rec['signal_name']``.

MdsSignal
=========

Fetches MDSplus tree data.  Use raw strings for backslash node names::

    MdsSignal(r'\\ipmhd', 'efit01')                        # default location
    MdsSignal(r'\\ipmhd', 'efit01', location='remote://atlas.gat.com')

``location=None`` (default) reads ``${treename}_path`` or ``default_tree_path``
from the environment — correct for ``fdp run`` workflows.

For multi-dimensional data, use ``dims`` to label axes::

    MdsSignal(r'\\psirz', 'efit01', dims=('times',))

Datasets and Alignment
======================

Combine signals into an ``xr.Dataset`` and align to a common time base::

    pipeline.fetch_dataset('data', {
        'ip':   MdsSignal(r'\\ipmhd', 'efit01'),
        'q95':  MdsSignal(r'\\q95',   'efit01'),
    })
    pipeline.align('data', align_with=1.0, method='linear')

``align_with`` accepts: a variable name, explicit array, sample period (float),
or a callable ``(ds, dim) -> array``.

Backends
========

All backends process shots independently — choose by scale:

=========  ============================  ======================
Backend    Method                        Best for
=========  ============================  ======================
Serial     ``compute_serial()``          Development, <10 shots
Multiproc  ``compute_multiprocessing()`` Local parallel, 10-1000
Ray        ``compute_ray()``             Distributed / large
Spark      ``compute_spark()``           Existing Spark infra
=========  ============================  ======================

Working with Results
====================

All backends return an iterable ``RecordSet``.  Iterate directly::

    for rec in records:
        print(rec['shot'], rec.get('max_ip', None))

To build a pandas DataFrame, convert records explicitly —
``pd.DataFrame(records)`` does **not** work because RecordSet is not a list
of dicts::

    import pandas as pd
    df = pd.DataFrame([dict(r) for r in records])

Critical Gotchas
================

==========================  ==================================  ==============================
Gotcha                      Wrong                               Right
==========================  ==================================  ==============================
``map`` return value        ``return {'key': val}``             ``rec['key'] = val`` (in-place)
``keep`` signature          ``keep('a', 'b')``                  ``keep(['a', 'b'])``
DataFrame from results      ``pd.DataFrame(records)``           ``pd.DataFrame([dict(r)...])``
Record safe access          ``rec.get('k')``                    ``rec.get('k', None)``
==========================  ==================================  ==============================

Error Handling
==============

Failed fetches are recorded in ``rec.errors``; the field is absent::

    @pipeline.map
    def safe(rec):
        if 'ip' in rec.errors:
            rec['max_ip'] = None
            return
        rec['max_ip'] = float(np.max(np.abs(rec['ip']['data'])))

Single-Shot Debugging
=====================

Run the pipeline for one shot without building the full list::

    record = pipeline.compute_shot(202161)
"""

from .signal.signal import Signal
from .signal.zarr import ZarrSignal
from .signal.mds import MdsSignal, MdsTreePath
from .pipeline.align import XarrayAligner
from .pipeline import Pipeline

from pathlib import Path

from . import _version

__version__ = _version.get_versions()["version"]
