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

Reading a pinned version
========================

A record can name which version of a shot to read, and ``MdsSignal`` honours
it on both transports::

    Pipeline([
        {'shot': 165920, 'version': 2},                   # one shot, one version
        {'shot': 165921, 'snapshot': 'catalog_20260907T232802Z'},
    ])

``version`` pins a single shot. ``snapshot`` resolves through that catalog
rather than the newest, so one recorded value reproduces a whole campaign.
Omit both and the newest version is read, falling back to the unversioned
archive for anything the store has not absorbed.

**A pin is a guarantee, not a preference.** If it cannot be satisfied the
fetch raises ``StoreVersionError``; it never quietly answers from another
version or from the archive. That is the point -- a pin exists so a rerun can
prove it read the same bytes, and a silent substitution would destroy exactly
that.

Reading from the store needs ``ptdata >= 2.8.0`` and a deployment that
declares where its store is. Over ``fdp://`` the origin declares it; for a
local read set ``FDP_VIEWS_ROOT``. Without one, pinned reads raise and
unpinned reads behave exactly as they always have.

Pinning a whole run
===================

"Latest" is a lookup that moves. A run that performs it more than once can
read its early shots from one catalog snapshot and its later ones from the
next -- and under ``compute_multiprocessing``, Ray or Spark each worker
resolves independently, so they can disagree from the first fetch. Every
``compute_*`` call therefore settles one snapshot *before any worker exists*
and hands it to them, so a run always reads from exactly one catalog.

Three places can name it, most specific first::

    Pipeline.from_snapshot('catalog_20260907T232802Z', shots)   # 1. in code
    $ fdp run --snapshot catalog_20260907T232802Z python x.py   # 2. FDP_STORE_SNAPSHOT
    Pipeline(shots)                                             # 3. newest, frozen

Code outranks the environment. The environment outranks the default, because
there it was *told* rather than guessed. Whichever wins is recorded in the
provenance ``RunContext`` and folded into ``input_identity()`` -- two runs
over the same shots at different snapshots read different bytes, so they are
different inputs.

``from_snapshot('latest', ...)`` names *nothing*: it falls through to (2) and
then (3). That is so a script can take the snapshot as an argument without
special-casing the word, and still be overridable from the command line::

    parser.add_argument('--snapshot', default='latest')
    pipe = Pipeline.from_snapshot(args.snapshot, shots)

**A process reads from one snapshot.** Worker processes are reused between
runs and keep the environment they were started with, so a second run in the
same process cannot be pinned differently -- it raises rather than reading
from a catalog it did not report. To compare snapshots, run one process each.

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

To build a pandas DataFrame from scalar results, use ``to_dataframe``::

    df = records.to_dataframe()

``pd.DataFrame(records)`` does **not** work because RecordSet is not a list.
For records holding arrays or datasets, write per-shot files from inside the
pipeline with ``Pipeline.write`` instead.

Critical Gotchas
================

==========================  ==================================  ==============================
Gotcha                      Wrong                               Right
==========================  ==================================  ==============================
``map`` return value        ``return {'key': val}``             ``rec['key'] = val`` (in-place)
``keep`` signature          ``keep('a', 'b')``                  ``keep(['a', 'b'])``
DataFrame from results      ``pd.DataFrame(records)``           ``records.to_dataframe()``
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
from .signal.mds import MdsSignal, MdsTreePath, StoreVersionError
from .pipeline.align import XarrayAligner
from .pipeline import Pipeline

from pathlib import Path

from . import _version

__version__ = _version.get_versions()["version"]

__llm_description__ = (
    "core toksearch - Pipeline, MdsSignal, ZarrSignal, fetch_dataset, "
    "and SQL helpers"
)
