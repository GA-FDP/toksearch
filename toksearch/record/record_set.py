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

from abc import ABC, abstractmethod


class RecordSet(ABC):
    """Abstract base class for a set of Records

    This class provides a common interface for a set of Records
    that are produced by the various execution backends.

    Methods:
        __len__: Returns the number of records in the set
        __getitem__: Returns a record by index
        __iter__: Iterates over the records in the set as a generator
        map: Applies one or more functions to the records
        cleanup: Cleans up any resources used by the RecordSet, such as
            shutting down a SparkContext or Ray cluster
    """

    #: Set by Pipeline.compute when a provenance backend is in use. Carrying
    #: it on the record set is what makes an in-process pipeline chain
    #: automatically linkable: Pipeline(previous_recordset) can read it.
    provenance = None
    run_id = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def map(self, *operations):
        pass

    @abstractmethod
    def cleanup(self, **kwargs):
        pass

    #: Record fields that are bookkeeping, not results. Excluded from
    #: to_dataframe unless explicitly requested.
    _NON_RESULT_FIELDS = ("errors", "_toksearch_write_dir")

    def to_dataframe(self, fields=None):
        """Return a pandas DataFrame with one row per record.

        Intended for scalar summary results. Array-valued and dataset-valued
        fields belong in per-shot files written by ``Pipeline.write``, not in
        a driver-side frame.

        Rows are built from each record's own keys rather than one shared
        column list: a shot whose map function failed never gains the field
        its siblings have, and pandas unions the columns and fills NaN. That
        keeps a partial run visibly partial instead of silently smaller.

        Arguments:
            fields: Field names to include, in order. When omitted, every
                field except bookkeeping ones is included. ``shot`` is always
                the first column.
        """
        import pandas as pd

        rows = []
        for record in self:
            if fields is None:
                keys = [
                    k
                    for k in record.keys()
                    if k != "shot" and k not in self._NON_RESULT_FIELDS
                ]
            else:
                keys = list(fields)

            row = {"shot": record.shot}
            for key in keys:
                row[key] = record.get(key, None)
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["shot"] + list(fields or []))

        return pd.DataFrame(rows)

    def to_parquet(self, path, fields=None):
        """Write the record set to a parquet file and return its path.

        If this record set came from a ``compute_*`` call with a provenance
        backend, the file is declared to that backend as an output artifact.
        The file is written first: a provenance failure must not cost the user
        their output.
        """
        self.to_dataframe(fields=fields).to_parquet(path)

        if self.provenance is not None:
            from ..provenance.base import safe_call

            safe_call(self.provenance, "output", path, source="to_parquet")

        return path
