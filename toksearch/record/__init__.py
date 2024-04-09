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


from typing import Any, List
from ..utilities.utilities import capture_exception


class InvalidShotNumber(Exception):
    pass


class InvalidRecordField(Exception):
    pass


class MissingShotNumber(Exception):
    pass


class Record(object):
    """A class to hold a record of data for a single shot

    The Record class is a dictionary-like object that holds data for a single shot.
    It has a 'shot' field that is required to be an integer, and it has an 'errors'
    field that is a dictionary of errors that have occurred while processing the data.


    Most of the time, you will not need to instantiate a Record object directly. Instead,
    the Pipeline class will create Record objects for you.
    """

    @classmethod
    def from_dict(cls, input_dict: dict) -> "Record":
        """
        Instantiate a Record object from a dictionary

        Arguments:
            input_dict: Must contain the key 'shot', and
                must NOT contain the keys 'key' or 'errors'

        Returns:
            Record: A Record object with the fields from input_dict

        Raises:
            MissingShotNumber: If the input_dict does not contain the key 'shot'
            InvalidRecordField: If the input_dict contains the key 'key' or 'errors'
        """
        try:
            shot = input_dict["shot"]
        except KeyError as e:
            raise MissingShotNumber("Cannot create Record without a shot number")

        rec = cls(shot)

        for key, val in input_dict.items():
            if key == "shot":
                continue
            elif key in {"key", "errors"}:
                raise InvalidRecordField(
                    f"Illegal field: {key} when trying to create Record"
                )

            rec[key] = val

        return rec

    def __init__(self, shot: int):
        """Create a Record object with a shot number

        Parameters:
            shot: The shot number for this record

        Raises:
            InvalidShotNumber: If the shot number is not castable to an integer
        """
        try:
            shot = int(shot)
        except Exception as e:
            raise InvalidShotNumber(
                f"Shot must be castable to type int. Got {shot} of type {type(shot)}"
            )

        self.shot = shot

        self["errors"] = {}
        self.errors = self["errors"]

    def __getitem__(self, key: str):
        """Get a field from the record"""
        return self.__dict__[key]

    def __setitem__(self, key, value):
        """Set a field in the record"""
        self.__dict__[key] = value

    def __delitem__(self, key):
        """Delete a field from the record"""
        del self.__dict__[key]

    def __contains__(self, key):
        """Check if a field is in the record"""
        return key in self.__dict__

    def __len__(self):
        """Get the number of fields in the record"""
        return len(self.__dict__)

    def __repr__(self) -> str:
        """Get a string representation of the record"""
        return repr(self.__dict__)

    def get(self, key, default):
        """Get a field from the record with a default value if the field does not exist"""
        return self.__dict__.get(key, default)

    def keys(self):
        """Get the keys of the record"""
        return self.__dict__.keys()

    def set_error(self, field: str, exception: Exception):
        """Set an error for a field in the record

        Updates the 'errors' field in the record with key 'field'
        to store the exception (including the traceback in a serializable format)

        Parameters:
            field: The field name to set the error for
            exception: The exception to store
        """
        f = field
        i = 1
        while f in self.errors:
            f = "{}_{}".format(field, i)
            i += 1

        self.errors[f] = capture_exception(f, exception)

    def pop(self, key) -> Any:
        """Remove a field from the record"""
        if key not in {"key", "shot", "errors"}:
            self.__dict__.pop(key)

    def keep(self, keys: List[Any]):
        """Remove all fields from the record that are not in keys

        Parameters:
            keys: A list of keys to keep in the record
        """
        for existing_key in list(self.keys()):
            if existing_key not in keys:
                self.pop(existing_key)

    def discard(self, keys: List[Any]):
        """Remove all fields in keys from the record

        Parameters:
            keys: A list of keys to remove from the record
        """
        for key in keys:
            self.pop(key)

