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

import os
import getpass
from pathlib import Path

import pymssql

USER_HOME_DIR = str(Path.home())
USERNAME = getpass.getuser()
DEFAULT_PASSWORD_FILE = os.path.join(USER_HOME_DIR, "D3DRDB.sybase_login")


def _read_sybase_login_file(filename):
    with open(filename, "r") as f:
        username, password = [line.strip() for line in f.readlines()]
    return username, password


def connect_d3drdb(
    username=USERNAME,
    password=None,
    host="d3drdb.gat.com",
    db="d3drdb",
    port=8001,
    password_file=DEFAULT_PASSWORD_FILE,
):
    """
    Connect to the d3drdb

    Keyword Parameters:
        host (str): Defaults to 'd3drdb.gat.com'
        port (int): Defaults to 8001
        db (str): Name of the db (e.g. d3drdb, confinement). Defaults to d3drdb
        username (str): Defaults to current user
        password (str): Defaults to None. If not set, an attempt will be made
            to read a password file
       password_file (str): Full path the a password file, formatted with
            username on the first line and password on the second line.

    Returns:
        Python DB API compliant Connection object

    Can be used to return a Connection object directly, or used
    as a context manager.
    Examples:
       conn = connect_d3drdb()

       with connect_d3drdb() as conn:
           # Do stuff with conn

    """

    if password is None:
        username, password = _read_sybase_login_file(password_file)

    conn = pymssql.connect(host, username, password, db, port=str(port))
    return conn
