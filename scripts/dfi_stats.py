#!/usr/bin/env python

import argparse
import os
import sqlite3
import random


from multiprocessing import Pool

from toksearch.datasource.ptdata_fetch import PtDataHeader
from toksearch.datasource.ptdata_locator import PtDataLocator


def _get_dfi(shot, pointname):
    header = PtDataHeader(pointname, shot)
    return pointname, header.dfi()


def get_dfis_for_shot(shot, max_workers=10):
    """
    Given a shot number, return a dictionary with a list of pointames per dfi

    Restrictions: Only works with local indexes
    """

    db_file = PtDataLocator.index_file(shot)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("select pointname from pointmap where shot = ?", (shot,))
    rows = cursor.fetchall()

    pointnames = [row[0] for row in rows]

    shot_ptnames = [(shot, pointname) for pointname in pointnames]

    with Pool(max_workers) as p:
        pt_dfi = p.starmap(_get_dfi, shot_ptnames)
    return pt_dfi


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("shot", type=int)
    parser.add_argument("-n", "--num-workers", type=int, default=15)
    args = parser.parse_args()
    print(args)

    shot = args.shot
    num_workers = args.num_workers

    pt_dfis = get_dfis_for_shot(shot, max_workers=num_workers)

    dfi_map = {}
    for pointname, dfi in pt_dfis:
        if dfi not in dfi_map:
            dfi_map[dfi] = []

        dfi_map[dfi].append(pointname)

    dfi_counts = [(dfi, len(pointnames)) for dfi, pointnames in dfi_map.items()]

    dfi_counts.sort(key=lambda x: (x[1], x[0]), reverse=True)

    for dfi, count in dfi_counts:
        pointnames = dfi_map[dfi]
        random_pointnames = random.sample(pointnames, min(len(pointnames), 5))
        print(dfi, count, random_pointnames)
