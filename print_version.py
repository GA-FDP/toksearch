import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import versioneer

def get_version():
    return versioneer.get_version()

if __name__ == "__main__":
    print(get_version())
