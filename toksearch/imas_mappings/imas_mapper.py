import yaml
from toksearch.imas_mappings import tdi_utils
import glob
import os
import numpy as np
from toksearch.imas_mappings.mds_for_testing import mdstree, mdsvalue

defaults = {"EFIT_tree": "EFIT01"}

def remove_nans(x):
    import numpy as np
    if np.isscalar(x):
        if np.isnan(x):
            raise ValueError("Behavior of Nan filter undefined for scalar nan values")
        else:
            return x
    else:
        return x[~np.isnan(x)]

class IMASMapper:
    """
    Loads Rosetta Stone of mappings from yaml files. Provides list of signals 
    that need to be loaded for IMAS entries to be loaded in a single query.
    Assembles imas entries from multiple signals, does unit/COCOS conversions.    
    """
    def __init__(self):
        """
        Function to load the yaml mapping files (local or remote)
        This function sanity-checks and the mapping file and adds extra info required for mapping

        :param machine: machine for which to load the mapping files

        :param branch: GitHub branch from which to load the machine mapping information

        :param user_machine_mappings: Dictionary of mappings that users can pass to this function to temporarily use their mappings
                                    (useful for development and testinig purposes)

        :param return_raw_mappings: Return mappings without following __include__ statements nor resoliving `eval2TDI` directives

        :param raise_errors: raise errors or simply print warnings if something isn't right

        :return: dictionary with mapping transformations
        """
        self.machine_mappings = {}

        # figure out mapping file
        mapping_dir = os.path.join(os.path.dirname(__file__), "mapping_files")
        for machine_dir in glob.glob(os.path.join(mapping_dir, "*")):
            machine = os.path.basename(machine_dir)
            for filename in glob.glob(os.path.join(machine_dir, "*.yaml")):
                # load mappings from file following __include__ directives
                with open(filename, 'r') as f:
                    self.machine_mappings[machine] = yaml.load(f, yaml.CBaseLoader)
                # generate TDI for cocos_rules
                for item in self.machine_mappings[machine]['__cocos_rules__']:
                    if 'eval2TDI' in self.machine_mappings[machine]['__cocos_rules__'][item]:
                        try:
                            self.machine_mappings[machine]['__cocos_rules__'][item]['TDI'] = eval(
                                self.machine_mappings[machine]['__cocos_rules__'][item]['eval2TDI'].replace('\\', '\\\\'), tdi_utils.__dict__
                            )
                        except Exception as _excp:
                            text = f"Error evaluating eval2TDI in ['__cocos_rules__'][{item!r}]: {self.machine_mappings[machine]['__cocos_rules__'][item]['eval2TDI']}:\n{_excp!r}"
                            raise _excp.__class__(text)

                # generate TDI and sanity check mappings
                for location in self.machine_mappings[machine]:
                    # generate DTI functions based on eval2DTI
                    if 'eval2TDI' in self.machine_mappings[machine][location]:
                        self.machine_mappings[machine][location]['TDI'] = eval(self.machine_mappings[machine][location]['eval2TDI'].replace('\\', '\\\\'), tdi_utils.__dict__)


    def resolve_mapped(self, machine, pulses, imas_paths, options=defaults):
        """
        Routine to resolve a mapping

        :param machine: machine name

        :param pulse: pulse numbers to load

        :param imas_paths: IMAS paths to load

        :param options: Special options like EFIT_tree specifications
        """
        dd = {}
        for pulse in pulses:
            dd[pulse] = {}
            for imas_path in imas_paths:
                if imas_path not in self.machine_mappings[machine]:
                    dd[pulse][imas_path] = NotImplementedError(f"Mapping for {imas_path} not yet implemented")
                    continue
                mapped = self.machine_mappings[machine][imas_path]
                # CONSTANT VALUE
                if 'VALUE' in mapped:
                    dd[pulse][imas_path] = mapped['VALUE']

                # EVAL
                elif 'EVAL' in mapped:
                    dd[pulse][imas_path] = eval(mapped['EVAL'].format(**options), tdi_utils.__dict__)

                # ENVIRONMENTAL VARIABLE
                elif 'ENVIRON' in mapped:
                    dd[pulse][imas_path] = os.environ.get(mapped['ENVIRON'].format(**options))
                    if dd[pulse][imas_path] is None:
                        raise ValueError(
                            f'Environmental variable {mapped["ENVIRON"].format(**options)} is not defined'
                        )

                # PYTHON
                elif 'PYTHON' in mapped:
                    call = mapped['PYTHON'].format(**options)
                    # python functions tend to set multiple locations at once
                    # it is thus very beneficial to cache that
                    # Add the callback for mapping updates
                    # By supplyinh the function to the decorator we avoid a ringinclusion
                    call_w_update_mapping = call[:-1] + ", update_callback=update_mapping)"
                    exec( machine + "." + call_w_update_mapping)
                # MDS+
                elif 'TDI' in mapped:
                    try:
                        TDI = mapped['TDI'].format(**options)
                        treename = mapped['treename'].format(**options) if 'treename' in mapped else None
                        dd[pulse][imas_path] = mdsvalue(machine, treename, pulse, TDI).raw()
                        if dd[pulse][imas_path] is None:
                            raise ValueError('Data is None')
                    except Exception as e:
                        print(mapped['TDI'].format(**options).replace('\\n', '\n'))
                        if "eval2TDI" in mapped:
                            e.eval2TDI = mapped['eval2TDI']
                        e.TDI = mapped['TDI']
                        raise e

                else:
                    raise ValueError(f"Could not fetch data for {imas_path}.")

                # handle size definition for array of structures
                if mapped.get('TRANSPOSE', False):
                    for k in range(len(mapped['TRANSPOSE']) - len(dd[pulse][imas_path].shape)):
                        dd[pulse][imas_path] = np.array([dd[pulse][imas_path]])
                    dd[pulse][imas_path] = np.transpose(dd[pulse][imas_path], mapped['TRANSPOSE'])

                # transpose filter
                nanfilter = lambda x: x
                if mapped.get('NANFILTER', False):
                    #lambda x: x[~np.isnan(x)]
                    nanfilter = remove_nans
                dd[pulse][imas_path] =  nanfilter(dd[pulse][imas_path])

        return dd
