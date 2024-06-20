import yaml
from toksearch.imas_mappings import tdi_utils
import glob
import os

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
                                self.machine_mappings[machine]['__cocos_rules__'][item]['eval2TDI'].replace('\\', '\\\\'), tdi_utils
                            )
                        except Exception as _excp:
                            text = f"Error evaluating eval2TDI in ['__cocos_rules__'][{item!r}]: {self.machine_mappings[machine]['__cocos_rules__'][item]['eval2TDI']}:\n{_excp!r}"
                            raise _excp.__class__(text)

                # generate TDI and sanity check mappings
                for location in self.machine_mappings[machine]:
                    # generate DTI functions based on eval2DTI
                    if 'eval2TDI' in self.machine_mappings[machine][location]:
                        self.machine_mappings[machine][location]['TDI'] = eval(self.machine_mappings[machine][location]['eval2TDI'].replace('\\', '\\\\'), tdi_utils)


    def resolve_mapped(ods, machine, pulse,  mappings, location, idm, options_with_defaults, branch, cache=None):
        """
        Routine to resolve a mapping

        :param ods: input ODS to populate

        :param machine: machine name

        :param pulse: pulse number

        :param mappings: Dictionary of available mappings

        :param location: ODS location to be resolved

        :param idm: Tuple with machine and branch

        :param options_with_defaults: dictionary with options to use when loading the data including default settings

        :param branch: load machine mappings and mapping functions from a specific GitHub branch

        :param cache: if cache is a dictionary, this will be used to establiish a cash

        :return: updated ODS and data before being assigned to the ODS
        """
        mapped = mappings[location]
        # cocosio
        cocosio = None
        if 'COCOSIO' in mapped:
            if isinstance(mapped['COCOSIO'], int):
                cocosio = mapped['COCOSIO']
        elif 'COCOSIO_PYTHON' in mapped:
            call = mapped['COCOSIO_PYTHON'].format(**options_with_defaults)
            if cache and call in cache:
                cocosio = cache[call]
            else:
                namespace = {}
                namespace.update(_namespace_mappings[idm])
                namespace['__file__'] = machines(machine, branch)[:-5] + '.py'
                tmp = compile(call, machines(machine, branch)[:-5] + '.py', 'eval')
                cocosio = eval(tmp, namespace)
                if isinstance(cache, dict):
                    cache[call] = cocosio
        elif 'COCOSIO_TDI' in mapped:
            TDI = mapped['COCOSIO_TDI'].format(**options_with_defaults)
            treename = mapped['treename'].format(**options_with_defaults) if 'treename' in mapped else None
            cocosio = int(mdsvalue(machine, treename, pulse, TDI).raw())

        # CONSTANT VALUE
        if 'VALUE' in mapped:
            data0 = data = mapped['VALUE']

        # EVAL
        elif 'EVAL' in mapped:
            data0 = data = eval(mapped['EVAL'].format(**options_with_defaults), _namespace_mappings[idm])

        # ENVIRONMENTAL VARIABLE
        elif 'ENVIRON' in mapped:
            data0 = data = os.environ.get(mapped['ENVIRON'].format(**options_with_defaults))
            if data is None:
                raise ValueError(
                    f'Environmental variable {mapped["ENVIRON"].format(**options_with_defaults)} is not defined'
                )

        # PYTHON
        elif 'PYTHON' in mapped:
            if 'mast' in machine:
                printe(f"MAST is currently not supported because of UDA", topic='machine')
                return ods
            call = mapped['PYTHON'].format(**options_with_defaults)
            # python functions tend to set multiple locations at once
            # it is thus very beneficial to cache that
            if cache and call in cache:
                ods = cache[call]
            else:
                namespace = {}
                namespace.update(_namespace_mappings[idm])
                namespace['ods'] = ODS()
                namespace['__file__'] = machines(machine, branch)[:-5] + '.py'
                printd(f"Calling `{call}` in {os.path.basename(namespace['__file__'])}", topic='machine')
                # Add the callback for mapping updates
                # By supplyinh the function to the decorator we avoid a ringinclusion
                call_w_update_mapping = call[:-1] + ", update_callback=update_mapping)"
                exec( machine + "." + call_w_update_mapping)
                if isinstance(cache, dict):
                    cache[call] = ods
            if location.endswith(':'):
                return (
                    int(len(ods[u2n(location[:-2], [0] * 100)])),
                    {'raw_data': ods, 'processed_data': ods, 'cocosio': cocosio, 'branch': mappings['__branch__']},
                )
            else:
                return ods, {'raw_data': ods, 'processed_data': ods, 'cocosio': cocosio, 'branch': mappings['__branch__']}

        # MDS+
        elif 'TDI' in mapped:
            try:
                TDI = mapped['TDI'].format(**options_with_defaults)
                treename = mapped['treename'].format(**options_with_defaults) if 'treename' in mapped else None
                data0 = data = mdsvalue(machine, treename, pulse, TDI).raw()
                if data is None:
                    raise ValueError('data is None')
            except Exception as e:
                printe(mapped['TDI'].format(**options_with_defaults).replace('\\n', '\n'))
                if "eval2TDI" in mapped:
                    e.eval2TDI = mapped['eval2TDI']
                e.TDI = mapped['TDI']
                raise e

        else:
            raise ValueError(f"Could not fetch data for {location}. Must define one of {machine_expression_types}")

        # handle size definition for array of structures
        if location.endswith(':'):
            return int(data), {'raw_data': data0, 'processed_data': data, 'cocosio': cocosio, 'branch': mappings['__branch__']}

        # transpose manipulation
        if mapped.get('TRANSPOSE', False):
            for k in range(len(mapped['TRANSPOSE']) - len(data.shape)):
                data = numpy.array([data])
            data = numpy.transpose(data, mapped['TRANSPOSE'])

        # transpose filter
        nanfilter = lambda x: x
        if mapped.get('NANFILTER', False):
            #lambda x: x[~numpy.isnan(x)]
            nanfilter = remove_nans

        # assign data to ODS
        if not hasattr(data, 'shape'):
            ods[location] = data
        else:
            with omas_environment(ods, cocosio=cocosio):
                dsize = len(data.shape)  # size of the data fetched from MDS+
                csize = len(mapped.get('COORDINATES', []))  # number of coordinates
                osize = len([c for c in mapped.get('COORDINATES', []) if c != '1...N'])  # number of named coordinates
                asize = location.count(':') + csize  # data size required from MDS+ to make the assignement
                if asize != dsize:
                    raise Exception(
                        f"Experiment data {data.shape} does not fit in `{location}` [{', '.join([':'] * location.count(':') + mapped.get('COORDINATES', []))}]"
                    )
                if dsize - osize == 0 or ':' not in location:
                    if data.size == 1:
                        data = data.item()
                    ods[location] = nanfilter(data)
                else:
                    for k in itertools.product(*list(map(range, data.shape[: location.count(':')]))):
                        ods[u2n(location, k)] = nanfilter(data[k])

        return ods, {'raw_data': data0, 'processed_data': data, 'cocosio': cocosio, 'branch': mappings['__branch__']}


_machine_mappings = {}
_namespace_mappings = {}
_user_machine_mappings = {}
_python_tdi_namespace = {}


