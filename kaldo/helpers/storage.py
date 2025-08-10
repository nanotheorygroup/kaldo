import numpy as np
import os
import h5py
from kaldo.helpers.logger import get_logger
logging = get_logger()


LAZY_PREFIX = '_lazy__'
FOLDER_NAME = 'data'



def parse_pair(txt):
    return complex(txt.strip("()"))


def load(property, folder, instance, format='formatted'):
    # TODO: move this into single observables
    name = folder + '/' + property
    if format == 'numpy':
        loaded = np.load(name + '.npy', allow_pickle=True)
        return loaded
    elif format == 'hdf5':
        with h5py.File(name.split('/')[0] + '.hdf5', 'r') as storage:
            loaded = storage[name]
            return loaded[()]
    elif format == 'formatted':
        if property == 'physical_mode':
            loaded = np.loadtxt(name + '.dat', skiprows=1)
            loaded = np.round(loaded, 0).astype(bool)
        elif property == 'velocity':
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1))
            loaded = np.array(loaded).transpose(1, 2, 0)
        elif property == 'mean_free_path':
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1))
            loaded = np.array(loaded).transpose(1, 2, 0)
        elif property == 'conductivity':
            loaded = []
            for alpha in range(3):
                for beta in range(3):
                    loaded.append(np.loadtxt(name + '_' + str(alpha) + '_' + str(beta) + '.dat', skiprows=1))
            loaded = np.array(loaded).reshape((3, 3, instance.n_phonons)).transpose(2, 0, 1)
        elif '_sij' in property:
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1, dtype=complex))
            loaded = np.array(loaded).transpose(1, 0)
        else:
            if property == 'diffusivity':
                dt = complex
            else:
                dt = float
            loaded = np.loadtxt(name + '.dat', skiprows=1, dtype=dt)
        return loaded
    elif format == 'memory':
        if hasattr(instance, LAZY_PREFIX + property):
            logging.info('Loading from memory ' + str(property) + ' property.')
            return getattr(instance, LAZY_PREFIX + property)
        else:
            logging.info(str(property) + ' not found.')
            raise KeyError('Property not found')
    else:
        raise ValueError('Storing format not implemented')


def save(property, folder, loaded_attr, format='formatted'):
    # TODO: move this into single observables
    name = folder + '/' + property
    if format == 'numpy':
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(name + '.npy', loaded_attr)
        logging.info(name + ' stored')
    elif format == 'hdf5':
        with h5py.File(name.split('/')[0] + '.hdf5', 'a') as storage:
            if not name in storage:
                storage.create_dataset(name, data=loaded_attr, chunks=True, compression='gzip', compression_opts=9)
        logging.info(name + ' stored')
    elif format == 'formatted':
        # loaded_attr = np.nan_to_num(loaded_attr)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if property == 'physical_mode':
            fmt = '%d'
        else:
            fmt = '%.18e'
        if property == 'velocity':
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', loaded_attr[..., alpha], fmt=fmt, header=str(loaded_attr[..., 0].shape))
        elif property == 'mean_free_path':
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', loaded_attr[..., alpha], fmt=fmt,
                           header=str(loaded_attr[..., 0].shape))
        elif '_sij' in property:
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', loaded_attr[..., alpha].flatten(), fmt=fmt, header=str(loaded_attr[..., 0].shape))
        elif 'conductivity' in property:
            for alpha in range(3):
                for beta in range(3):
                    np.savetxt(name + '_' + str(alpha) + '_' + str(beta) + '.dat', loaded_attr[..., alpha, beta], fmt=fmt,
                           header=str(loaded_attr[..., 0, 0].shape))
        else:
            np.savetxt(name + '.dat', loaded_attr, fmt=fmt, header=str(loaded_attr.shape))
    elif format=='memory':
        logging.warning('Property ' + str(property) + ' will be lost when calculation is over.')
    else:
        raise ValueError('Storing format not implemented')


def get_folder_from_label(instance, label='', base_folder=None):
    # TODO: move this into single observables
    if base_folder is None:
        if instance.folder:
            base_folder = instance.folder
        else:
            base_folder = FOLDER_NAME
    try:
        if np.prod(instance.kpts) > 1:
            kpts = instance.kpts
            base_folder += '/' + str(kpts[0]) + '_' + str(kpts[1]) + '_' + str(kpts[2])
    except AttributeError:
        try:
            q_point = instance.q_point
            base_folder += '/single_q/' + str(q_point[0]) + '_' + str(q_point[1]) + '_' + str(q_point[2])
        except AttributeError:
            pass
    if label != '':
        if '<diffusivity_bandwidth>' in label:
            if instance.diffusivity_bandwidth is not None:
                base_folder += '/db_' + str(np.mean(instance.diffusivity_bandwidth))
        if '<diffusivity_threshold>' in label:
            if instance.diffusivity_threshold is not None:
                base_folder += '/dt_' + str(instance.diffusivity_threshold)

        if '<temperature>' in label:
            base_folder += '/' + str(int(instance.temperature))
        if '<statistics>' in label:
            if instance.is_classic:
                base_folder += '/classic'
            else:
                base_folder += '/quantum'
        if '<third_bandwidth>' in label:
            if instance.third_bandwidth is not None:
                base_folder += '/tb_' + str(np.mean(instance.third_bandwidth))
        if '<include_isotopes>' in label:
            if instance.include_isotopes:
                base_folder +='/isotopes'

        if '<method>' in label:
            base_folder += '/' + str(instance.method)
            if (instance.method == 'rta' or instance.method == 'sc' or instance.method == 'inverse') \
                    and (instance.length is not None):
                if not (np.array(instance.length) == np.array([None, None, None])).all()\
                    and not (np.array(instance.length) == np.array([0, 0, 0])).all():
                    if '<length>' in label:
                        base_folder += '/l'
                        for alpha in range(3):
                            if instance.length[alpha] is not None:
                                base_folder += '_' + str(instance.length[alpha])
                            else:
                                base_folder += '_0'
                    if '<finite_length_method>' in label:
                        if instance.finite_length_method is not None:
                            base_folder += '/fs' + str(instance.finite_length_method)
    return base_folder


def lazy_property(label='', format=None):
    """
    Decorator for lazy evaluation of properties with storage support.
    
    Parameters
    ----------
    label : str
        Label for folder structure generation (e.g., '<temperature>/<statistics>')
    format : str, optional
        Storage format for this specific property. If None, uses object's format hierarchy.
    
    Returns
    -------
    property
        A property that lazy loads and caches data with storage support
    """
    def _lazy_property(fn):
        @property
        def __lazy_property(self):
            # Determine storage format using hierarchy
            storage_format = _get_storage_format(self, fn.__name__, format)
            
            if storage_format != 'memory':
                folder = get_folder_from_label(self, label)
                property_name = fn.__name__
                try:
                    loaded_attr = load(property_name, folder, self, format=storage_format)
                    logging.info('Loading ' + folder + '/' + str(property_name))
                except (FileNotFoundError, OSError, KeyError):
                    logging.info(folder + '/' + str(property_name) + ' not found in ' + storage_format + ' format, calculating ' + str(fn.__name__))
                    loaded_attr = fn(self)
                    save(property_name, folder, loaded_attr, format=storage_format)
            else:
                # Memory storage
                attr = LAZY_PREFIX + fn.__name__
                if not hasattr(self, attr):
                    loaded_attr = fn(self)
                    setattr(self, attr, loaded_attr)
                else:
                    loaded_attr = getattr(self, attr)
            return loaded_attr
        __lazy_property.__doc__ = fn.__doc__
        return __lazy_property
    return _lazy_property


def _get_storage_format(instance, property_name, format_override=None):
    """
    Determine storage format using hierarchical approach:
    1. format_override (decorator parameter)
    2. instance._store_formats[property_name] (object-level, only when storage == 'formatted')
    3. instance.storage (general setting)
    4. 'memory' (final fallback)
    
    Parameters
    ----------
    instance : object
        The object instance
    property_name : str
        Name of the property
    format_override : str, optional
        Format specified in decorator
        
    Returns
    -------
    str
        Storage format to use
    """
    # 1. Decorator override has highest priority
    if format_override is not None:
        return format_override
    
    # Check if we should use formatted-specific formats
    use_formatted_formats = hasattr(instance, 'storage') and instance.storage == 'formatted'
    
    # 2. Object-level store formats (only when storage == 'formatted')
    if use_formatted_formats and hasattr(instance, '_store_formats'):
        if property_name in instance._store_formats:
            return instance._store_formats[property_name]
    
    # 3. General storage setting
    if hasattr(instance, 'storage'):
        return instance.storage
    
    # 4. Final fallback
    return 'memory'


def is_calculated(property, self, label='', format='formatted'):
    # TODO: remove this function
    attr = LAZY_PREFIX + property
    try:
        is_calculated = not getattr(self, attr) is None
    except AttributeError:
        is_calculated = False
    if not is_calculated:
        try:
            folder = get_folder_from_label(self, label)
            loaded_attr = load(property, folder, self, format=format)
            setattr(self, attr, loaded_attr)
            return not loaded_attr is None
        except (FileNotFoundError, OSError, KeyError):
            return False
    return is_calculated