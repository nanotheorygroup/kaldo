import numpy as np
import os
import h5py
from typing import Any, Callable, Optional, Union
from kaldo.helpers.logger import get_logger
logging = get_logger()


LAZY_PREFIX = '_lazy__'
FOLDER_NAME = 'data'


class Storable:
    """
    Mixin class that provides storage functionality to classes using lazy_property.
    Classes should inherit from this and define their own _store_formats.
    """
    
    def _load_property(self, property_name, folder, format='formatted'):
        """
        Load a property from storage. Override this method in subclasses for 
        property-specific loading behavior.
        
        Parameters
        ----------
        property_name : str
            Name of the property to load
        folder : str
            Folder path where the property is stored
        format : str
            Storage format ('numpy', 'hdf5', 'formatted', 'memory')
            
        Returns
        -------
        loaded_data : any
            The loaded property data
        """
        name = folder + '/' + property_name
        
        if format == 'numpy':
            return np.load(name + '.npy', allow_pickle=True)
            
        elif format == 'hdf5':
            with h5py.File(name.split('/')[0] + '.hdf5', 'r') as storage:
                loaded = storage[name]
                return loaded[()]
                
        elif format == 'formatted':
            return self._load_formatted_property(property_name, name)
            
        elif format == 'memory':
            if hasattr(self, LAZY_PREFIX + property_name):
                logging.info('Loading from memory ' + str(property_name) + ' property.')
                return getattr(self, LAZY_PREFIX + property_name)
            else:
                logging.info(str(property_name) + ' not found.')
                raise KeyError('Property not found')
        else:
            raise ValueError(f'Storage format {format} not implemented')
    
    def _load_formatted_property(self, property_name, name):
        """
        Load a property from formatted files. Override in subclasses for 
        property-specific formatted loading.
        
        Parameters
        ----------
        property_name : str
            Name of the property
        name : str
            Full file path without extension
            
        Returns
        -------
        loaded_data : any
            The loaded property data
        """
        # Default: single file with float dtype
        if property_name == 'diffusivity':
            dt = complex
        else:
            dt = float
        return np.loadtxt(name + '.dat', skiprows=1, dtype=dt)
    
    def _save_property(self, property_name, folder, data, format='formatted'):
        """
        Save a property to storage. Override this method in subclasses for 
        property-specific saving behavior.
        
        Parameters
        ----------
        property_name : str
            Name of the property to save
        folder : str
            Folder path where to save the property
        data : any
            The property data to save
        format : str
            Storage format ('numpy', 'hdf5', 'formatted', 'memory')
        """
        name = folder + '/' + property_name
        
        if format == 'numpy':
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.save(name + '.npy', data)
            logging.info(name + ' stored')
            
        elif format == 'hdf5':
            with h5py.File(name.split('/')[0] + '.hdf5', 'a') as storage:
                if not name in storage:
                    storage.create_dataset(name, data=data, chunks=True, compression='gzip', compression_opts=9)
            logging.info(name + ' stored')
            
        elif format == 'formatted':
            if not os.path.exists(folder):
                os.makedirs(folder)
            self._save_formatted_property(property_name, name, data)
            
        elif format == 'memory':
            logging.warning('Property ' + str(property_name) + ' will be lost when calculation is over.')
        else:
            raise ValueError(f'Storage format {format} not implemented')
    
    def _save_formatted_property(self, property_name, name, data):
        """
        Save a property to formatted files. Override in subclasses for 
        property-specific formatted saving.
        
        Parameters
        ----------
        property_name : str
            Name of the property
        name : str
            Full file path without extension
        data : any
            The property data to save
        """
        # Default: single file with scientific notation
        fmt = '%.18e'
        np.savetxt(name + '.dat', data, fmt=fmt, header=str(data.shape))
    
    def get_folder_from_label(self, label='', base_folder=None):
        """
        Generate folder path from label template and instance attributes.
        """
        if base_folder is None:
            if hasattr(self, 'folder') and self.folder:
                base_folder = self.folder
            else:
                base_folder = FOLDER_NAME
        
        # Add k-point or q-point structure
        base_folder = self._add_grid_components(base_folder)
        
        # Add label-specific components
        if label != '':
            components = self._get_folder_path_components(label)
            for component in components:
                if component:  # Only add non-empty components
                    base_folder += '/' + component
        
        return base_folder
    
    def _add_grid_components(self, base_folder):
        """Add k-point grid or q-point components to folder path."""
        # Check if this is a multi-kpoint system
        if hasattr(self, 'kpts') and np.prod(self.kpts) > 1:
            kpts = self.kpts
            base_folder += '/' + str(kpts[0]) + '_' + str(kpts[1]) + '_' + str(kpts[2])
        # Otherwise check if it's a single q-point system
        elif hasattr(self, 'q_point'):
            q_point = self.q_point
            base_folder += '/single_q/' + str(q_point[0]) + '_' + str(q_point[1]) + '_' + str(q_point[2])
        return base_folder
    
    def _get_folder_path_components(self, label):
        """
        Get folder path components based on label template.
        Override in subclasses to provide class-specific components.
        
        Parameters
        ----------
        label : str
            Label template with placeholders like '<temperature>'
            
        Returns
        -------
        list of str
            List of folder path components to append
        """
        return []



def parse_pair(txt):
    return complex(txt.strip("()"))





def lazy_property(label: str = '', format: Optional[str] = None) -> Callable[[Callable], property]:
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
    def _lazy_property(fn: Callable) -> property:
        @property
        def __lazy_property(self) -> Any:
            # Determine storage format using hierarchy
            storage_format = _get_storage_format(self, fn.__name__, format)
            
            if storage_format != 'memory':
                folder = self.get_folder_from_label(label)
                property_name = fn.__name__
                try:
                    loaded_attr = self._load_property(property_name, folder, format=storage_format)
                    logging.info('Loading ' + folder + '/' + str(property_name))
                except (FileNotFoundError, OSError, KeyError):
                    logging.info(folder + '/' + str(property_name) + ' not found in ' + storage_format + ' format, calculating ' + str(fn.__name__))
                    loaded_attr = fn(self)
                    self._save_property(property_name, folder, loaded_attr, format=storage_format)
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


def is_calculated(property_name, instance, label='', format='formatted'):
    """
    Check if a property has been calculated and cached, or can be loaded from storage.
    
    Parameters
    ----------
    property_name : str
        Name of the property to check
    instance : Storable
        Instance that should have the property
    label : str
        Label for folder generation
    format : str
        Storage format to check
        
    Returns
    -------
    bool
        True if property is calculated/available, False otherwise
    """
    # Check if already cached in memory
    attr = LAZY_PREFIX + property_name
    try:
        is_calculated_val = not getattr(instance, attr) is None
        if is_calculated_val:
            return True
    except AttributeError:
        pass
    
    # Try to load from storage if instance is a StorageMixin
    if hasattr(instance, '_load_property') and hasattr(instance, 'get_folder_from_label'):
        try:
            folder = instance.get_folder_from_label(label)
            loaded_attr = instance._load_property(property_name, folder, format=format)
            setattr(instance, attr, loaded_attr)
            return loaded_attr is not None
        except (FileNotFoundError, OSError, KeyError):
            return False
    
    return False