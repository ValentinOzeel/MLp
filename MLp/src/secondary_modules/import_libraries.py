import importlib

from MLp.conf.config_functions import get_config
mlp_config = get_config()
RANDOM_STATE = mlp_config['MLP_CONFIG']['RANDOM_STATE']
SKLEARN_CPU = mlp_config['MLP_CONFIG']['SKLEARN_CPU']
SKLEARN_CPU_INTELEX_ACCEL = mlp_config['MLP_CONFIG']['SKLEARN_CPU_INTELEX_ACCEL']
SKLEARN_GPU_RAPIDS = mlp_config['MLP_CONFIG']['SKLEARN_GPU_RAPIDS']
PANDAS_GPU = mlp_config['MLP_CONFIG']['PANDAS_GPU']


def import_cpu_gpu_sklearn(sub_module, import_class):
    ''' 
    Flexible import mechanism that tries to import a specified module or class from either 'sklearn', 'sklearnex' or 'cuml'
    based on the SKLEARN_CPU, SKLEARN_CPU_INTELEX_ACCEL, SKLEARN_GPU_RAPIDS flags. If import is not available (e.g algos not avaialble in sklearnex for instance),
    then we fall back to classical sklearn import.
    
        Parameters:
        - sub_module: sub_module to import (e.g preprocessing in -- from sklearn.preprocessing import OneHotEncoder)
        - import_class: class to import    (e.g OneHotEncoder in -- from sklearn.preprocessing import OneHotEncoder)
        - module (first args of the nested actual_imports function): 'sklearn', 'sklearnex' or 'cuml'

        Returns:
        - imported: The class or the list of classes imported
        
    '''
    def _actual_imports(module, sub_module, import_class):
        module_potential_sub_mod = f'{module}' if not sub_module else f'{module}.{sub_module}'
        try:
            module_and_sub = importlib.import_module(module_potential_sub_mod)
            imported_class = getattr(module_and_sub, import_class)
            return imported_class
        except (AttributeError, ModuleNotFoundError) as e:
            #print(f"Could not import {module}.{sub_module}.{import_class} Error: {e}. Falling back to classical sklearn import.")
            module_potential_sub_mod = 'sklearn' if not sub_module else f'sklearn.{sub_module}'
            module_and_sub = importlib.import_module(module_potential_sub_mod)
            imported_class = getattr(module_and_sub, import_class)
            return imported_class  
    
    module_map = {
        "sklearn": SKLEARN_CPU,
        "sklearnex": SKLEARN_CPU_INTELEX_ACCEL,
        "cuml": SKLEARN_GPU_RAPIDS
    }

    for module, flag in module_map.items():
        if flag:       
            if isinstance(import_class, list):
                imported = []
                imported 
                for wanted_class in import_class:
                    imported.append(_actual_imports(module, sub_module, wanted_class))
            else:
                imported = _actual_imports(module, sub_module, import_class)
            return imported

    raise ImportError("No valid module found. Check the flag configurations.")


def import_cpu_gpu_pandas():
    if PANDAS_GPU:
#        import cudf as pd
        print('Should import cudf\n')
    else:
        import pandas as pd
    return pd


def sklearnex_patch():
    from sklearnex import patch_sklearn
    return patch_sklearn