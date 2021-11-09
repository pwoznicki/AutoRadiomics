import importlib.util

# from https://github1s.com/jrieke/traingenerator/blob/HEAD/app/utils.py#L1-L177
def import_from_file(module_name: str, filepath: str):
    """
    Imports a module from file.

    Args:
        module_name (str): Assigned to the module's __name__ parameter (does not
            influence how the module is named outside of this function)
        filepath (str): Path to the .py file

    Returns:
        The module
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
