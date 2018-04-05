def get_var_name(obj) -> str:
    """Get literal name of the given variable.

    Adapted from https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string

    Parameters
    ----------
    obj: object
        Variable whose name you want to get.

    Returns
    -------
    name: string
        Name of the given variable.
    """
    names = [objname for objname, oid in globals().items() if id(oid) == id(obj)]
    name = names[0] if len(names) > 0 else None
    return name
