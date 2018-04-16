import types


class InfoUtils:
    """Helper to handle common information

    Parameters
    ----------
    glb: types.FunctionType
        Function that returns a global variable dictionary that maps variable names
        to their ids in the memory. If you are using this class by directly importing,
        it's necessary to set this parameter, or the instance will be instantiated with
        the its own `globals` function, i.e. the `globals` function in this module,
        therefore will not contain the variable table of the module where you call this.

    Examples
    --------
    >>> iu = InfoUtils(globals)
    """
    def __init__(self, glb):
        self.glb = glb

    def get_var_name(self, obj) -> str:
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
        names = [objname for objname, oid in self.globals().items() if id(oid) == id(obj)]
        name = names[0] if len(names) > 0 else None
        return name
