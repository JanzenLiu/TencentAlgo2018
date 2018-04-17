import types


glbs_string = "" \
              "def globals_wrapper():\n" \
              " def f():\n" \
              "     return globals()\n" \
              " return f"


def globals_wrapper():
    def f():
        return None
    return f


class InfoUtils:
    """Helper to handle common information

    Parameters
    ----------
    glbs: types.FunctionType
        Function that returns a global variable dictionary that maps variable names
        to their ids in the memory. If you are using this class by directly importing,
        it's necessary to set this parameter, or the instance will be instantiated with
        the its own `globals` function, i.e. the `globals` function in this module,
        therefore will not contain the variable table of the module where you call this.

    Examples
    --------
    >>> # from info_utils import InfoUtils, glbs_string, globals_wrapper  # uncomment this during actual use
    >>> exec(glbs_string)  # This's the best way I figured out to fix globals problem. fxxk globals
    >>> iu = InfoUtils(globals_wrapper())
    """
    def __init__(self, glbs):
        self.glbs = glbs

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
        names = [objname for objname, oid in self.glbs().items() if id(oid) == id(obj)]
        name = names[0] if len(names) > 0 else None
        return name
