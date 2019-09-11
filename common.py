class Toolkit:
    def __init__(self, glb=None):
        self.glb = glb if glb is not None else globals

    def set_global(self, glb):
        self.glb = glb

    def get_var_name(self, obj) -> str:
        names = [objname for objname, oid in self.glb().items() if id(oid) == id(obj)]
        name = names[0] if len(names) > 0 else None
        return name

    def batch_print_shape(self, *args):
        print("Shape")
        print("-----")
        for arg in args:
            print("{}: {}".format(self.get_var_name(arg), arg.shape))
        print()
