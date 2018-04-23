import pandas as pd
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_files(f_dict, mode="w"):
    """Context Manager to handle the opening and closing of a group files.

    Why use this? At least for me, using a `for` loop to open a bunch of files at the
    beginning of a piece of codes and another one to close them looks inelegant.
    On the other hand, using a single `with` line makes instead of two loops makes
    the main purpose of the the other codes more clear.

    Parameters
    ----------
    f_dict: dict
        Dictionary whose values are file paths to open. The keys can be any string
        in your convenience.

    mode: { "w", "wb", "a", "r", "rb" }
        I think any programmer knows it.

    Examples
    --------
    >>> f_dict = {i: "{}.txt".format(i) for i in range(10)}
    >>> with open_files(f_dict) as fs:
    ...     # do something
    """
    fs = {k: open(v, mode) for k, v in f_dict.items()}
    yield fs
    for k, f in fs.items():
        f.close()


def batch_load_csv(folder, *files):
    input_dir = Path(folder)
    return (pd.read_csv(input_dir / f) for f in files)


class FileWriter:
    """Writer to write file with buffer

    Buffer mechanism is introduced since too frequency I/O is expensive.

    Parameters
    ----------
    f: Writable File Object
        Opened file object to write. Remember that it must be opened before passed
        to the `FileWriter` constructor and you must close it by yourself after use.
        `FileWriter` is not responsible for the opening and closing.

    Attributes
    ----------
    buffer: string
        Buffer of characters waiting to written to the file object.

    Examples
    --------
    >>> f = open("test.txt", "w")
    >>> fw = FileWriter(f)
    >>> for i in range(100):
    ...     fw.write_buffer(str(i))
    >>> fw.flush()
    >>> f.close()  # Don't forget to close it and the fact that, once you close it, you should not use `fw` anymore
    """
    def __init__(self, f):
        self.f = f
        self.buffer = ""

    def write_buffer(self, chars):
        self.buffer += chars

    def clear_buffer(self):
        self.buffer = ""

    def flush(self):
        self.f.write(self.buffer)
        self.clear_buffer()


class FileWriterGroup:
    """Group of writers to write files with buffer

    Buffer mechanism is introduced since too frequency I/O is expensive.

    Parameters
    ----------
    f_dict: Dictionary whose values are opened File objects to write. The keys can be
        any string in your convenience. Remember that they must be opened before passed
        to the `FileWriterGroup` constructor and you must close them by yourself after use.
        `FileWriterGroup` is not responsible for the opening and closing.

    Attributes
    ----------
    writers: dict
        Dictionary than maps keys of files you passed to the constructor to the
        corresponding `FileWriter` instances.

    n_writers: int
        Number of `FileWriter`/files in the group.

    Examples
    --------
    >>> f_dict = {i: "{}.txt".format(i) for i in range(10)}
    >>> with open_files(f_dict) as fs:
    ...     fwg = FileWriterGroup(fs)
    ...     fwg.write_buffer("Hey there")
    ...     # anything else ...
    ...     fwg.flush()
    """
    def __init__(self, f_dict):
        self.writers = {k: FileWriter(v) for k, v in f_dict.items()}
        self.n_writers = len(f_dict)

    def write_buffer(self, name, chars):
        self.writers[name].write_buffer(chars)

    def clear_buffer(self, name):
        self.writers[name].clear_buffer()

    def clear_buffers(self):
        for name, writer in self.writers.items():
            writer.clear_buffer()

    def flush(self):
        for name, writer in self.writers.items():
            writer.flush()
