import psutil
import os


def format_secs(secs, decimal=1) -> str:
    """Get formatted literal string for a time duration.

    Parameters
    ----------
    secs: int | float
        Number of seconds in the time duration.

    decimal: int
        Number of decimal places to round to in the literal string.

    Returns
    -------
    ret: Formatted literal string for the time duration.
    """
    assert isinstance(secs, int) or isinstance(secs, float)
    assert isinstance(decimal, int)
    if secs <= 60:
        ret = "{} seconds".format(round(secs, decimal))
    elif secs <= 60 * 60:
        ret = "{} minutes".format(round(secs/60.0, decimal))
    elif secs <= 60 * 60 * 24:
        ret = "{} hours".format(round(secs/(60 * 60), decimal))
    else:
        ret = "{} days".format(round(secs/(60 * 60 * 24), decimal))
    return ret


def format_memory(num_units, decimal=2, units=None) -> str:
    """Get formatted literal string for a memory size.

    Parameters
    ----------
    num_units: int | float
        Number of memory units(e.g. Bytes, KB).

    decimal: int
        Number of decimal places to round to in the literal string.

    units: {"B", "KB", "MB", "GB", "TB"}
        Unit of memory.

    Returns
    -------
    ret: str
        Formatted literal string for the memory size.
    """
    assert num_units >= 0
    assert isinstance(num_units, int) or isinstance(num_units, float)
    assert isinstance(decimal, int)
    units = ['B', 'KB', 'MB', 'GB', "TB"] if units is None else units
    if num_units < 1024:
        ret = "{}{}".format(round(num_units, decimal), units[0])
    else:
        ret = format_memory(num_units / 1024.0, decimal, units[1:])
    return ret


def format_memory_diff(num_bytes) -> str:
    """Get formatted literal string for a difference in memory size.

    Parameters
    ----------
    num_bytes: int
        Number of bytes in the memory difference.

    Returns
    -------
    ret: string
        Formatted literal string for the memory difference.
    """
    assert isinstance(num_bytes, int)
    sign = '+' if num_bytes >= 0 else '-'
    ret = "{}{}".format(sign, format_memory(abs(num_bytes)))
    return ret


def get_memory_bytes() -> int:
    """Get the memory usage in bytes of the current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_memory_str() -> str:
    """Get formatted literal string for the memory usage of the current process."""
    return format_memory(get_memory_bytes())
