"""Win32 queries used only when cleaning local Windows artifacts."""
import ctypes as ct
from ctypes import wintypes as w
from functools import lru_cache


class Stream(ct.Structure):
    _fields_ = [("size", ct.c_int64), ("name", w.WCHAR * 296)]


class Standard(ct.Structure):
    _fields_ = [("allocation", ct.c_int64), ("end", ct.c_int64),
                ("links", w.DWORD), ("pending", ct.c_byte), ("directory", ct.c_byte)]


@lru_cache(maxsize=1)
def api():
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    adv = ct.WinDLL("advapi32", use_last_error=True)
    specs = [
        (kernel.FindFirstStreamW, w.HANDLE, [w.LPCWSTR, ct.c_int, ct.POINTER(Stream), w.DWORD]),
        (kernel.FindNextStreamW, w.BOOL, [w.HANDLE, ct.POINTER(Stream)]),
        (kernel.FindClose, w.BOOL, [w.HANDLE]),
        (kernel.GetFileInformationByHandleEx, w.BOOL, [w.HANDLE, ct.c_int, w.LPVOID, w.DWORD]),
        (adv.GetFileSecurityW, w.BOOL, [w.LPCWSTR, w.DWORD, w.LPVOID, w.DWORD, ct.POINTER(w.DWORD)]),
    ]
    for fn, result, args in specs:
        fn.argtypes, fn.restype = args, result
    return kernel, adv


def security_key(path):
    kernel, adv = api()
    raw = str(path.absolute())
    name = "\\\\?\\UNC\\" + raw[2:] if raw.startswith("\\\\") else "\\\\?\\" + raw
    data = Stream()
    handle = kernel.FindFirstStreamW(name, 0, ct.byref(data), 0)
    if handle == ct.c_void_p(-1).value:
        raise ct.WinError(ct.get_last_error())
    try:
        if data.name != "::$DATA" or kernel.FindNextStreamW(handle, ct.byref(data)):
            raise ValueError("檔案有其他 NTFS 資料流")
        if ct.get_last_error() != 38:  # ERROR_HANDLE_EOF
            raise ct.WinError(ct.get_last_error())
    finally:
        kernel.FindClose(handle)
    size = w.DWORD()
    flags = 0x17  # OWNER | GROUP | DACL | mandatory integrity label
    adv.GetFileSecurityW(name, flags, None, 0, ct.byref(size))
    if not size.value:
        raise ct.WinError(ct.get_last_error())
    buf = ct.create_string_buffer(size.value)
    if not adv.GetFileSecurityW(name, flags, buf, size, ct.byref(size)):
        raise ct.WinError(ct.get_last_error())
    return buf.raw[:size.value]


def allocation(path):
    import msvcrt
    kernel, _ = api()
    row = Standard()
    with path.open("rb") as stream:
        if not kernel.GetFileInformationByHandleEx(msvcrt.get_osfhandle(stream.fileno()), 1,
                                                  ct.byref(row), ct.sizeof(row)):
            raise ct.WinError(ct.get_last_error())
    return row.allocation
