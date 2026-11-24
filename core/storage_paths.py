"""Physical-file and project-relative path checks shared by storage owners."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat


def plain(info):
    return (stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def clean_path(path):
    return Path(path).absolute() == Path(path).resolve(strict=False)


def relative_path(value: str) -> Path:
    """Project-relative paths only, including when inspected on another OS."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or ":" in value:
        raise ValueError("工件路徑必須是使用 / 的專案相對路徑")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("工件路徑不允許絕對路徑或父目錄")
    return Path(*path.parts)
