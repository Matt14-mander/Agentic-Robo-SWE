"""Cheap source polling plus stable content reads and explicit write notifications."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Callable, Iterable
from weakref import WeakSet

_DETECTORS: WeakSet[SourceDetector] = WeakSet()
_NOTIFY_LOCK = RLock()


def notify_source_changed(path: Path) -> None:
    with _NOTIFY_LOCK:
        for detector in list(_DETECTORS):
            try:
                detector.dirty.add(path.relative_to(detector.root).as_posix())
            except ValueError:
                pass


@lru_cache(maxsize=1)
def _windows_api():
    from ctypes import wintypes

    class BasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in
                    ("creation", "access", "write", "change")] + [("attributes", wintypes.DWORD)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel, BasicInfo


def _windows_change_time(path: Path) -> int | None:
    """Read ChangeTime (not creation time); unsupported filesystems force hashing."""
    kernel, BasicInfo = _windows_api()
    handle = kernel.CreateFileW(str(path), 0x80, 7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        return None
    try:
        value = BasicInfo()
        if not kernel.GetFileInformationByHandleEx(handle, 0, ctypes.byref(value), ctypes.sizeof(value)):
            return None
        return int(value.change) or None
    finally:
        kernel.CloseHandle(handle)


class SourceDetector:
    def __init__(self, root: Path, resolve: Callable[[str], Path],
                 read: Callable[[str], tuple[Path, str]], *, strict: bool = False) -> None:
        self.root, self.resolve, self.read_source = root, resolve, read
        self.strict = strict
        self.stamps: dict[str, tuple | None] = {}
        self.dirty: set[str] = set()
        self.metadata_checks = self.content_reads = self.bytes_read = 0
        with _NOTIFY_LOCK:
            _DETECTORS.add(self)

    def fingerprint(self, relative: str) -> tuple | None:
        self.metadata_checks += 1
        path = self.resolve(relative)
        try:
            info = path.stat()
        except FileNotFoundError:
            return ("missing",)
        change = _windows_change_time(path) if os.name == "nt" else info.st_ctime_ns
        if change is None:
            return None  # No reliable change token: read/hash on every query.
        return (info.st_size, info.st_mtime_ns, change, info.st_dev, info.st_ino, info.st_mode)

    def changed(self, paths: Iterable[str]) -> set[str]:
        changed = set()
        for relative in paths:
            stamp = self.fingerprint(relative)
            if (self.strict or stamp is None or relative in self.dirty
                    or relative not in self.stamps or self.stamps[relative] != stamp):
                changed.add(relative)
        return changed

    def read_stable(self, relative: str) -> tuple[Path, str, tuple | None]:
        for _ in range(3):
            before = self.fingerprint(relative)
            path, source = self.read_source(relative)
            self.content_reads += 1
            self.bytes_read += len(source.encode("utf-8"))
            after = self.fingerprint(relative)
            if before == after:
                return path, source, after
        raise RuntimeError(f"Source changed repeatedly while reading: {relative}")

    def accept(self, relative: str, stamp: tuple | None) -> None:
        self.stamps[relative] = stamp
        with _NOTIFY_LOCK:
            self.dirty.discard(relative)
