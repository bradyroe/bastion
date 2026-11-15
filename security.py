"""
Security helpers — input validation, path containment, sanitisation.

All untrusted input (tool arguments from the LLM, user-typed strings) must
pass through one of these functions before touching the filesystem or shell.
Raises SecurityError — callers convert this to a plain-string tool result so
Bastion can communicate the refusal naturally rather than crashing.
"""

import ipaddress
import re
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import config


class SecurityError(ValueError):
    """Raised when a tool argument fails a security check."""


def safe_url(raw: str, *, block_private: bool = False) -> str:
    """
    Validate a URL before opening or fetching it.

      - Only http/https schemes (blocks file://, javascript:, data:, etc.).
      - When block_private=True (for server-side fetches), resolve the host and
        reject loopback/private/link-local IPs. This prevents the LLM from
        coaxing Bastion into hitting internal services (SSRF), e.g. the HUD
        WebSocket, a router admin page, or cloud metadata endpoints.
    """
    raw = safe_string(raw, 2048, "url")
    parsed = urlparse(raw)

    if parsed.scheme not in ("http", "https"):
        raise SecurityError(
            f"Only http/https URLs are allowed (got scheme {parsed.scheme!r})."
        )
    if not parsed.hostname:
        raise SecurityError(f"URL has no host: {raw!r}")

    if block_private:
        try:
            infos = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as exc:
            raise SecurityError(f"Could not resolve host: {parsed.hostname}") from exc
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise SecurityError(
                    f"Refusing to fetch a private/internal address: {ip}"
                )

    return raw


def safe_read_path(raw: str) -> Path:
    """
    Resolve a path and verify it sits inside an ALLOWED_READ_DIR.

    Path.resolve() collapses '..' traversal before we check containment,
    so 'Documents/../../../Windows/System32' is caught correctly.
    """
    if not isinstance(raw, str):
        raise SecurityError(f"Path must be a string, got {type(raw).__name__}")

    try:
        resolved = Path(raw).resolve()
    except (TypeError, ValueError) as exc:
        raise SecurityError(f"Invalid path: {raw!r}") from exc

    for allowed in config.ALLOWED_READ_DIRS:
        try:
            resolved.relative_to(allowed.resolve())
            return resolved
        except ValueError:
            continue

    allowed_str = [str(d) for d in config.ALLOWED_READ_DIRS]
    raise SecurityError(
        f"Path '{resolved}' is outside allowed directories. "
        f"Allowed roots: {allowed_str}"
    )


# C0 controls (including NUL), DEL, and the C1 block. Interior control
# characters are not cosmetic: pynput turns \n and \t into real Enter and Tab
# presses, so one type_text call would otherwise be a multi-line keystroke
# script; \n inside a stored memory value forges extra rows in recall's joined
# output; and ESC reaches the log file and the terminal as an ANSI sequence.
_CONTROL_CHARS = frozenset(
    [chr(c) for c in range(0x00, 0x20)] + [chr(c) for c in range(0x7F, 0xA0)]
)


def safe_string(value: str, max_len: int, field: str = "value",
                *, allow_newlines: bool = False) -> str:
    """
    Strip outer whitespace, reject control characters, enforce a maximum length.

    Control characters are refused rather than stripped, so a caller sees an
    explicit failure instead of silently acting on altered text.

    allow_newlines=True permits \\n and \\t only — for the two callers holding
    genuinely multi-line user text (a remembered note, a clipboard write). Line
    endings are normalised to \\n so \\r cannot smuggle in a second form. Never
    pass it on a path that reaches the keyboard.
    """
    if not isinstance(value, str):
        raise SecurityError(f"{field} must be a string, got {type(value).__name__}")

    cleaned = value.strip()

    if allow_newlines:
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        forbidden = _CONTROL_CHARS - {"\n", "\t"}
    else:
        forbidden = _CONTROL_CHARS

    found = sorted({c for c in cleaned if c in forbidden})
    if found:
        raise SecurityError(
            f"{field} contains control characters "
            f"({', '.join(f'0x{ord(c):02x}' for c in found)}), which are refused."
        )

    if len(cleaned) > max_len:
        raise SecurityError(
            f"{field} is too long ({len(cleaned)} chars, limit {max_len})"
        )

    return cleaned


def safe_app_name(raw: str) -> str:
    """
    Validate an app name:
      1. Must be alphanumeric + hyphens/underscores (no shell metacharacters).
      2. Must exist in the ALLOWED_APPS whitelist.

    Even if step 2 somehow passed a malicious name, step 1 ensures it can
    never be interpreted as a shell command fragment.
    """
    name = safe_string(raw, 64, "app name").lower()

    if not re.fullmatch(r"[a-z0-9_\-]+", name):
        raise SecurityError(
            f"App name contains invalid characters: {name!r}. "
            f"Only letters, digits, hyphens, and underscores are allowed."
        )

    if name not in config.ALLOWED_APPS:
        raise SecurityError(
            f"App '{name}' is not in the allowed list. "
            f"Allowed: {sorted(config.ALLOWED_APPS.keys())}"
        )

    return name


def _foreground_window() -> tuple[str, str]:
    """
    Return (process exe basename lowercased, window class) of the window that
    currently has focus, via ctypes — no extra dependency.

    Raises SecurityError if any step fails. Every caller is about to synthesise
    keystrokes, so "I don't know what has focus" must mean "don't send them".
    """
    if sys.platform != "win32":
        raise SecurityError(
            f"the foreground window can't be checked on {sys.platform}"
        )

    import ctypes
    from ctypes import wintypes

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetForegroundWindow.argtypes = []
        user32.GetClassNameW.restype = ctypes.c_int
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND,
                                                    ctypes.POINTER(wintypes.DWORD)]

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            raise SecurityError("there is no foreground window")

        buf = ctypes.create_unicode_buffer(256)
        if user32.GetClassNameW(hwnd, buf, len(buf)) == 0:
            raise SecurityError("the foreground window's class name is unreadable")
        window_class = buf.value.strip()
        if not window_class:
            raise SecurityError("the foreground window has an empty class name")

        pid = wintypes.DWORD(0)
        if user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)) == 0 or not pid.value:
            raise SecurityError("the foreground window's owning process is unknown")

        import psutil
        proc_name = psutil.Process(pid.value).name().strip().lower()
        if not proc_name:
            raise SecurityError("the foreground process has no name")

    except SecurityError:
        raise
    except Exception as exc:  # OSError, psutil.NoSuchProcess, anything else
        raise SecurityError(
            f"the foreground window could not be identified ({type(exc).__name__}: {exc})"
        ) from exc

    return proc_name, window_class


def safe_foreground_window(action: str = "send keystrokes") -> str:
    """
    Verify the focused window may receive synthesised keystrokes.

    Order matters: identify the window (failing closed if we can't), refuse the
    always-dangerous classes and processes, then require the allowlist. The
    class/process blocks sit ON TOP of the allowlist, not instead of it — a
    dialog owned by an allowlisted app is still a dialog.

    Returns a short "name [class]" description for logging. Raises SecurityError
    with a message naming the window, so the refusal tells the owner what to add.
    """
    # Normalise here rather than trusting the resolver to have done it. Today
    # there is one caller and it lowercases; this function must still be correct
    # if a second one appears, since a stray "Notepad.EXE" would otherwise miss
    # the blocked-process check entirely.
    proc_name, window_class = _foreground_window()
    proc_name = (proc_name or "").strip().lower()
    window_class = (window_class or "").strip()
    where = f"{proc_name} [{window_class}]"

    # Belt and braces: _foreground_window already rejects these, but a blank
    # name would match no blocked substring, so never let one reach the checks.
    if not proc_name.strip() or not window_class.strip():
        raise SecurityError(
            f"Refusing to {action}: the focused window could not be identified "
            f"({where})."
        )

    cls_lower = window_class.lower()

    for blocked in config.HANDS_BLOCKED_WINDOW_CLASSES:
        if blocked in cls_lower:
            raise SecurityError(
                f"Refusing to {action}: the focused window is {where}, which is a "
                f"dialog or console ({blocked}). Keystrokes there can execute "
                f"commands or confirm irreversible actions."
            )

    if proc_name in config.HANDS_BLOCKED_PROCESSES:
        raise SecurityError(
            f"Refusing to {action}: the focused window is {where}, a console. "
            f"Keystrokes there are command execution."
        )

    if proc_name not in config.HANDS_ALLOWED_WINDOWS:
        raise SecurityError(
            f"Refusing to {action}: the focused window is {where}, which isn't in "
            f"HANDS_ALLOWED_WINDOWS. Allowed: "
            f"{sorted(config.HANDS_ALLOWED_WINDOWS) or ['(none)']}. "
            f"Focus an allowed window, or add this one in config.py."
        )

    return where
