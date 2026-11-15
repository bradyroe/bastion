"""Attack tests for security.py.

Every test here is a trick a prompt-injected model (or a malicious web page,
email or file it was asked to read) could try through a tool argument. Each
one must be refused. These run offline: no model, no microphone, no network.
"""
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config    # noqa: E402
import security  # noqa: E402
from security import SecurityError  # noqa: E402


# ── Files: the model may read inside the allowed folders and nowhere else ──

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    docs = tmp_path / "Documents"
    docs.mkdir()
    (docs / "notes.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "secrets.txt").write_text("api key", encoding="utf-8")
    monkeypatch.setattr(config, "ALLOWED_READ_DIRS", [docs])
    return docs


def test_a_file_inside_an_allowed_folder_is_readable(sandbox):
    assert security.safe_read_path(str(sandbox / "notes.txt")) == (sandbox / "notes.txt").resolve()


@pytest.mark.parametrize("trick", [
    "{docs}/../secrets.txt",                 # climb out with ..
    "{docs}/sub/../../secrets.txt",          # climb out through a fake subfolder
    "{root}/secrets.txt",                    # just name the file directly
])
def test_path_traversal_is_refused(sandbox, trick):
    raw = trick.format(docs=sandbox, root=sandbox.parent)
    with pytest.raises(SecurityError):
        security.safe_read_path(raw)


def test_a_sibling_folder_with_a_matching_prefix_is_refused(sandbox):
    """'Documents-evil' starts with the same letters as 'Documents'. A naive
    startswith() check would let it through; containment must be by path."""
    evil = sandbox.parent / "Documents-evil"
    evil.mkdir()
    (evil / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SecurityError):
        security.safe_read_path(str(evil / "x.txt"))


def test_a_non_string_path_is_refused(sandbox):
    with pytest.raises(SecurityError):
        security.safe_read_path(["Documents", "notes.txt"])


@pytest.mark.skipif(sys.platform != "win32", reason="UNC paths are a Windows thing")
def test_a_network_share_path_is_refused(sandbox):
    with pytest.raises(SecurityError):
        security.safe_read_path(r"\\attacker-host\share\payload.txt")


# ── URLs: no file://, no javascript:, and no reaching inside the network ──

@pytest.mark.parametrize("url", [
    "file:///C:/Windows/System32/config/SAM",
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "ftp://example.com/x",
    "http://",                                  # no host at all
])
def test_dangerous_url_schemes_are_refused(url):
    with pytest.raises(SecurityError):
        security.safe_url(url)


@pytest.mark.parametrize("ip", [
    "127.0.0.1",          # this machine (the HUD socket, local admin panels)
    "10.0.0.5",           # private network
    "192.168.1.1",        # the home router's admin page
    "169.254.169.254",    # cloud instance metadata, the classic SSRF prize
    "::1",                # loopback over IPv6
])
def test_server_side_fetches_cannot_reach_internal_addresses(monkeypatch, ip):
    """SSRF: a page the model was asked to summarise says 'now fetch
    http://169.254.169.254/...'. The host is resolved first and refused."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, port: [(family, 0, 0, "", (ip, 0))])
    with pytest.raises(SecurityError):
        security.safe_url("http://innocent-looking.example/", block_private=True)


def test_a_dns_name_that_resolves_inside_is_refused(monkeypatch):
    """DNS rebinding shape: the NAME looks public, the ADDRESS is internal."""
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, port: [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0)),
                                            (socket.AF_INET, 0, 0, "", ("127.0.0.1", 0))])
    with pytest.raises(SecurityError):
        security.safe_url("http://looks-public.example/", block_private=True)


def test_a_public_address_is_allowed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, port: [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))])
    assert security.safe_url("https://example.com/page", block_private=True)


# ── Text: control characters turn typing into a keystroke script ──

@pytest.mark.parametrize("payload", [
    "hello\nrm -rf /",          # a newline is a real Enter keypress when typed
    "ok\tthen",                 # a tab moves focus to the next button
    "\x1b[2J",                  # ESC starts an ANSI escape in logs and terminals
    "name\x00.txt",             # NUL truncates strings in C APIs
    "next\u0085line",           # C1 control: a newline in some APIs
])
def test_control_characters_are_refused_not_stripped(payload):
    with pytest.raises(SecurityError):
        security.safe_string(payload, 500, "text to type")


def test_multi_line_notes_may_keep_newlines_but_never_a_carriage_return_trick():
    assert security.safe_string("line one\r\nline two", 500, allow_newlines=True) == "line one\nline two"
    with pytest.raises(SecurityError):
        security.safe_string("fine\x1b[31m", 500, allow_newlines=True)


def test_overlong_input_is_refused():
    with pytest.raises(SecurityError):
        security.safe_string("a" * 501, 500)


# ── Apps: only named apps, and never a shell fragment ──

@pytest.mark.parametrize("name", [
    "notepad & calc",
    "notepad; del C:\\*",
    "notepad && powershell",
    "../../windows/system32/cmd",
    "powershell",                 # well-formed, but not on the allowlist
])
def test_only_allowlisted_app_names_launch(name):
    with pytest.raises(SecurityError):
        security.safe_app_name(name)


def test_an_allowlisted_app_is_accepted():
    name = sorted(config.ALLOWED_APPS)[0]
    assert security.safe_app_name(name.upper()) == name


# ── Keystrokes: never into a console, a dialog, or an unknown window ──

@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(config, "HANDS_ALLOWED_WINDOWS", {"notepad.exe"})
    monkeypatch.setattr(config, "HANDS_BLOCKED_PROCESSES", {"cmd.exe", "powershell.exe"})
    monkeypatch.setattr(config, "HANDS_BLOCKED_WINDOW_CLASSES", ("#32770", "consolewindowclass"))

    def focus(proc, cls):
        monkeypatch.setattr(security, "_foreground_window", lambda: (proc, cls))
    return focus


def test_typing_into_an_allowed_editor_is_permitted(windows):
    windows("notepad.exe", "Notepad")
    assert "notepad.exe" in security.safe_foreground_window()


@pytest.mark.parametrize("proc,cls,why", [
    ("cmd.exe", "ConsoleWindowClass", "typing into a console is running commands"),
    ("powershell.exe", "ConsoleWindowClass", "same, PowerShell"),
    ("notepad.exe", "#32770", "a dialog owned by an allowed app: Enter confirms it"),
    ("chrome.exe", "Chrome_WidgetWin_1", "not on the allowlist"),
    ("", "Notepad", "unidentifiable window: fail closed"),
])
def test_typing_is_refused_where_keystrokes_are_dangerous(windows, proc, cls, why):
    windows(proc, cls)
    with pytest.raises(SecurityError):
        security.safe_foreground_window()


def test_an_uppercase_process_name_cannot_dodge_the_blocklist(windows):
    windows("CMD.EXE", "ConsoleWindowClass")
    with pytest.raises(SecurityError):
        security.safe_foreground_window()


def test_when_focus_cannot_be_read_typing_is_refused(monkeypatch):
    def broken():
        raise SecurityError("the foreground window could not be identified")
    monkeypatch.setattr(security, "_foreground_window", broken)
    with pytest.raises(SecurityError):
        security.safe_foreground_window()
