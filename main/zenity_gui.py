import os
import sys
import shutil
import subprocess
from typing import Tuple

from . import state
from .utils import utfprint


def zenity(filetypes=None, initialdir="", initialfile="", multiple=False, **kwargs) -> Tuple[int, object]:
    if not state.zenity_permitted:
        raise Exception("Zenity disabled, attempting to use TK GUI.")
    if sys.platform != "linux":
        raise Exception("Zenity GUI is only usable on Linux, attempting to use TK GUI.")
    zenity_bin = shutil.which("yad")
    using_yad = True
    if not zenity_bin:
        zenity_bin = shutil.which("zenity")
        using_yad = False
    if not zenity_bin:
        using_yad = False
        raise Exception("Zenity not present, falling back to TK GUI.")

    def zenity_clean(txt: str):
        return txt.replace("\\", "\\\\").replace("$", "\\$").replace("!", "\\!").replace("*", "\\*")\
        .replace("?", "\\?").replace("&", "&amp;").replace("|", "&#124;").replace("<", "&lt;").replace(">", "&gt;")\
        .replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]").replace("{", "\\{").replace("}", "\\}")

    def zenity_sanity_check(zenity_bin):  # make sure zenity is sane
        try:  # Run `zenity --help` and pipe to grep
            sc_clean_env = os.environ.copy()
            sc_clean_env.pop("LD_LIBRARY_PATH", None)
            sc_clean_env["PATH"] = "/usr/bin:/bin"
            scargs = ['/usr/bin/env', zenity_bin, '--help']
            result = subprocess.run(scargs, env=sc_clean_env, capture_output=True, text=True, encoding="utf-8", timeout=10)

            if result.returncode == 0 and "--file" in result.stdout:
                return True
            else:
                utfprint(f"Zenity/YAD sanity check failed - ReturnCode={result.returncode}", 0)
                return False
        except FileNotFoundError:
            utfprint(f"Zenity/YAD sanity check failed - {zenity_bin} not found", 0)
            return False

    if not zenity_sanity_check(zenity_bin):
        raise Exception("Zenity not working correctly, falling back to TK GUI.")

    # Build args based on keywords
    args = ['/usr/bin/env', zenity_bin, ('--file' if using_yad else '--file-selection')]
    for k, v in kwargs.items():
        if v is True:
            args.append(f'--{k.replace("_", "-").strip("-")}')
        elif isinstance(v, str):
            cv = zenity_clean(v) if k != "title" else v
            args.append(f'--{k.replace("_", "-").strip("-")}={cv}')

    # Build filetypes specially if specified
    if filetypes:
        for name, globs in filetypes:
            if name:
                globlist = globs.split()
                args.append(f'--file-filter={name.replace("|", "")} ({", ".join(t for t in globlist)})|{globs}')

    # Default filename and folder
    if initialdir is None:
        initialdir = state.zenity_recent_dir
    if initialfile is None:
        initialfile = ""
    initialpath = os.path.join(initialdir, initialfile)
    args.append(f'--filename={initialpath}')

    if multiple:
        args.append("--multiple")
        args.append("--separator=|")

    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    clean_env["PATH"] = "/usr/bin:/bin"

    procres = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=clean_env,
        check=False
    )
    result = procres.stdout.decode('utf-8').strip()
    if procres.returncode == 0 and result:
        directory = result
        if multiple:
            result = tuple(result.split("|"))
            directory = result[0]
        if not os.path.isdir(directory):
            directory = os.path.dirname(directory)
        state.zenity_recent_dir = directory
    return (procres.returncode, result)


# note: In this section we wrap around file dialogues to allow for zenity
def zentk_askopenfilename(**options):
    try:
        result = zenity(filetypes=options.get("filetypes"), initialdir=options.get("initialdir"), multiple=False, title=options.get("title"))[1]
        if result and not os.path.isfile(result):
            print("A folder was selected while we need a file, ignoring selection.")
            return ''
    except Exception:
        from tkinter.filedialog import askopenfilename
        result = askopenfilename(**options)
    return result


def zentk_askopenfilenames(**options):
    try:
        result = zenity(filetypes=options.get("filetypes"), initialdir=options.get("initialdir"), multiple=True, title=options.get("title"))[1]
        for itm in result:
            if itm and not os.path.isfile(itm):
                print("A folder was selected while we need a file, ignoring selection.")
                return ''
    except Exception:
        from tkinter.filedialog import askopenfilenames
        result = askopenfilenames(**options)
    return result


def zentk_askdirectory(**options):
    try:
        result = zenity(initialdir=options.get("initialdir"), multiple=False, title=options.get("title"), directory=True)[1]
    except Exception:
        from tkinter.filedialog import askdirectory
        result = askdirectory(**options)
    return result


def zentk_asksaveasfilename(**options):
    try:
        result = zenity(filetypes=options.get("filetypes"), initialdir=options.get("initialdir"), initialfile=options.get("initialfile"), multiple=False, title=options.get("title"), save=True)[1]
    except Exception:
        from tkinter.filedialog import asksaveasfilename
        result = asksaveasfilename(**options)
    return result
