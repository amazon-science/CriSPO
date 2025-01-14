# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
IO utilities.
Copied from https://github.com/hankcs/HanLP/blob/master/hanlp/utils/log_util.py
Apache License 2.0
"""

import datetime
import io
import logging
import os
import sys

import termcolor

IPYTHON = False


class ErasablePrinter(object):
    def __init__(self, out=sys.stderr):
        self._last_print_width = 0
        self.out = out

    def erase(self):
        if self._last_print_width:
            if IPYTHON:
                self.out.write("\r")
                self.out.write(" " * self._last_print_width)
            else:
                self.out.write("\b" * self._last_print_width)
                self.out.write(" " * self._last_print_width)
                self.out.write("\b" * self._last_print_width)
            self.out.write("\r")  # \r is essential when multi-lines were printed
            self._last_print_width = 0

    def print(self, msg: str, color=True):
        self.erase()
        if color:
            if IPYTHON:
                msg, _len = color_format_len(msg)
                _len = len(msg)
            else:
                msg, _len = color_format_len(msg)
            self._last_print_width = _len
        else:
            self._last_print_width = len(msg)
        self.out.write(msg)
        self.out.flush()


_printer = ErasablePrinter()


def flash(line: str, color=True):
    _printer.print(line, color)


def color_format(msg: str):
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f"[{c}]", f"[/{c}]"
            msg = msg.replace(start, "\033[%dm" % v).replace(end, termcolor.RESET)
    return msg


def remove_color_tag(msg: str):
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f"[{c}]", f"[/{c}]"
            msg = msg.replace(start, "").replace(end, "")
    return msg


def color_format_len(msg: str):
    _len = len(msg)
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f"[{c}]", f"[/{c}]"
            msg, delta = _replace_color_offset(msg, start, "\033[%dm" % v)
            _len -= delta
            msg, delta = _replace_color_offset(msg, end, termcolor.RESET)
            _len -= delta
    return msg, _len


def _replace_color_offset(msg: str, color: str, ctrl: str):
    chunks = msg.split(color)
    delta = (len(chunks) - 1) * len(color)
    return ctrl.join(chunks), delta


def cprint(*args, file=None, **kwargs):
    out = io.StringIO()
    print(*args, file=out, **kwargs)
    text = out.getvalue()
    out.close()
    c_text = color_format(text)
    print(c_text, end="", file=file)


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%", enable=True):
        super().__init__(fmt, datefmt, style)
        self.enable = enable

    def formatMessage(self, record: logging.LogRecord) -> str:
        message = super().formatMessage(record)
        if self.enable:
            return color_format(message)
        else:
            return remove_color_tag(message)


def init_logger(
    name=None,
    save_dir=None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    mode="w",
    fmt="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    if not name:
        name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    rootLogger = logging.getLogger(os.path.join(save_dir, name) if save_dir else name)
    rootLogger.propagate = False

    consoleHandler = logging.StreamHandler(
        sys.stdout
    )  # stderr will be rendered as red which is bad
    consoleHandler.setFormatter(ColoredFormatter(fmt, datefmt=datefmt))
    attached_to_std = False
    for handler in rootLogger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stderr or handler.stream == sys.stdout:
                attached_to_std = True
                break
    if not attached_to_std:
        rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(min(console_level, file_level))
    consoleHandler.setLevel(console_level)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_path = f"{save_dir}/{name}"
        if "." not in name:
            log_path += ".log"
        fileHandler = logging.FileHandler(log_path, mode=mode)
        fileHandler.setFormatter(ColoredFormatter(fmt, datefmt=datefmt, enable=False))
        rootLogger.addHandler(fileHandler)
        fileHandler.setLevel(file_level)

    return rootLogger
