# -*- coding: utf-8 -*-
#
# Copyright (C) 2017-2020 Matthias Klumpp <matthias@tenstral.net>
#
# Licensed under the GNU Lesser General Public License Version 3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the license, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.

import os
import re
import sys


def console_supports_color():
    '''
    Returns True if the running system's terminal supports color, and False
    otherwise.
    '''

    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return 'ANSICON' in os.environ or is_a_tty


def print_textbox(title, tl, hline, tr, vline, bl, br):
    def write_utf8(s):
        sys.stdout.buffer.write(s.encode('utf-8'))

    tlen = len(title)
    write_utf8('\n{}'.format(tl))
    write_utf8(hline * (10 + tlen))
    write_utf8('{}\n'.format(tr))

    write_utf8('{}  {}'.format(vline, title))
    write_utf8(' ' * 8)
    write_utf8('{}\n'.format(vline))

    write_utf8(bl)
    write_utf8(hline * (10 + tlen))
    write_utf8('{}\n'.format(br))

    sys.stdout.flush()


def print_header(title):
    print_textbox(title, '╔', '═', '╗', '║', '╚', '╝')


def print_section(title):
    print_textbox(title, '┌', '─', '┐', '│', '└', '┘')


def print_task(*arg):
    '''
    Prints a task info and ensures that it shows up on
    stdout immediately.
    '''
    print('->', *arg)
    sys.stdout.flush()


def print_info(*arg):
    '''
    Prints an information message and ensures that it shows up on
    stdout immediately.
    '''
    print(*arg)
    sys.stdout.flush()


def print_warn(*arg):
    '''
    Prints an information message and ensures that it shows up on
    stdout immediately.
    '''
    if console_supports_color():
        print('\033[93m/!\\\033[0m', *arg)
    else:
        print('/!\\', *arg)
    sys.stdout.flush()


def print_error(*arg):
    '''
    Prints an information message and ensures that it shows up on
    stdout immediately.
    '''
    if console_supports_color():
        print('\033[91mERROR:\033[0m', *arg, file=sys.stderr)
    else:
        print('ERROR:', *arg, file=sys.stderr)
    sys.stderr.flush()
