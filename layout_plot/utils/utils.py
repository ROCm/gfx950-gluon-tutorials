##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

import argparse
import subprocess


def run_bash_command(commandstring):
    proc = subprocess.run(
        commandstring, shell=True, check=False, executable="/bin/bash", stdout=subprocess.PIPE
    )
    return proc.returncode


class OneLineFormatter(argparse.HelpFormatter):

    def _format_action_invocation(self, action):
        if not action.option_strings:
            (metavar,) = self._metavar_formatter(action, action.dest)(1)
            return metavar

        parts = []

        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default_metavar = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default_metavar)
            parts.extend([f"{opt} {args_string}" for opt in action.option_strings])

        return ", ".join(parts)

    def _get_help_string(self, action):
        help_text = action.help or ""
        if "%(default)" not in help_text and action.default is not argparse.SUPPRESS:
            if action.default is not None:
                help_text += f" (default: {action.default})"
        return help_text
