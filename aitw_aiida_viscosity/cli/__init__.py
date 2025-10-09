# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
"""Module for the command line interface."""
import aiida
# from aiida.cmdline.groups import VerdiCommandGroup as _VerdiCommandGroup
from aiida.cmdline.groups import VerdiCommandGroup
from aiida.cmdline.params import options, types
import click as original_click
import plumpy
from rich.traceback import install
import rich_click as click

install(
    suppress=[original_click, click, aiida, plumpy]
)

@click.group(
    'aitw-aiida-viscosity',
    cls=VerdiCommandGroup,
    context_settings={'help_option_names': ['-h', '--help']}
)
@options.PROFILE(type=types.ProfileParamType(load_profile=True), expose_value=False)
def cmd_root():
    """CLI for the `AITW-aiida-viscosity` plugin."""

@cmd_root.command()
def hello():
    """Print a hello world message."""
    click.echo('Hello, AITW-aiida-viscosity!')

from .workflows import cmd_workflow

__all__ = (
    'cmd_root',
    'cmd_workflow',
)
