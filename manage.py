#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

""" A script to manage development tasks """
from os import path as p
from subprocess import call, check_call, CalledProcessError
from manager import Manager

manager = Manager()
BASEDIR = p.dirname(__file__)
DEF_WHERE = ["meza", "tests", "setup.py", "manage.py"]


def upload_():
    """Upload distribution files"""
    command = 'twine upload --repository-url https://upload.pypi.org/legacy/ {0}'
    check_call(command.format(p.join(BASEDIR, 'dist', '*')).split(' '))


def sdist_():
    """Create a source distribution package"""
    check_call(p.join(BASEDIR, 'helpers', 'srcdist'))


def wheel_():
    """Create a wheel package"""
    check_call(p.join(BASEDIR, 'helpers', 'wheel'))


def clean_():
    """Remove Python file and build artifacts"""
    check_call(p.join(BASEDIR, 'helpers', 'clean'))


@manager.command
def check():
    """Check staged changes for lint errors"""
    exit(call(p.join(BASEDIR, 'helpers', 'check-stage')))


@manager.arg('where', 'w', help='Modules to check', default='meza')
@manager.arg('strict', 's', help='Check with pylint')
@manager.arg('compatibility', 'c', help='Check with pylint porting checker')
@manager.command
def lint(where=None, strict=False, compatibility=False):
    """Check style with linters"""
    _where = where or ' '.join(DEF_WHERE)
    command = f"pylint --rcfile=tests/standard.rc -rn -fparseable {_where}"

    try:
        check_call(['flake8'] + _where.split(' '))

        if strict:
            check_call(command, shell=True)

        if compatibility:
            check_call(f"{command} --py3k", shell=True)
    except CalledProcessError as e:
        exit(e.returncode)


@manager.command
def pipme():
    """Install requirements.txt"""
    exit(call('pip install -r requirements.txt'.split(' ')))


@manager.command
def require():
    """Create requirements.txt"""
    cmd = 'pip freeze -l | grep -vxFf dev-requirements.txt > requirements.txt'
    exit(call(cmd.split(' ')))


@manager.arg('source', 's', help='the tests to run', default=None)
@manager.arg('where', 'w', help='test path', default=None)
@manager.arg(
    'stop', 'x', help='Stop after first error', type=bool, default=False)
@manager.arg(
    'failed', 'f', help='Run failed tests', type=bool, default=False)
@manager.arg(
    'cover', 'c', help='Add coverage report', type=bool, default=False)
@manager.arg('tox', 't', help='Run tox tests', type=bool, default=False)
@manager.arg('detox', 'd', help='Run detox tests', type=bool, default=False)
@manager.arg(
    'verbose', 'v', help='Use detailed errors', type=bool, default=False)
@manager.arg(
    'parallel', 'p', help='Run tests in parallel in multiple processes',
    type=bool, default=False)
@manager.arg(
    'debug', 'D', help='Use nose.loader debugger', type=bool, default=False)
@manager.command
def test(source=None, where=None, stop=False, **kwargs):
    """Run nose, tox, and script tests"""
    opts = '-xv' if stop else '-v'
    opts += ' --with-coverage' if kwargs.get('cover') else ''
    opts += ' --failed' if kwargs.get('failed') else ' --with-id'
    opts += ' --processes=-1' if kwargs.get('parallel') else ''
    opts += ' --detailed-errors' if kwargs.get('verbose') else ''
    opts += ' --debug=nose.loader' if kwargs.get('debug') else ''
    opts += ' -w {}'.format(where) if where else ''
    opts += ' {}'.format(source) if source else ''

    try:
        if kwargs.get('tox'):
            check_call('tox')
        elif kwargs.get('detox'):
            check_call('detox')
        else:
            check_call(('nosetests {}'.format(opts)).split(' '))
    except CalledProcessError as e:
        exit(e.returncode)


@manager.command
def release():
    """Package and upload a release"""
    try:
        clean_()
        sdist_()
        wheel_()
        upload_()
    except CalledProcessError as e:
        exit(e.returncode)


@manager.command
def build():
    """Create a source distribution and wheel package"""
    try:
        clean_()
        sdist_()
        wheel_()
    except CalledProcessError as e:
        exit(e.returncode)


@manager.command
def upload():
    """Upload distribution files"""
    try:
        upload_()
    except CalledProcessError as e:
        exit(e.returncode)


@manager.command
def sdist():
    """Create a source distribution package"""
    try:
        sdist_()
    except CalledProcessError as e:
        exit(e.returncode)


@manager.command
def wheel():
    """Create a wheel package"""
    try:
        wheel_()
    except CalledProcessError as e:
        exit(e.returncode)


@manager.command
def clean():
    """Remove Python file and build artifacts"""
    try:
        clean_()
    except CalledProcessError as e:
        exit(e.returncode)


if __name__ == '__main__':
    manager.main()
