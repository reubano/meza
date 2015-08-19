.PHONY: help clean clean-pyc clean-build check-stage pipme require lint test release sdist wheel

help:
	@echo "clean - remove all artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "check-stage - check staged changes for lint errors"
	@echo "pipme - install requirements.txt"
	@echo "require - create requirements.txt"
	@echo "lint - check style with flake8"
	@echo "test - run nose and script tests"
	@echo "release - package and upload a release"
	@echo "sdist - create a source distribution package"
	@echo "wheel - create a wheel package"

clean: clean-build clean-pyc

clean-build:
	helpers/clean-build

clean-pyc:
	helpers/clean-pyc

check-stage:
	helpers/check-stage

pipme:
	pip install -r requirements.txt

require:
	pip freeze -l | grep -vxFf dev-requirements.txt > requirements.txt

lint:
	flake8 tabutils tests

test:
	nosetests -xv

release:
	helpers/release

sdist:
	helpers/sdist

wheel:
	helpers/wheel
