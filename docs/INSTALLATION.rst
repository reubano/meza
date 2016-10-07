Installation
------------

(You are using a `virtualenv`_, right?)

At the command line, install meza using either ``pip`` (recommended)

.. code-block:: bash

    pip install meza

or ``easy_install``

.. code-block:: bash

    easy_install meza

Detailed installation instructions
----------------------------------

If you have `virtualenvwrapper`_ installed, at the command line type:

.. code-block:: bash

    mkvirtualenv meza
    pip install meza

Or, if you only have ``virtualenv`` installed:

.. code-block:: bash

	virtualenv ~/.venvs/meza
	source ~/.venvs/meza/bin/activate
	pip install meza

Otherwise, you can install locally::

    pip install --user meza

.. _virtualenv: https://virtualenv.pypa.io/en/latest/index.html
.. _virtualenvwrapper: https://virtualenvwrapper.readthedocs.org/en/latest/
