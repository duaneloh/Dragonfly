Frame Viewer
============

The Frame Viewer is a GUI for browsing individual diffraction frames stored in
EMC format files. It supports displaying masked pixels, computing a powder sum
of all frames, and comparing measured frames against predicted intensities from
a reconstruction.

.. image:: /_static/images/frameviewer_screenshot.png
   :alt: Frame Viewer screenshot
   :width: 50%

Launch
------

From within a reconstruction directory (reads detector and data file paths from
``config.ini``)::

    dragonfly.frameviewer

With a specific config file::

    dragonfly.frameviewer -c config.ini

To view a standalone EMC file without a config file (requires both detector
and EMC file)::

    dragonfly.frameviewer -e data/photons.emc -d data/det.h5

Command-line Options
--------------------

.. code-block:: text

    usage: dragonfly.frameviewer [-c CONFIG_FILE] [-M] [-P] [-C]
                                 [-e EMC_FNAME -d DET_FNAME]

    -c, --config_file   Path to configuration file (default: config.ini)
    -M, --mask          Zero out masked pixels
    -P, --powder        Show powder sum of all frames
    -C, --compare       Compare with predicted intensities (needs data/quat.dat)
    -e, --emc_fname     Path to EMC file (requires -d)
    -d, --det_fname     Path to detector file (requires -e)

Modes
-----

**Default mode**
    Browse frames one at a time. Use Next, Previous, or Random to navigate.
    Enter a frame number directly in the text field.

**Powder mode** (``-P``)
    Display the sum of all frames (powder pattern). Useful for checking
    data quality and detector geometry.

**Compare mode** (``-C``)
    Show the measured frame alongside the predicted intensity from the best
    matching orientation. Requires ``data/quat.dat`` (generated during
    reconstruction). Cannot be used with standalone EMC/detector file mode.

**Mask display** (``-M``)
    Zero out masked detector pixels when displaying frames.

Color Maps
----------

Select a color map from the menu bar. Available maps: ``coolwarm``,
``cubehelix``, ``CMRmap``, ``gray``, ``gray_r``, ``jet``.

Keyboard Shortcuts
------------------

===================  ==================================
Shortcut             Action
===================  ==================================
``Ctrl+Q``           Quit
===================  ==================================
