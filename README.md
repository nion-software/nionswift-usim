# STEM Microscope Simulator

A STEM microscope simulator for use with Nion Swift. Used for debugging Nion Swift acquisition and developing
acquisition tools, techniques, and apps.

This project is in rapid development and does not yet have a stable release. To run with Nion Swift 0.11, use the
instructions below. To do development on this project, you will typically need to install the master (developer) version
of Nion Swift to get everything running.

## Stable Installation (for Nion Swift 0.11)

If you are looking for a version to work with Nion Swift 0.11 (the latest public release of Nion Swift),
download `nionswift-usim` here:

[Microscope Simulator for NionSwift 0.11](https://github.com/nion-software/nionswift-usim/archive/nionswift-0.11.1.zip)

Unzip the file. Then using the command line, run the following commands.

    $ cd nionswift-usim-nionswift-0.11.1
    $ /path/to/python/pip install --no-deps .
    
The `no-deps` argument is required since with Nion Swift 0.11 the required packages will not automatically be on
the `PYTHONPATH`.

If you are running on Windows, you will also need to download `nionswift-instrumentation-kit` here:

[Instrumentation Kit for NionSwift 0.11 -- Windows Only](https://github.com/nion-software/nionswift-instrumentation-kit/archive/nionswift-0.11.1.zip)

Unzip the file. Then using the command line, run the following commands.

    $ cd nionswift-instrumentation-kit-nionswift-0.11.1
    $ /path/to/python/pip install --no-deps .

## Developer Installation

Instructions to install Nion Swift as a developer:

[Nion Swift Developer Installation](https://github.com/nion-software/nionswift/wiki/Developer-installation) 

At the bottom of the instructions above are command line instructions for installing this project. 

## Future Installation Plans

Nion Swift is currently (2017-October) in the process of being migrated to PyPI. When that happens, stable
versions of this project will be provided and installation will simply be `pip install nionswift-usim`.

## Versioning

We use [SemVer](http://semver.org/) for versioning. 

## License

This project is licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html).
