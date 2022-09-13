# CIFAR10_Benchmark
## Design
| Environment      | Version            |
| ---------------- | ------------------ |
| Operating System | Ubuntu 20.04.4 LTS, Windows 10 21H2 |
| Python           | 3.8.10 |
| PIP              | pip 22.1.2 |

| Experimental Setup    | Component | 
| --- | --- |
| Windows 10 21H2       | CPU: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz   1.80 GHz |
|                       | RAM: 32.0 GB (31.4 GB usable) |
| Ubuntu 20.04.5 LTS    | CPU: Intel® Core™ i7-8700 CPU @ 3.20GHz × 12  |
|                       | RAM: 31,2 GiB |

# System 
## Installation
The installation of python depends per operating system.

The python dependencies can be installed using REQUIREMENTS.txt (see INSTALL.sh).

## Execution
There are two distinct ways to run code located in ''/src'':
1. **Jupyter**: The IPYNB files are considered the main files. It is recommended to use the program
[Visual Studio Code](https://code.visualstudio.com/) with the [Jupyter-Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter). Any alternative may also work.
2. **Native python**: The PY files are simply copied from the content of the Jupyter files. Mainly used in context of the RUN scripts (Powershell or Shell) for multiple execution.

> Note that ''/src/config.py'' contains additional settings which are relevant for both, IPYNB and PY files.

The execution of either Jupyter or native python files will create log-files in ''/log'', which contain the most crucial information on the programs performance.

## Troubleshooting / Errors

None as of now.

# Documentation
## Report
Available in [tex](tex\main.pdf)

## Log-Files
Some are archived in [log-lnx](log-lnx) and [log-win](log-win). New logs are generated as the python scripts are executed.