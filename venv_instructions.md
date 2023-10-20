# Virtual Environment Setup and Usage

This document outlines the steps to create, activate, install packages, and deactivate a virtual environment using `venv` in Python.

## Creating a Virtual Environment

1. Navigate to the project directory in your terminal.
2. Run the following command to create a virtual environment in the current directory:
```bash
python3 -m venv venv
```

## Activating the Virtual Environment

### On Windows:
```bash
.\venv\Scripts\activate
```

### On macOS and Linux:
```bash
source venv/bin/activate
```

You should now see the name of your virtual environment in the command prompt.

## Installing Packages

1. With the virtual environment activated, you can now install packages using `pip`. For example:
```bash
pip install requests
```

2. To install multiple packages listed in a `requirements.txt` file, use:
```bash
pip install -r requirements.txt
```

## Deactivating the Virtual Environment

To exit the virtual environment and return to the global Python environment, run:
```bash
deactivate
```
