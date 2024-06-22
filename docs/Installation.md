**Installing Bindwell**
=====================

This guide will walk you through the steps to install and set up Bindwell on your system.

**Prerequisites**
---------------

* Python 3.10.14 (make sure you have this version installed on your system)
* Git (for cloning the repository)


### Clone the Repository

Clone the Bindwell repository using Git:

```bash
git clone https://github.com/trrt-good/Bindwell.git
```

Then navigate into the directory:

```bash
cd Bindwell
```

### (Recommended) Create and Activate a Virtual Environment

Create a new virtual environment using Python's built-in `venv` module:

```bash
python -m venv bindwell-env
```

Activate the virtual environment:

**On Windows:**
```
bindwell-env\Scripts\activate
```

**On macOS/Linux:**
```
source bindwell-env/bin/activate
```

You should now see the name of the virtual environment in your command prompt.

### Install Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

This command will install all the necessary packages listed in the `requirements.txt` file.

**Run Bindwell**
--------------

To run Bindwell, simply execute:

```bash
python app.py
```

To run Bindwell with Nvidia CUDA support, run:

```bash
python app.py --device cuda
```

This will launch the Bindwell application.

**Troubleshooting**
---------------

If you encounter any issues during the installation process, please check the following:

* Make sure you have Python 3.10.14 installed on your system.
* Verify that you have cloned the repository correctly.
* Check the `requirements.txt` file for any missing dependencies.

If you still encounter issues, feel free to open an issue on the Bindwell GitHub page.
