# Installation

The installation is tested on arm64 MacOS system as well as x86_64 Linux system, on python 3.9, 3.10 and 3.11.

If you are on ilifu jump straight to [`Installing on ilifu`](#ilifu).

## 1. Create `conda` environment
```
conda create -n meer21cm python=3.10
conda activate meer21cm
```

### HPC
If you are using your own computer, you can simply download and install anaconda.
If you are on a cluster, most likely you already have some conda module available. You can check its availability by
```
module avail
```
Typically this will be called `conda`, `anaconda` or `anaconda3`.
Then do
```
module load anaconda3
```

If the HPC does not have conda, you can also install your own miniconda (check the [official guide](https://docs.conda.io/en/latest/miniconda.html) for more details). One thing to note is that you need to change the path to which conda is installed when using the installation prompt as you typically do not have access to the default system path.

## 2. Installing dependencies (optional)
In most cases, `meer21cm` can be installed directly via pip.
Occationally, you may run into installation errors due to not being able
to install hdf5 and c-blosc.
In that case, you can

### MacOS
For MacOS, you can install hdf5 through brew.
```
brew install hdf5
brew install c-blosc
export HDF5_DIR=/opt/homebrew/opt/hdf5
export BLOSC_DIR=/opt/homebrew/opt/c-blosc
```

### Linux PC
For Linux, most of the time you can install the h5df dependency through `conda`.

```
conda install hdf5
```

### HPC
Again, if you are using HPC, most likely there are hdf5 module available, and you can load the module by simply `module load hdf5`.

## 3. Install `meer21cm`
Finally, clone the repo for `meer21cm`
```
git clone git@github.com:meerklass/meer21cm.git
```

Go to the directory
```
cd meer21cm
```

And run
```
pip install -e .
```
Note that development install `-e` is needed, as this package is in early stage and will not have a stable version before the official release.

If you want to run tests, or if you want to use `baccoemu` instead of `camb`, or run sampler with `mpi` pools, instead of the installation above do
```
pip install -e ".[test]"
```

and you can run
```
pytest tests/
```
to see if the current installation passes all the tests.

If you want to develop `meer21cm`, you should install the full dependency
```
pip install -e ".[full]"
```

For developing the code, the development install `-e` is always needed since the installation needs to change dynamically with your edits.

## 4. Check if installation is successful
If you did not run the tests, in the conda environment, do
```
python -c "import meer21cm; print(meer21cm.__file__)"
```
If you see the output, then the installation is successful.
The file path is also a useful indicator of the installation. If you have installed it with `-e`, the file path should be the cloned repo,
whereas a static installation will be in the `site-packages` directory.

## A. Installing on ilifu
<a name="ilifu"></a>
If you are on ilifu, the installation has been tested so you can follow the exact steps listed here.

```
git clone https://github.com/meerklass/meer21cm
module load anaconda3
conda create -n meer21cm python=3.10
conda activate meer21cm
cd meer21cm
pip install -e ".[full]"
```

Optionally, to use the environment as a jupyter kernel run
```
pip install ipykernel
python -m ipykernel install --name "meer21cm" --user
```

If you have accidentally linked an existing "meer21cm" kernel created externally and want to use the latest version, simply change the "meer21cm" above to a different kernel name and use it instead.
