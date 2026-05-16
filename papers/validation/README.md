# Validation of the `meer21cm` pipeline
This page holds the validation pipeline for the `meer21cm` package.
Each validation test is numbered by the two digits and described below.
For each test, the `func_xx.py` holds the script for the validation calculations.

All tests are done assuming a survey area similar to one patch of the MeerKLASS UHF survey,
in the redshift range of [0.4,1.1] in cross-correlation with DESI LRG.

## 00: Mock simulation test
In a given rectangular box, we generate mock HI temperature box and mock galaxy positions following an input power spectrum, HI and galaxy bias and galaxy redshift kernel. And then we test if the estimated power spectrum from the box matches the input.

## 01: Power spectrum recovery test
Given the mock field, propagate the mock HI temperature and galaxy positions into sky map and catalogue, regrid the sky map into observed density fields with observational effects applied. Then we test if the output power spectrum agrees with the input model with additional terms accounting for the observational effects.

## 02: Transfer function validation test
Similar to 01, with foreground removal effects applied. We then test two ways of recovering the power spectrum, either by numerically constructing the transfer function through mock injection tests, or through analytically calculating the window function based on the PCA eigenvectors. The transfer function is then applied to the model and test the agreement between the corrected model and the foreground cleaned mock signal.