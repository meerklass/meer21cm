# Validation of the `meer21cm` pipeline
This page holds the validation pipeline for the `meer21cm` package.
Each validation test is numbered by the two digits and described below.
For each test, the `func_xx.py` holds the script for the validation calculations.

All tests are done assuming a survey area similar to one patch of the MeerKLASS UHF survey,
in the redshift range of [0.4,1.1] in cross-correlation with DESI LRG.

## 00: Mock simulation test
In a given rectangular box, we generate mock HI temperature box and mock galaxy positions following an input power spectrum, HI and galaxy bias and galaxy redshift kernel. And then we test if the estimated power spectrum from the box matches the input.

## 01: Gridding test
