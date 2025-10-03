graphpancake
==============================

[![GitHub Actions Build Status](https://github.com/sneha-sil/graphpancake/workflows/CI/badge.svg)](https://github.com/sneha-sil/graphpancake/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/sneha-sil/graphpancake/branch/main/graph/badge.svg)](https://codecov.io/gh/sneha-sil/graphpancake/branch/main)

graphpancake is a Python library for generating molecular graphs of small organic molecules (elements H, B, C, N, O, F, Si, P, S, Cl, Br, I) from electronic structure theory (i.e., DFT and WFT quantum chemistry calculations). 

Outputs from DFT calculations (.xyz coordinates), Natural Population Analysis (NPA) from JANPA, Natural Bond Orbital (NBO) analysis, and thermodynamic data from Shermo are parsed to extract atom (node), bond (edge), and graph-level features.

Essentially, graphpancake takes three-dimensional data and flattens it into molecular graph representations, because computers haven't taken organic chemistry class.

---

## Installation

### Setup with Conda (Recommended)

Create a complete conda environment with all dependencies:

```bash
# Clone or download the environment.yml file from the repository
curl -O https://raw.githubusercontent.com/sneha-sil/graphpancake/main/environment.yml

# Create and activate the environment
conda env create -f environment.yml
conda activate graphpancake-env
```

### Alternative Installation Methods

#### Using pip (PyPI)
```bash
pip install graphpancake
```

#### Using conda (recommended for RDKit)
```bash
conda install -c conda-forge rdkit
pip install graphpancake
```
> **Note:** RDKit is not available on PyPI for all Python versions. It is best to install it with conda due to binary dependencies. This can be done via:
```bash
conda install -c conda-forge rdkit
```

### Development Installation
```bash
git clone https://github.com/sneha-sil/graphpancake.git
cd graphpancake
conda env create -f environment.yml
conda activate graphpancake-env
pip install -e .

---

## Example: Single-molecule processing using the command line interface

```bash
graphpancake create-db database_name.db

graphpancake load-data --database database_name.db --xyz-file pentane.xyz --shermo-output pentane_shermo.txt --janpa-output pentane.JANPA --nbo-output pentane_nbo.out --mol-id pentane_QM --smiles "CCCCCC" --graph-type QM
```

A full list of commands and options are available using:

```bash
graphpancake --help
```

> **Alternative:** If the `graphpancake` command is not available, you can also use `python -m graphpancake.cli`

---

## Example: Batch processing of several molecules using scripts

Batch processing works best when you have a folder or gzip file of hundreds or thousands of data files, with a corresponding CSV of identification names and SMILES strings. Examples are available in the auxiliary graphpancake_data.zip folder available in this repository. 

1. Make a copy of config_template.yaml, rename as config.yaml
2. Adjust file names and operation settings as necessary
3. Run the script. A database .db file will be created with all of your molecular graph data. 

---

## Documentation

Full documentation is available at:  
https://graphpancake.readthedocs.io/

---

## References

- Neese, F. et al. The ORCA quantum chemistry program package. J. Chem. Phys. 2020, 152, 224108
- Nikolaienko et al. JANPA: an open source cross-platform implementation of the Natural Population Analysis on the Java platform, Computational and Theoretical Chemistry 2014, 1050, 15-22, DOI: 10.1016/j.comptc.2014.10.002, http://janpa.sourceforge.net
- Glendening, E. D., Landis, C. R., Weinhold, F. NBO 7.0: New vistas in localized and delocalized chemical bonding theory. Journal of Computational Chemistry 2019, 40 (25), 2234-2241. https://doi.org/10.1002/jcc.25873
- Tian, L., Qinxue, C., Shermo: A general code for calculating molecular thermodynamic properties, Comput. Theor. Chem. 2021, 1200, 113249 DOI: 10.1016/j.comptc.2021.113249

---

## Citation

If you use `graphpancake` in your research, please cite this repository and the relevant references above.

Sil, S., Scheidt, K.A. graphpancake: A Python library to represent organic molecules as molecular graphs using electronic structure theory, ChemRxiv, 2025. DOI: 

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11. Code written with assistance from Claude Sonnet 4.
=======
# graphpancake
A Python library for generating molecular graphs of small organic molecules from DFT and WFT calculations (elements: H, B, C, N, O, F, Si, P, S, Cl, Br, I).
Input requires an Orca .out file, .xyz coordinates file, and .mp2nat file. Generates thermostatistical data via Shermo and natural population analysis via JANPA. 

Citations: 

Neese, F. et al. The ORCA quantum chemistry program package. J. Chem. Phys. 2020, 152, 224108

Nikolaienko et al. JANPA: an open source cross-platform implementation of the Natural Population Analysis on the Java platform, Computational and Theoretical Chemistry 2014, 1050, 15-22, DOI: 10.1016/j.comptc.2014.10.002, http://janpa.sourceforge.net

Tian Lu, Qinxue Chen, Shermo: A general code for calculating molecular thermodynamic properties, Comput. Theor. Chem. 2021, 1200, 113249 DOI: 10.1016/j.comptc.2021.113249

Kneiding et al. Deep learning metal complex properties with natural quantum graphs. Digit. Discov. 2023, 2, 618-633

Rasmussen et al. SMILES all around: structure to SMILES conversion for transition metal complexes. J Cheminform. 2025, 17, 63. https://doi.org/10.1186/s13321-025-01008-1
>>>>>>> 7c90d1e36b4fc8183ae643043760e1ef3b7e8563
