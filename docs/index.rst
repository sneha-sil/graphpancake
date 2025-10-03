.. graphpancake documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

graphpancake's documentation
=========================================

**graphpancake** is a Python library for generating molecular graphs of small organic molecules (elements H, B, C, N, O, F, Si, P, S, Cl, Br, I) from electronic structure theory (i.e., DFT and WFT quantum chemistry calculations). 

Outputs from DFT calculations (.xyz coordinates), Natural Population Analysis (NPA) from JANPA, Natural Bond Orbital (NBO) analysis, and thermodynamic data from Shermo are parsed to extract atom (node), bond (edge), and graph-level features.

Essentially, graphpancake takes three-dimensional data and flattens it into molecular graph representations, because computers haven't taken organic chemistry class.

.. grid:: 2 2 2 2

    .. grid-item-card:: User Guide
      :margin: 0 3 0 0
      
      Comprehensive guide covering all features, with example code.

      .. button-link:: ./user_guide.html
         :color: primary
         :outline:
         :expand:

         To the user guide
      
    .. grid-item-card:: API Reference
      :margin: 0 3 0 0
      
      Complete API documentation for all classes and functions.

      .. button-link:: ./api.html
         :color: primary
         :outline:
         :expand:

         To the API reference


Getting started
--------------

Please refer to the :doc:`user_guide` for complete installation instructions, tutorials, and examples.

Create a complete conda environment with all dependencies:

.. code-block:: bash
   # Clone or download the environment.yml file from the repository
   curl -O https://raw.githubusercontent.com/sneha-sil/graphpancake/main/environment.yml

   # Create and activate the environment
   conda env create -f environment.yml
   conda activate graphpancake-env

Alternatively, installation with pip:

.. code-block:: bash

   pip install graphpancake

Single-molecule processing using the command-line interface:

.. code-block:: bash

   python -m graphpancake.cli create-db database_name.db

   python -m graphpancake.cli load-data --database database_name.db --xyz-file pentane.xyz --shermo-output pentane_shermo.txt --janpa-output pentane.JANPA --nbo-output pentane_nbo.out --mol-id pentane_QM --smiles "CCCCCC" --graph-type QM

   # Full list of commands and options available via:
   python -m graphpancake.cli --help

Batch-processing several molecules using a configurable script:

Batch processing works best when you have a folder or gzip file of hundreds or thousands of data files, with a corresponding CSV of identification names and SMILES strings. Examples are available in the auxiliary graphpancake_data.zip folder available in this repository.

1. Make a copy of config_template.yaml, rename as config.yaml
2. Adjust file names and operation settings as necessary
3. Run the script. A database .db file will be created with all of your molecular graph data.


Citation & license
------------
If you use graphpancake in your research, please cite:

Sil, S., Scheidt, K.A. graphpancake: A Python library to represent organic molecules as molecular graphs using electronic structure theory, ChemRxiv, 2025. DOI: 

This work is licensed under the MIT License. See the LICENSE file in the repository root for details.

References
-----------
Neese, F. et al. The ORCA quantum chemistry program package. J. Chem. Phys. 2020, 152, 224108

Nikolaienko et al. JANPA: an open source cross-platform implementation of the Natural Population Analysis on the Java platform, Computational and Theoretical Chemistry 2014, 1050, 15-22, DOI: 10.1016/j.comptc.2014.10.002, http://janpa.sourceforge.net

Glendening, E. D., Landis, C. R., Weinhold, F. NBO 7.0: New vistas in localized and delocalized chemical bonding theory. Journal of Computational Chemistry 2019, 40 (25), 2234-2241. https://doi.org/10.1002/jcc.25873

Tian, L., Qinxue, C., Shermo: A general code for calculating molecular thermodynamic properties, Comput. Theor. Chem. 2021, 1200, 113249 DOI: 10.1016/j.comptc.2021.113249

Acknowledgements
-----------------
Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11. Code written with assistance from Claude Sonnet 4.

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   user_guide
   api
