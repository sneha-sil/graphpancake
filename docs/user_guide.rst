graphpancake user guide
=======================

This is a tutorial/user guide for all aspects of graphpancake for molecular graph generation and post-processing.

Installation
============

Basic installation
------------------

Install graphpancake using pip:

.. code-block:: bash

    pip install graphpancake

Recommended installation (with RDKit)
-------------------------------------

For full functionality, install RDKit via conda first:

.. code-block:: bash

    conda install -c conda-forge rdkit
    pip install graphpancake

Development installation
------------------------

For development or to get the latest features:

.. code-block:: bash

    git clone https://github.com/sneha-sil/graphpancake.git
    cd graphpancake
    pip install -e .

Processing QM data and creating molecular graphs
===============================================

graphpancake supports different levels of quantum chemical analysis:

.. list-table:: Supported graph types
   :header-rows: 1
   :widths: 15 30 25 30

   * - Type
     - Description
     - Required Files
     - Features
   * - **DFT**
     - Basic quantum chemistry
     - XYZ, Shermo
     - Coordinates, thermodynamics
   * - **NPA**
     - Natural Population Analysis
     - + JANPA
     - Wiberg matrices, NPA charges
   * - **NBO**
     - Natural Bond Orbital
     - + NBO
     - Natural charges, orbital data
   * - **QM**
     - Combined analysis
     - + JANPA + NBO
     - All NPA + NBO features

Everything you need to include for a molecular graph
---------------------------------------------------

- *Basic information*: SMILES string, molecule identifier (name, ID, etc.)
  - Provided via command line or CSV file
- *XYZ file*: Cartesian coordinates from DFT optimizations
- *Shermo output*: Thermodynamic properties from frequency calculations
- *JANPA output*: Natural population analysis (for NPA and QM graphs)
- *NBO output*: Natural bond orbital analysis (for NBO and QM graphs)

Constructing molecular graphs
=============================

Command line interface
----------------------

The CLI is intended to provide a simple UX for processing quantum mechanical data for small organic molecules and subsequently building molecular graphs, all stored conveniently in a SQLite database (with tables for graph, node, edge, and target features). The CLI is included with the graphpancake package:

.. code-block:: bash

    pip install graphpancake


After installing graphpancake, you can use the command line interface directly:

.. code-block:: bash

    graphpancake --help

If the ``graphpancake`` command is not available (e.g., if entry points are not properly installed), you can also run the CLI using:

.. code-block:: bash

    python -m graphpancake.cli --help

Available commands::

  create-db    Create a new database
  load-data    Load QM data for a single molecule
  query        Query and filter the database (filtering and sorting)
  filter       Preview graphs by a filter criteria (e.g., # of atoms, mass range, SMILES)
  stats        Show database statistics (# nodes, edges, etc.)
  delete       Delete graphs from database (single graphs, all, or by a filter)
  add-labels   Add target labels from CSV file for machine learning
  export-ml    Export ML-ready features

--------

Docs and help
-------------

.. code-block:: bash

    # General help
    graphpancake --help

    # Command-specific help
    graphpancake COMMAND --help
    graphpancake load-data --help
    graphpancake create-db --help
    graphpancake query --help
    graphpancake stats --help


Commands overview
=================

create-db
---------
Create a new database for molecular graphs.

.. code-block:: bash

    graphpancake create-db DATABASE_FILE [OPTIONS]

  - ``--graph-types TYPES``: Graph types to support (DFT, NPA, NBO, QM)
  - ``--force``: Overwrite existing database

    # Example:
    graphpancake create-db molecules.db --graph-types DFT NPA QM

load-data
---------
Load data from QM output files for a single molecule.

.. code-block:: bash

    graphpancake load-data [OPTIONS]

*Required:*

  - ``--database, -d PATH``: database file path
  - ``--mol-id ID``: identifier/name
  - ``--smiles SMILES``: SMILES string
  - ``--xyz-file PATH``: .xyz coordinate file
  - ``--shermo-output PATH``: Shermo output file

*Optional:*

  - ``--janpa-output PATH``: JANPA output  
  - ``--nbo-output PATH``: NBO output
  - ``--graph-type TYPE``: graph type (DFT, NPA, NBO, QM)

Examples:

.. code-block:: bash

    # Load a single molecule (minimum data - DFT level)
    graphpancake load-data \
        --database DFT_molecules.db  \
        --xyz-file benzaldehyde.xyz \
        --shermo-output benzaldehyde_shermo.txt \
        --mol-id benzaldehyde_DFT \
        --smiles "C1=CC=C(C=C1)C=O" \
        --graph-type DFT

    # Load a single molecule (with only JANPA output - NPA level)
    graphpancake load-data \
        --database NPA_molecules.db \
        --xyz-file benzaldehyde.xyz \
        --janpa-output benzaldehyde.JANPA \
        --mol-id benzaldehyde_NPA \
        --smiles "C1=CC=C(C=C1)C=O" \
        --graph-type NPA

    # Load a single molecule (with only NBO output - NBO level)
    graphpancake load-data \
        --database NBO_molecules.db \
        --xyz-file benzaldehyde.xyz \
        --nbo-output benzaldehyde_nbo.out \
        --mol-id benzaldehyde_NBO \
        --smiles "C1=CC=C(C=C1)C=O" \
        --graph-type NBO

    # Load a single molecule (with all possible data - QM level)
    graphpancake load-data \
        --database QM_molecules.db \
        --xyz-file benzaldehyde.xyz \
        --shermo-output benzaldehyde_shermo.txt \
        --janpa-output benzaldehyde.JANPA \
        --nbo-output benzaldehyde_nbo.out \
        --mol-id benzaldehyde_QM \
        --smiles "C1=CC=C(C=C1)C=O" \
        --graph-type QM

query
-----
Query and filter the molecular graph database.

.. code-block:: bash

    graphpancake query [OPTIONS]

*Required:*

  - ``--database, -d PATH``: Database file path

*Optional:*

  - ``--graph-type TYPE``: Filter by graph type
  - ``--limit N``: Maximum number of results
  - ``--output PATH``: Save results to CSV file
  - ``--format FORMAT``: Output format (table, json, csv)

Examples:

.. code-block:: bash

    # List all molecules in a table format
    graphpancake query \
        --database molecules.db

    # List n molecules in a table format (n = 3)
    graphpancake query \
        --database molecules.db \
        --limit 3

    # List all molecules in a JSON format
    graphpancake query \
        --database molecules.db \
        --format json
    
    # List all molecules in a CSV format
    graphpancake query \
        --database molecules.db \
        --format csv

    # Filter by graph type (e.g., QM)
    graphpancake query \
        --database molecules.db \
        --graph-type QM

    # Save query results to a CSV file
    graphpancake query \
        --database molecules.db \
        --output results.csv \
        --format csv

    # Save query results to a JSON file
    graphpancake query \
        --database molecules.db \
        --output results.json \
        --format json

filter
------
The filter option allows you to preview graphs from the database by a filter criteria, such as number of atoms, mass range, or SMILES patterns. This command does not modify the database.

.. code-block:: bash

    graphpancake filter [OPTIONS]

*Required:*

  - ``--database, -d PATH``: Database file path

*Optional:*

  - ``--where CONDITION``: SQL WHERE clause to filter graphs (e.g., "num_atoms > 10")
  - ``--show DETAILS``: Show detailed information (e.g., molecular formula, mass)
  - ``--limit N``: Maximum number of results
  - ``--format FORMAT``: Output format (table, json, csv)
  - ``--output PATH``: Save results to file

Examples:

.. code-block:: bash
  
    # Preview graphs by number of atoms (e.g., > 18)
    graphpancake filter by-criteria \
        --database molecules.db \
        --where "num_atoms > 18"

    # Preview graphs by molecular mass range (e.g., between 70 and 80)
    graphpancake filter by-criteria \
        --database molecules.db \
        --where "molecular_mass BETWEEN 70 and 80"

    # Preview graphs by SMILES pattern (e.g., containing 'CCC')
    graphpancake filter by-criteria \
        --database molecules.db \
        --where "smiles LIKE '%CCC%'"

    # Save filter results to a CSV file
    graphpancake filter by-criteria \
        --database molecules.db \
        --where "num_atoms > 18" \
        --output filtered_results.csv \
        --format csv

    # Save filter results to a JSON file
    graphpancake filter by-criteria \
        --database molecules.db \
        --where "num_atoms > 18" \
        --output filtered_results.json \
        --format json

stats
-----
The stats command provides the total number of molecules, total number of nodes and edges, and average nodes and edges per molecule. There are also options for obtaining molecular weight distributions, number of graphs by type, number of targets.

.. code-block:: bash

    graphpancake stats DATABASE_FILE [OPTIONS]

*Optional:*

  - ``--detailed``: Show detailed statistics

.. code-block:: bash
    # Example:
    graphpancake stats molecules.db --detailed

delete
------
Delete graphs from the database, either a single graph by its ID, all graphs, or graphs filtered by type or molecule ID.

.. code-block:: bash

    graphpancake delete [OPTIONS]

*Options:*

  - ``--database, -d PATH``: Database file path (required)
  - ``--graph-id ID``: Delete a single graph by its ID
  - ``--all``: Delete all graphs in the database
  - ``--graph-type TYPE``: Delete graphs by type (DFT, NPA, NBO, QM)
  - ``--mol-id ID``: Delete graphs by molecule ID
  - ``--confirm``: Confirm deletion without prompt

Examples:

.. code-block:: bash

    # Delete a single graph by its ID
    graphpancake delete \
        --database molecules.db \
        --graph-id 42 \
        --confirm

    # Delete all graphs for a specific molecule ID
    graphpancake delete \
        --database molecules.db \
        --mol-id benzaldehyde_DFT \
        --confirm

    # Delete full database, preserving schema
    graphpancake delete \
        --database molecules.db \
        --all \
        --confirm

    # Delete all graphs of a specific type (e.g., NPA)
    graphpancake delete \
        --database molecules.db \
        --graph-type NPA \
        --confirm

    # Delete via where logic - # of atoms
    graphpancake delete \
        --database molecules.db \
        --where "num_atoms < 10" \
        --confirm

add-labels
----------
Add target labels to molecules in the database from a CSV file or individual manual input.

.. code-block:: bash

    graphpancake add-labels [OPTIONS]

*Required:*

  - ``--database, -d PATH``: Database file path
  - Either ``--csv-file PATH`` OR ``--graph-id TEXT`` (mutually exclusive)

*CSV Mode Options:*

  - ``--csv-file PATH``: CSV file containing labels
  - ``--id-column TEXT``: Column name for graph IDs (default: 'graph_id')
  - ``--label-column TEXT``: Column name for target label values (default: 'label')

*Manual Mode Options:*

  - ``--graph-id TEXT``: Graph ID for single label input
  - ``--value FLOAT``: Target label value (required with --graph-id)

*Common Options:*

  - ``--label-name TEXT``: Name for this label type (default: 'label')
  - ``--label-type TEXT``: Label type: 'regression' or 'classification' (default: 'regression')

Examples:

.. code-block:: bash

    # Add binding energy labels from CSV
    graphpancake add-labels \
        --database molecules.db \
        --csv-file binding_energies.csv \
        --label-column "binding_energy" \
        --label-name "binding_energy" \
        --label-type regression

    # Add single logP value manually
    graphpancake add-labels \
        --database molecules.db \
        --graph-id "caffeine_001" \
        --label-name "logP" \
        --value 2.32 \
        --label-type regression

    # Add solubility classification labels from CSV
    graphpancake add-labels \
        --database molecules.db \
        --csv-file solubility.csv \
        --id-column "molecule_id" \
        --label-column "soluble" \
        --label-name "solubility" \
        --label-type classification

    # Simple manual case with default label name
    graphpancake add-labels \
        --database molecules.db \
        --graph-id "benzene_001" \
        --value -6.2

CSV file format example:

.. code-block:: text

    graph_id,binding_energy
    caffeine_001,-8.5
    benzene_001,-6.2
    methanol_001,-4.1

export-ml
----------
Export quantitative graph features and target labels for machine learning applications, in CSV or JSON format.

.. code-block:: bash

    graphpancake export-ml [OPTIONS]

*Required:*

  - ``--database, -d PATH``: Database file path
  - ``--output PATH``: Output directory path

*Optional:*
  - ``--format FORMAT``: Output format (csv, json) (default: csv)
  - ``--graph-type TYPE``: Filter by graph type (DFT, NPA, NBO, QM)

Examples: 

.. code-block:: bash
    
    # Export all features to CSV files
    graphpancake export-ml \
        --database molecules.db \
        --output ml_features/ \
        --graph-type QM \
        --format csv

    # Export all features to JSON files  
    graphpancake export-ml \
        --database molecules.db \
        --output ml_features/ \
        --graph-type QM \
        --format json

logging
-------
Enable verbose logging for troubleshooting:

.. code-block:: bash

    graphpancake --verbose --log-file processing.log load-data ...

Database details & SQLite
=========================

A database can be created and populated via Python and subprocess instead of the CLI:

.. code-block:: python

    from graphpancake.graph_v2 import MolecularGraph

    MolecularGraph.create_database("molecules.db")

Here is a querying example:

.. code-block:: python

    # Get all graphs
    all_graphs = MolecularGraph.query_database("molecules.db")

    # Filter by graph type
    dft_graphs = MolecularGraph.query_database("molecules.db", 
                                              conditions={'graph_type': 'DFT'})

    # Get database summary
    summary = MolecularGraph.get_feature_summary("molecules.db")
    print(f"Graph types: {summary['graph_types']}")
    print(f"Element distribution: {summary['element_distribution']}")

All SQLite commands work as well, i.e.,

.. code-block:: bash

  sqlite3 test.db "SELECT graph_id, electronic_energy, gibbs_free_energy, ZPE FROM targets ORDER BY electronic_energy;"

Here's an example of a Python script using SQLite commands directly:

.. code-block:: python

    import pandas as pd
    import sqlite3

    # Connect to database
    conn = sqlite3.connect("organic_molecules.db")

    # Query all molecules
    molecules = pd.read_sql_query("SELECT * FROM graphs", conn)
    print(f"Total molecules: {len(molecules)}")

    # Query by graph type
    npa_molecules = pd.read_sql_query(
        "SELECT * FROM graphs WHERE graph_type = 'NPA'", 
        conn
    )
    
    # Get detailed node information
    node_data = pd.read_sql_query("""
        SELECT g.mol_id, n.element, n.x, n.y, n.z, n.charge 
        FROM graphs g 
        JOIN nodes n ON g.id = n.graph_id 
        WHERE g.graph_type = 'NPA'
    """, conn)

    conn.close()

Database schema
===============

Database tables adapt based on the type of graph specified, and allow for efficient storage and retrieval of molecular graph data. Export and query commands have been included to make machine-learning post-processing pipelines faster and easier.

graphs table
------------

- ``graph_id`` (TEXT, PRIMARY KEY)
- ``graph_type`` (TEXT) — one of ``DFT``, ``NPA``, ``NBO``, or ``QM``
- ``smiles`` (TEXT)
- ``formula`` (TEXT)
- ``molecular_mass`` (REAL)
- ``num_atoms`` (INTEGER)
- ``num_electrons`` (INTEGER)
- ``charge`` (REAL)
- ``num_bonds`` (INTEGER)
- ``created_at`` (TIMESTAMP)

nodes table
-----------

Common features (all types):

- ``graph_id`` (TEXT)
- ``node_id`` (TEXT)
- ``atomic_index`` (INTEGER)
- ``atomic_number`` (INTEGER)
- ``atom_label`` (TEXT)
- ``x_position`` (REAL)
- ``y_position`` (REAL)
- ``z_position`` (REAL)
- ``atomic_mass`` (REAL)
- ``electronegativity`` (REAL)
- ``covalent_radius`` (REAL)

NPA-specific features:

- ``wiberg_bond_order_total`` (REAL)
- ``bound_hydrogens`` (INTEGER)
- ``node_degree`` (INTEGER)
- ``electron_population`` (REAL)
- ``nmb_population`` (REAL)
- ``npa_charge`` (REAL)

NBO-specific features:

- ``natural_charge`` (REAL)
- ``core_population`` (REAL)
- ``valence_population`` (REAL)
- ``rydberg_population`` (REAL)
- ``total_population`` (REAL)
- ``core_orbital_occupancy`` (REAL)
- ``core_orbital_energy`` (REAL)
- ``lone_pair_occupancy`` (REAL)
- ``lone_pair_energy`` (REAL)

edges table
-----------

Common features:

- ``edge_id`` (TEXT)
- ``graph_id`` (TEXT)
- ``atom_i`` (INTEGER)
- ``atom_j`` (INTEGER)
- ``distance`` (REAL)
- ``edge_type`` (TEXT)

NPA-specific features:

- ``num_2C_BDs`` (INTEGER)

NBO-specific features:

- ``bond_order`` (REAL)
- ``conventional_bond_order`` (REAL)
- ``bonding_orbital_occupancy`` (REAL)
- ``bonding_orbital_energy`` (REAL)
- ``antibonding_orbital_occupancy`` (REAL)
- ``antibonding_orbital_energy`` (REAL)

targets table
-------------

- ``graph_id`` (TEXT)
- ``frequencies`` (TEXT)
- ``num_frequencies`` (INTEGER)
- ``lowest_frequency`` (REAL)
- ``highest_frequency`` (REAL)
- ``moment_1`` (REAL)
- ``moment_2`` (REAL)
- ``moment_3`` (REAL)
- ``rot_1`` (REAL)
- ``rot_2`` (REAL)
- ``rot_3`` (REAL)
- ``rot_temp_1`` (REAL)
- ``rot_temp_2`` (REAL)
- ``rot_temp_3`` (REAL)
- ``heat_capacity_Cv`` (REAL)
- ``heat_capacity_Cp`` (REAL)
- ``entropy`` (REAL)
- ``ZPE`` (REAL)
- ``electronic_energy`` (REAL)
- ``potential_energy`` (REAL)
- ``potential_energy_correction`` (REAL)
- ``enthalpy`` (REAL)
- ``enthalpy_correction`` (REAL)
- ``gibbs_free_energy`` (REAL)
- ``gibbs_free_energy_correction`` (REAL)

NBO-specific features:
- ``natural_minimal_basis`` (REAL)
- ``natural_rydberg_basis`` (REAL)
- ``total_core_population`` (REAL)
- ``total_valence_population`` (REAL)
- ``total_rydberg_population`` (REAL)
- ``total_population`` (REAL)

Batch processing
================

Batch processing using CLI and Python subprocess
-----------------------------------------------

Here's an example of creating and populating a database through a Python script using subprocess calls to the CLI commands.

.. code-block:: python

    import sqlite3
    from pathlib import Path
    from graphpancake.graph import MolecularGraph

    # Create database using CLI
    import subprocess
    subprocess.run([
        "python", "-m", "graphpancake.cli", "create-db", 
        "organic_molecules.db", "--graph-types", "NPA", "QM"
    ])

    # Process multiple molecules
    data_dir = Path("qm_calculations/")
    molecules = [
        {"id": "methanol_001", "smiles": "CO"},
        {"id": "ethanol_001", "smiles": "CCO"},
        {"id": "propanol_001", "smiles": "CCCO"}
    ]

    for mol in molecules:
        mol_id = mol["id"]
        
        # Define file paths
        files = {
            'xyz': data_dir / f"{mol_id}.xyz",
            'shermo': data_dir / f"{mol_id}_shermo.out",
            'janpa': data_dir / f"{mol_id}_janpa.out"
        }
        
        if all(f.exists() for f in files.values()):
            # Process and store
            qm_data = generate_qm_data_dict(files)
            dict_data = DictData(qm_data)
            mol_graph = MolecularGraph(dict_data)
            
            # Save to database
            mol_graph.save_to_database(
                "organic_molecules.db", 
                mol_id, 
                "NPA"
            )
            print(f"Stored {mol_id}")
        else:
            print(f"Missing files for {mol_id}")

Batch processing using configurable script
-----------------------------------------

For processing large datasets, use the provided batch processing script. Processing options include directory-based or gzip-file based. See the :doc:`user_guide` for detailed instructions.

Configuration steps:

Copy the template and customize the configuration paths by following the instructions in the template file:

.. code-block:: bash

    cp config_template.yaml my_config.yaml

Run the batch processing script:

.. code-block:: bash

    python -m graphpancake.batch_processing --config my_config.yaml

CSV format:
Required columns:
- `mol_id`: Unique molecule identifier
- `smiles`: SMILES string
- Target properties (energy, homo, lumo, etc.)

Performance tips
----------------

1. Use parallel processing for batch operations
2. Process in chunks to manage memory usage
3. Clear file cache periodically for large datasets
4. Use SSD storage for database files

Machine learning featurization methods and workflows
------------------------------------------------------------------

graphpancake provides specialized methods for extracting machine learning-ready features from molecular graphs. These methods filter out non-quantitative data and return only numeric features via either scripting or the CLI. Pytorch Geometric and NetworkX integration is also available.

Workflow examples are provided below.

ML workflow via CLI:
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    # 1. Create database for ML
    graphpancake create-db ml_molecules.db --graph-types QM

    # 2. Load molecular data
    graphpancake load-data \
        --database ml_molecules.db \
        --mol-id "caffeine_001" \
        --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
        --xyz-file caffeine.xyz \
        --shermo-output caffeine_shermo.txt \
        --janpa-output caffeine.JANPA \
        --nbo-output caffeine.nbo \
        --graph-type QM

    # 3. Query and inspect the data
    graphpancake query --database ml_molecules.db --format json

    # 4. Filter molecules for specific ML criteria and export to CSV
    graphpancake filter \
        --database ml_molecules.db \
        --where "num_atoms BETWEEN 10 AND 50" \
        --output ml_candidates.csv \
        --format csv

    # 5. Add target labels for ML training (CSV method)
    graphpancake add-labels \
        --database ml_molecules.db \
        --csv-file binding_energies.csv \
        --label-column "binding_energy" \
        --label-name "binding_energy"

    # Alternative: Add individual labels manually
    graphpancake add-labels \
        --database ml_molecules.db \
        --graph-id "caffeine_001" \
        --label-name "logP" \
        --value 1.8 \
        --label-type regression

    # 6. Export ML features to CSV files
    graphpancake --filter \
        --database ml_molecules.db \
        --output ml_data.csv \
        --format csv

ML workflow via scripting:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from graphpancake.graph import GraphDatabase
    
    # Initialize database interface
    db = GraphDatabase("molecules.db")
    
    # Add labels from CSV file
    db.add_labels_from_csv(
        "targets.csv",
        id_column="graph_id",
        target_column="energy",
        label_name="binding_energy",
        label_type="regression"
    )
    
    # Get graphs and labels for ML training
    graphs, labels = db.get_ml_dataset(
        label_name="binding_energy",
        graph_type="QM"
    )
    
    # Convert to PyTorch Geometric dataset
    data_list = []
    for graph, label in zip(graphs, labels):
        data = graph.to_pytorch_geometric()
        data.y = torch.tensor([label], dtype=torch.float)
        data_list.append(data)

    # Export specific feature level to CSV
    GraphDatabase.export_features_csv(
        db_path="molecules.db",
        output_path="graph_features.csv",
        feature_level="graph"
    )

Feature summary
^^^^^^^^^^^^^^^^

Analyze feature distributions across your dataset:

.. code-block:: python

    summary = MolecularGraph.get_feature_summary("molecules.db")
    
    print(f"Graph types: {summary['graph_types']}")
    print(f"Molecular mass range: {summary['molecular_mass']}")
    print(f"Atom count distribution: {summary['atom_counts']}")
    print(f"Element distribution: {summary['element_distribution']}")
    
    # Individual graph summary
    graph_summary = mol_graph.summary()
    print(f"Atom composition: {graph_summary['atom_composition']}")
    print(f"Average bond order: {graph_summary.get('avg_bond_order', 'N/A')}")

PyTorch Geometric integration
^^^^^^^^^^^^^^^^^^^^^^^^

Convert molecular graphs to PyTorch Geometric format for Graph Neural Networks:

.. code-block:: python

    # Convert single graph to PyTorch Geometric Data object
    data = mol_graph.to_pytorch_geometric()
    
    # With target labels from DataFrame
    labels_df = pd.DataFrame({
        'graph_id': ['mol_001', 'mol_002'],
        'target': [1.5, 2.3]
    })
    data = mol_graph.to_pytorch_geometric(labels_df)
    
    # Data object contains:
    # data.x - node features tensor
    # data.edge_index - edge connectivity tensor
    # data.edge_attr - edge features tensor
    # data.y - target labels tensor (if provided)
    # data.graph_attr - graph-level features tensor

NetworkX integration
^^^^^^^^^^^^^^^^^^^^^

Convert to NetworkX graphs:

.. code-block:: python

    # Convert to NetworkX graph
    G = mol_graph.to_networkx()
    
    # NetworkX graph includes:
    # - Nodes with all molecular features as attributes
    # - Edges with bond information as attributes
    # - Graph-level attributes in G.graph


Visualization
=============

3D Molecular Visualization
-------------------------

.. code-block:: python

    # Interactive 3D plot with Plotly
    mol_graph.visualize_graph(
        output_file="molecule_3d.html",
        show_labels=True,
        color_by_element=True
    )

2D Network Visualization
-----------------------

.. code-block:: python

    # 2D network plot with matplotlib
    mol_graph.visualize_graph_2d(
        output_file="molecule_2d.png",
        layout="spring",
        node_size=500
    )