API Documentation
=================

This section provides detailed API reference for all graphpancake modules.

Core Classes
------------

.. automodule:: graphpancake.classes
   :members:
   :undoc-members:
   :show-inheritance:

Data Dictionary Class
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: graphpancake.classes.DictData
   :members:
   :undoc-members:
   :show-inheritance:

Molecular Graph
---------------

.. automodule:: graphpancake.graph
   :members:
   :undoc-members:
   :show-inheritance:

MolecularGraph Class
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: graphpancake.graph.MolecularGraph
   :members:
   :undoc-members:
   :show-inheritance:

Processing Functions
--------------------

.. automodule:: graphpancake.functions
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: graphpancake.functions.generate_qm_data_dict

.. autofunction:: graphpancake.functions.parse_shermo_output

.. autofunction:: graphpancake.functions.parse_janpa_output

.. autofunction:: graphpancake.functions.parse_nbo_output

Command Line Interface
----------------------

.. automodule:: graphpancake.cli
   :members:
   :undoc-members:
   :show-inheritance:

CLI Commands
^^^^^^^^^^^^

.. autoclass:: graphpancake.cli.graphpancakeCLI
   :members:
   :undoc-members:
   :show-inheritance:

Database Operations
-------------------

Database Schema
^^^^^^^^^^^^^^^

graphpancake creates the following SQLite tables:

**graphs table:**
  - mol_id: Unique molecule identifier
  - graph_type: Type of analysis (DFT, NPA, NBO, QM)
  - smiles: SMILES string representation
  - formula: Molecular formula
  - num_atoms: Number of atoms
  - num_edges: Number of bonds
  - created_at: Timestamp

**nodes table:**
  - graph_id: Foreign key to graphs table
  - node_id: Node identifier within graph
  - atomic_number: Atomic number
  - element: Element symbol
  - x, y, z: Atomic coordinates
  - charge: Atomic charge (if available)
  - population: Natural population (if available)

**edges table:**
  - graph_id: Foreign key to graphs table
  - edge_id: Edge identifier within graph
  - atom_i, atom_j: Connected atom indices
  - bond_order: Wiberg bond order
  - distance: Interatomic distance

**targets table:**
  - graph_id: Foreign key to graphs table
  - property_name: Name of target property
  - property_value: Numerical value
  - property_units: Units (if applicable)


.. autosummary::
   :toctree: autosummary
   :recursive:

   graphpancake.cli
   graphpancake.classes
   graphpancake.graph
   graphpancake.functions
