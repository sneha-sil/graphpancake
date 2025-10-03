from graphpancake.classes import DictData, Node, Edge, GraphInfo, Targets
from graphpancake.functions import *
import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Optional imports for visualization and ML frameworks
try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import networkx as nx
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MolecularGraph:
    
    def __init__(self, qm_data: DictData):
        """
        Initialize molecular graph from quantum mechanical data.
        
        Args:
            qm_data (DictData): QM data object
        """
        self._qm_data = qm_data
        self.graph_info = GraphInfo(qm_data)
        self.nodes = Node(qm_data) if qm_data.num_atoms else None
        self.edges = Edge(qm_data) if qm_data.num_atoms else None
        self.targets = Targets(qm_data)
        
        self._node_features = None
        self._edge_features = None
        self._graph_features = None
        self._target_features = None
        
        # Validate structure on creation
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate molecular graph structure consistency."""
        if not self.nodes:
            logger.warning(f"Graph {self.id} has no atoms")
            return
            
        node_features = self.get_node_features()
        edge_features = self.get_edge_features()
        
        # Validate node consistency
        if node_features:
            first_keys = set(node_features[0].keys())
            for i, node in enumerate(node_features[1:], 1):
                if set(node.keys()) != first_keys:
                    missing = first_keys - set(node.keys())
                    for key in missing:
                        node[key] = None
                    logger.warning(f"Padded missing keys {missing} for node {i}")
        
        # Validate edge indices
        num_nodes = len(node_features) if node_features else 0
        for edge in edge_features:
            i, j = edge.get('atom_i', 0), edge.get('atom_j', 0)
            if not (0 <= i < num_nodes and 0 <= j < num_nodes):
                logger.warning(f"Invalid edge indices ({i}, {j}) for {num_nodes} nodes")
    
    @property
    def id(self) -> Optional[str]:
        """Graph identifier."""
        return self.graph_info.id
    
    @property 
    def num_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return self._qm_data.num_atoms or 0
    
    @property
    def num_bonds(self) -> int:
        """Number of bonds in the molecule."""
        edge_features = self.get_edge_features()
        return len(edge_features) if edge_features else 0
    
    def get_node_features(self, refresh: bool = False) -> List[Dict]:
        """
        Get node features with caching.
        
        Args:
            refresh (bool): Force refresh of cached data
            
        Returns:
            List[Dict]: List of node feature dictionaries
        """
        if self._node_features is None or refresh:
            self._node_features = self.nodes.get_node_features() if self.nodes else []
        return self._node_features
    
    def get_node_features_ML(self) -> List[Dict]:
        """
        Get node features suitable for machine learning.
        
        Returns:
            List[Dict]: List of node feature dictionaries for ML
        """
        if not self.nodes:
            return []
        return self.nodes.get_node_features_ML()
    
    def get_edge_features(self, refresh: bool = False) -> List[Dict]:
        """
        Get edge features with caching.
        
        Args:
            refresh (bool): Force refresh of cached data
            
        Returns:
            List[Dict]: List of edge feature dictionaries
        """
        if self._edge_features is None or refresh:
            self._edge_features = self.edges.get_edge_features() if self.edges else []
        return self._edge_features
    
    def get_edge_features_ML(self) -> List[Dict]:
        """
        Get edge features suitable for machine learning.
        
        Returns:
            List[Dict]: List of edge feature dictionaries for ML
        """
        if not self.edges:
            return []
        return self.edges.get_edge_features_ML()
    
    def get_graph_features(self, refresh: bool = False) -> Dict:
        """
        Get graph-level features with caching.
        
        Args:
            refresh (bool): Force refresh of cached data
            
        Returns:
            Dict: Graph feature dictionary
        """
        if self._graph_features is None or refresh:
            self._graph_features = self.graph_info.get_graph_features()
        return self._graph_features
    
    def get_graph_features_ML(self) -> Dict:
        """
        Get graph-level features suitable for machine learning.
        
        Returns:
            Dict: Graph feature dictionary for ML
        """
        if not self.graph_info:
            return {}
        return self.graph_info.get_graph_features_ML()
    
    def get_target_features(self, refresh: bool = False) -> Dict:
        """
        Get target features with caching.
        
        Args:
            refresh (bool): Force refresh of cached data
            
        Returns:
            Dict: Target feature dictionary
        """
        if self._target_features is None or refresh:
            self._target_features = self.targets.get_targets()
        return self._target_features
    
    def get_target_features_ML(self) -> Dict:
        """
        Get target features suitable for machine learning.
        
        Returns:
            Dict: Target feature dictionary for ML
        """
        if not self.targets:
            return {}
        return self.targets.get_targets_ML()
    
    def get_molecular_data(self) -> Dict:
        """
        Get complete molecular data structure.
        
        Returns:
            Dict: Complete molecular data with graph, nodes, edges, and targets
        """
        return {
            'graph_id': self.id,
            'graph_info': self.get_graph_features(),
            'nodes': self.get_node_features(), 
            'edges': self.get_edge_features(),
            'targets': self.get_target_features(),
            'num_atoms': self.num_atoms,
            'num_bonds': self.num_bonds
        }
    
    def get_ML_features(self) -> Dict:
        """
        Produces features suitable for machine learning tasks.
        
        Returns:
            Dict: dictionary of quantitative features.
        """
        return {
            'graph_id': self.id,
            'graph_info': self.get_graph_features_ML(),
            'nodes': self.get_node_features_ML(),
            'edges': self.get_edge_features_ML(),
            'targets': self.get_target_features_ML(),
            'num_atoms': self.num_atoms,
            'num_bonds': self.num_bonds
        }
        
    @classmethod
    def create_database(cls, db_path: Union[str, Path], graph_types: List[str] = None) -> None:
        """
        Create SQLite database with graph-type dependent schema for molecular graphs.
        
        Args:
            db_path: Path to SQLite database file
            graph_types: List of graph types to support ["DFT", "NPA", "NBO", "QM"]. 
                        If None, creates schema supporting all types.
        """
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if graph_types is None:
            graph_types = ["DFT", "NPA", "NBO", "QM"]
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Graphs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graphs (
                graph_id TEXT PRIMARY KEY,
                graph_type TEXT NOT NULL,
                smiles TEXT,
                formula TEXT,
                molecular_mass REAL,
                num_atoms INTEGER,
                num_electrons INTEGER,
                charge REAL,
                num_bonds INTEGER,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Nodes table
        base_node_columns = '''
            node_id TEXT,
            graph_id TEXT,
            atom_index INTEGER,
            atomic_number INTEGER,
            atom_label TEXT,
            x_position REAL,
            y_position REAL, 
            z_position REAL,
            atomic_mass REAL,
            electronegativity REAL,
            covalent_radius REAL
        '''
        
        # graph-specific node columns
        npa_nbo_common_columns = ""
        npa_specific_columns = ""
        nbo_specific_columns = ""
        
        if any(gt in graph_types for gt in ["NPA", "QM"]):
            npa_nbo_common_columns += '''
                wiberg_bond_order_total REAL,
                bound_hydrogens INTEGER,
                node_degree INTEGER
            '''
        
        if "NPA" in graph_types or "QM" in graph_types:
            npa_specific_columns = '''
                electron_population REAL,
                nmb_population REAL,
                npa_charge REAL
            '''
        
        if "NBO" in graph_types or "QM" in graph_types:
            nbo_specific_columns = '''
                natural_charge REAL,
                core_population REAL,
                valence_population REAL,
                rydberg_population REAL,
                total_population REAL,
                core_orbital_occupancy REAL,
                core_orbital_energy REAL,
                lone_pair_1_occupancy REAL,
                lone_pair_1_energy REAL,
                lone_pair_2_occupancy REAL,
                lone_pair_2_energy REAL
            '''
        
        # Combine all node columns
        all_node_columns = base_node_columns
        if npa_nbo_common_columns:
            all_node_columns += "," + npa_nbo_common_columns
        if npa_specific_columns:
            all_node_columns += "," + npa_specific_columns  
        if nbo_specific_columns:
            all_node_columns += "," + nbo_specific_columns
        
        # Add constraints at the end
        all_node_columns += ''',
            PRIMARY KEY (graph_id, atom_index),
            FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
        '''
        
        cursor.execute(f'CREATE TABLE IF NOT EXISTS nodes ({all_node_columns})')
        
        # Edges table
        base_edge_columns = '''
            edge_id TEXT,
            graph_id TEXT,
            atom_i INTEGER,
            atom_j INTEGER,
            distance REAL,
            edge_type TEXT
        '''
        
        # graph-specific edge columns
        edge_specific_columns = ""
        
        if any(gt in graph_types for gt in ["NPA", "NBO", "QM"]):
            edge_specific_columns += '''
                bond_order REAL,
                conventional_bond_order REAL,
                bonding_orbital_occupancy REAL,
                bonding_orbital_energy REAL,
                antibonding_orbital_occupancy REAL,
                antibonding_orbital_energy REAL
            '''
        
        if "NPA" in graph_types or "QM" in graph_types:
            edge_specific_columns += ''',
                num_2C_BDs INTEGER
            '''
        
        # Combine all edge columns
        all_edge_columns = base_edge_columns
        if edge_specific_columns:
            all_edge_columns += "," + edge_specific_columns
        
        # Add constraints at the end
        all_edge_columns += ''',
            PRIMARY KEY (graph_id, atom_i, atom_j),
            FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
        '''
        
        cursor.execute(f'CREATE TABLE IF NOT EXISTS edges ({all_edge_columns})')
        
        # Targets table
        target_columns = '''
            graph_id TEXT,
            frequencies TEXT,  -- JSON array of frequencies
            num_frequencies INTEGER,
            lowest_frequency REAL,
            highest_frequency REAL,
            moment_1 REAL,
            moment_2 REAL,
            moment_3 REAL,
            rot_1 REAL,
            rot_2 REAL,
            rot_3 REAL,
            rot_temp_1 REAL,
            rot_temp_2 REAL,
            rot_temp_3 REAL,
            heat_capacity_Cv REAL,
            heat_capacity_Cp REAL,
            entropy REAL,
            ZPE REAL,
            electronic_energy REAL,
            potential_energy REAL,
            potential_energy_correction REAL,
            enthalpy REAL,
            enthalpy_correction REAL,
            gibbs_free_energy REAL,
            gibbs_free_energy_correction REAL,
        '''
        
        # graph-specific target columns
        if "NBO" in graph_types or "QM" in graph_types:
            target_columns += '''
                natural_minimal_basis REAL,
                natural_rydberg_basis REAL,
                total_core_population REAL,
                total_valence_population REAL,
                total_rydberg_population REAL,
                total_population REAL,
            '''
        
        target_columns += '''
            PRIMARY KEY (graph_id),
            FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
        '''
        
        cursor.execute(f'CREATE TABLE IF NOT EXISTS targets ({target_columns})')
        
        # Labels table - for ML pipeline
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                graph_id TEXT,
                label_name TEXT,
                label_value REAL,
                label_type TEXT,  -- 'regression', 'classification'
                PRIMARY KEY (graph_id, label_name),
                FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
            )
        ''')
        # Create ML views for quantitative data only (excluding categorical features)
        cls._create_ML_views(cursor, graph_types)

        # Create indices for efficient queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_type ON graphs (graph_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_atomic_number ON nodes (atomic_number)')
        
        # Only create bond_order index if bond_order column exists
        if any(gt in graph_types for gt in ["NPA", "NBO", "QM"]):
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bond_order ON edges (bond_order)')
            
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_label_type ON labels (label_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_smiles ON graphs (smiles)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_molecular_mass ON graphs (molecular_mass)')
        
        conn.commit()
        conn.close()
        logger.info(f"\tCreated molecular graph database for {graph_types}: {db_path}")
    
    @classmethod
    def _create_ML_views(cls, cursor, graph_types: List[str]) -> None:
        """
        Create ML-specific views that filter out non-quantitative features.
        
        Args:
            cursor: SQLite cursor
            graph_types: List of supported graph types
        """
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS ml_graphs AS
            SELECT 
                graph_id,
                molecular_mass,
                num_atoms,
                num_electrons,
                charge
            FROM graphs
        ''')
        
        base_ml_node_select = '''
            graph_id,
            atom_index,
            atomic_number,
            x_position,
            y_position,
            z_position,
            atomic_mass,
            electronegativity,
            covalent_radius
        '''
        
        ml_node_select = base_ml_node_select
        if any(gt in graph_types for gt in ["NPA", "QM"]):
            ml_node_select += '''
                , wiberg_bond_order_total,
                bound_hydrogens,
                node_degree
            '''
        
        if "NPA" in graph_types or "QM" in graph_types:
            ml_node_select += '''
                , electron_population,
                nmb_population,
                npa_charge
            '''
        
        if "NBO" in graph_types or "QM" in graph_types:
            ml_node_select += '''
                , natural_charge,
                core_population,
                valence_population,
                rydberg_population,
                total_population,
                core_orbital_occupancy,
                core_orbital_energy,
                lone_pair_1_occupancy,
                lone_pair_1_energy,
                lone_pair_2_occupancy,
                lone_pair_2_energy
            '''
        
        cursor.execute(f'''
            CREATE VIEW IF NOT EXISTS ml_nodes AS
            SELECT {ml_node_select}
            FROM nodes
        ''')
        
        base_ml_edge_select = '''
            graph_id,
            atom_i,
            atom_j,
            distance
        '''
        
        ml_edge_select = base_ml_edge_select
        if any(gt in graph_types for gt in ["NPA", "NBO", "QM"]):
            ml_edge_select += '''
                , bond_order,
                conventional_bond_order,
                bonding_orbital_occupancy,
                bonding_orbital_energy,
                antibonding_orbital_occupancy,
                antibonding_orbital_energy
            '''
        
        if "NPA" in graph_types or "QM" in graph_types:
            ml_edge_select += '''
                , num_2C_BDs
            '''
        
        cursor.execute(f'''
            CREATE VIEW IF NOT EXISTS ml_edges AS
            SELECT {ml_edge_select}
            FROM edges
        ''')
        
        ml_target_select = '''
            graph_id,
            num_frequencies,
            lowest_frequency,
            highest_frequency,
            moment_1, moment_2, moment_3,
            rot_1, rot_2, rot_3,
            rot_temp_1, rot_temp_2, rot_temp_3,
            heat_capacity_Cv,
            heat_capacity_Cp,
            entropy,
            ZPE,
            electronic_energy,
            potential_energy,
            potential_energy_correction,
            enthalpy,
            enthalpy_correction,
            gibbs_free_energy,
            gibbs_free_energy_correction
        '''
        
        if "NBO" in graph_types or "QM" in graph_types:
            ml_target_select += '''
                , natural_minimal_basis,
                natural_rydberg_basis,
                total_core_population,
                total_valence_population,
                total_rydberg_population,
                total_population
            '''
        
        cursor.execute(f'''
            CREATE VIEW IF NOT EXISTS ml_targets AS
            SELECT {ml_target_select}
            FROM targets
        ''')
        
        logger.info("\tCreated ML views for quantitative features only")
    
    def save_to_database(self, db_path: Union[str, Path]) -> None:
        """
        Save molecular graph to SQLite database using the new structured schema.
        
        Args:
            db_path: Path to SQLite database file
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            graph_features = self.get_graph_features()
            node_features = self.get_node_features()
            edge_features = self.get_edge_features()
            target_features = self.get_target_features()
            
            # Insert graph-level data
            cursor.execute('''
                INSERT OR REPLACE INTO graphs 
                (graph_id, graph_type, smiles, formula, molecular_mass, num_atoms, 
                 num_electrons, charge, num_bonds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.id,
                graph_features.get('graph_type'),
                graph_features.get('smiles'),
                graph_features.get('formula'),
                graph_features.get('molecular_mass'),
                graph_features.get('num_atoms'),
                graph_features.get('num_electrons'),
                graph_features.get('charge'),
                self.num_bonds
            ))
            
            # Insert node data with all features
            for node in node_features:
                # Validate and clean node data
                atom_index = node.get('atom_index', 0)
                atomic_number = node.get('atomic_number')
                atom_label = node.get('atom_label')
                
                # Ensure proper types
                try:
                    atom_index = int(atom_index) if atom_index is not None else 0
                    atomic_number = int(atomic_number) if atomic_number is not None else None
                    atom_label = str(atom_label) if atom_label is not None else None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid node data for atom {atom_index}: {e}")
                    continue
                
                # Extract coordinates from position field
                position = node.get('position', [None, None, None])
                x_pos = position[0] if position and len(position) > 0 else None
                y_pos = position[1] if position and len(position) > 1 else None  
                z_pos = position[2] if position and len(position) > 2 else None
                
                # Prepare base node data
                node_data = [
                    f"{self.id}_{atom_index}",  # node_id
                    self.id,  # graph_id
                    atom_index,
                    atomic_number,
                    atom_label,
                    x_pos,
                    y_pos, 
                    z_pos,
                    node.get('atomic_mass'),
                    node.get('electronegativity'),
                    node.get('covalent_radius')
                ]
                
                # Add graph-type specific features
                if self._qm_data.graph_type in ["NPA", "QM"]:
                    node_data.extend([
                        node.get('wiberg_bond_order_total'),
                        node.get('bound_hydrogens'),
                        node.get('node_degree')
                    ])
                else:
                    node_data.extend([None, None, None])
                
                if self._qm_data.graph_type == "NPA" or self._qm_data.graph_type == "QM":
                    node_data.extend([
                        node.get('electron_population'),
                        node.get('nmb_population'),
                        node.get('npa_charge')
                    ])
                else:
                    node_data.extend([None, None, None])
                
                if self._qm_data.graph_type in ["NBO", "QM"]:
                    node_data.extend([
                        node.get('natural_charge'),
                        node.get('core_population'),
                        node.get('valence_population'),
                        node.get('rydberg_population'),
                        node.get('total_population'),
                        node.get('core_orbital_occupancy'),
                        node.get('core_orbital_energy'),
                        node.get('lone_pair_1_occupancy'),
                        node.get('lone_pair_1_energy'),
                        node.get('lone_pair_2_occupancy'),
                        node.get('lone_pair_2_energy')
                    ])
                else:
                    node_data.extend([None] * 11)
                
                # Dynamic query based on graph type
                placeholders = ','.join(['?'] * len(node_data))
                try:
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO nodes VALUES ({placeholders})
                    ''', node_data)
                except Exception as e:
                    logger.error(f"Node insertion error for {self.id}: {e}")
                    logger.error(f"Node data: {node_data}")
                    logger.error(f"Node data types: {[type(x) for x in node_data]}")
                    raise
        
            expected_columns = ['edge_id', 'graph_id', 'atom_i', 'atom_j', 'distance', 'edge_type']
            if self._qm_data.graph_type in ["NPA", "NBO", "QM"]:
                expected_columns.extend(['bond_order', 'conventional_bond_order', 'bonding_orbital_occupancy', 
                                       'bonding_orbital_energy', 'antibonding_orbital_occupancy', 'antibonding_orbital_energy'])
            if self._qm_data.graph_type in ["NPA", "QM"]:
                expected_columns.append('num_2C_BDs')
            
            # logger.info(f"\tGraph type: {self._qm_data.graph_type}")
            # logger.info(f"\tProcessing {len(node_features)} atoms and {len(edge_features)} edges")
            
            for edge_idx, edge in enumerate(edge_features):
                try:
                    # Validate and clean edge data
                    atom_i = edge.get('atom_i')
                    atom_j = edge.get('atom_j')
                    
                    if isinstance(atom_i, (tuple, list)):
                        atom_i = atom_i[0] if len(atom_i) > 0 else 0
                    if isinstance(atom_j, (tuple, list)):
                        atom_j = atom_j[0] if len(atom_j) > 0 else 0
                    
                    try:
                        atom_i = int(atom_i) if atom_i is not None else 0
                        atom_j = int(atom_j) if atom_j is not None else 0
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid atom indices for edge: atom_i={atom_i}, atom_j={atom_j}")
                        continue
                    
                    distance = edge.get('distance')
                    
                    if hasattr(distance, 'item') and callable(distance.item):
                        distance = distance.item()
                    elif isinstance(distance, (np.floating, np.integer)):
                        distance = float(distance)
                    
                    edge_type = edge.get('edge_type')
                    
                    # Prepare base edge data
                    edge_data = [
                        f"{self.id}_{atom_i}_{atom_j}",  # edge_id
                        self.id,  # graph_id
                        atom_i,
                        atom_j,
                        distance,
                        edge_type
                    ]
                except Exception as edge_e:
                    logger.error(f"Failed to process edge {edge_idx} for {self.id}: {edge_e}")
                    logger.error(f"Edge data: {repr(edge)}")
                    # Continue with next edge instead of failing entire molecule
                    continue

                if self._qm_data.graph_type in ["NPA", "NBO", "QM"]:
                    features = [
                        edge.get('bond_order'),
                        edge.get('conventional_bond_order'),
                        edge.get('bonding_orbital_occupancy'),
                        edge.get('bonding_orbital_energy'),
                        edge.get('antibonding_orbital_occupancy'),
                        edge.get('antibonding_orbital_energy')
                    ]
                    
                    processed_features = []
                    for feature in features:
                        processed_features.append(feature)
                    
                    edge_data.extend(processed_features)
                else:
                    edge_data.extend([None] * 6)
                
                if self._qm_data.graph_type in ["NPA", "QM"]:
                    num_2c_bds = edge.get('num_2C_BDs')
                    if isinstance(num_2c_bds, (tuple, list)):
                        num_2c_bds = num_2c_bds[0] if len(num_2c_bds) > 0 else None
                    edge_data.append(num_2c_bds)
                else:
                    edge_data.append(None)
                
                # Dynamic query based on features
                placeholders = ','.join(['?'] * len(edge_data))
                
                # Insert edge into database
                try:
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO edges VALUES ({placeholders})
                    ''', edge_data)
                except Exception as e:
                    logger.error(f"Edge insertion error for {self.id} edge {atom_i}-{atom_j}: {e}")
                    logger.error(f"Edge data: {edge_data}")
                    logger.error(f"Edge data types: {[type(x) for x in edge_data]}")
                    continue
            
            # Insert target data
            target_data = [
                self.id,
                json.dumps(target_features.get('frequencies', [])),
                target_features.get('num_frequencies'),
                target_features.get('lowest_frequency'),
                target_features.get('highest_frequency'),
                target_features.get('moment_1'),
                target_features.get('moment_2'),
                target_features.get('moment_3'),
                target_features.get('rot_1'),
                target_features.get('rot_2'),
                target_features.get('rot_3'),
                target_features.get('rot_temp_1'),
                target_features.get('rot_temp_2'),
                target_features.get('rot_temp_3'),
                target_features.get('heat_capacity_Cv'),
                target_features.get('heat_capacity_Cp'),
                target_features.get('entropy'),
                target_features.get('ZPE'),
                target_features.get('electronic_energy'),
                target_features.get('potential_energy'),
                target_features.get('potential_energy_correction'),
                target_features.get('enthalpy'),
                target_features.get('enthalpy_correction'),
                target_features.get('gibbs_free_energy'),
                target_features.get('gibbs_free_energy_correction')
            ]
            
            if self._qm_data.graph_type in ["NBO", "QM"]:
                target_data.extend([
                    target_features.get('natural_minimal_basis'),
                    target_features.get('natural_rydberg_basis'),
                    target_features.get('total_core_population'),
                    target_features.get('total_valence_population'),
                    target_features.get('total_rydberg_population'),
                    target_features.get('total_population')
                ])
            else:
                target_data.extend([None] * 6)
            
            placeholders = ','.join(['?'] * len(target_data))
            cursor.execute(f'''
                INSERT OR REPLACE INTO targets VALUES ({placeholders})
            ''', target_data)
            
            conn.commit()
            logger.info(f"\tSaved graph {self.id} ({self._qm_data.graph_type}) to database")
            
        except Exception as e:
            logger.error(f"Database transaction failed for {self.id}: {e}")
            logger.exception("Full traceback:")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    @classmethod
    def load_from_database(cls, graph_id: str, db_path: Union[str, Path]) -> 'MolecularGraph':
        """
        Load molecular graph from SQLite database.
        
        Args:
            graph_id: Graph identifier to load
            db_path: Path to SQLite database file
            
        Returns:
            MolecularGraph: Loaded graph object
        """
        conn = sqlite3.connect(db_path)
        
        try:
            # Load graph metadata
            graph_query = "SELECT * FROM graphs WHERE graph_id = ?"
            graph_data = pd.read_sql_query(graph_query, conn, params=[graph_id])
            
            if graph_data.empty:
                raise ValueError(f"Graph {graph_id} not found in database")
            
            graph_row = graph_data.iloc[0]
            graph_type = graph_row['graph_type']
            
            # Load nodes
            nodes_query = "SELECT * FROM nodes WHERE graph_id = ? ORDER BY atom_index"
            nodes_data = pd.read_sql_query(nodes_query, conn, params=[graph_id])
            
            # Load edges  
            edges_query = "SELECT * FROM edges WHERE graph_id = ? ORDER BY atom_i, atom_j"
            edges_data = pd.read_sql_query(edges_query, conn, params=[graph_id])
            
            # Load targets
            targets_query = "SELECT * FROM targets WHERE graph_id = ?"
            targets_data = pd.read_sql_query(targets_query, conn, params=[graph_id])
            
        finally:
            conn.close()
        
        # Reconstruct molecular graph
        molecular_graph = cls()
        
        # Set graph metadata
        molecular_graph.graph_id = graph_id
        molecular_graph.graph_type = graph_type
        
        # Parse JSON fields back to objects
        molecular_graph.dict_data = DictData()
        molecular_graph.dict_data.data = json.loads(graph_row['dict_data'])
        
        molecular_graph.graph_info = GraphInfo()
        graph_info_data = json.loads(graph_row['graph_info'])
        for key, value in graph_info_data.items():
            setattr(molecular_graph.graph_info, key, value)
        
        # Reconstruct nodes
        molecular_graph.nodes = {}
        for _, node_row in nodes_data.iterrows():
            node = Node()
            node.node_id = node_row['node_id']
            node.element = node_row['element'] 
            node.atomic_number = node_row['atomic_number']
            
            # Set direct numeric/text properties based on graph type
            numeric_cols = ['mass', 'x', 'y', 'z', 'partial_charge']
            if graph_type in ['NPA', 'QM']:
                numeric_cols.extend(['wiberg_total', 'npa_charge', 'npa_core', 'npa_valence', 'npa_rydberg'])
            if graph_type in ['NBO', 'QM']:
                numeric_cols.extend(['natural_charge', 'natural_core', 'natural_valence', 'natural_rydberg'])
            
            for col in numeric_cols:
                if col in node_row and pd.notna(node_row[col]):
                    setattr(node, col, node_row[col])
            
            # Parse JSON attributes for complex data structures
            json_cols = []
            if graph_type in ['NPA', 'QM']:
                json_cols.extend(['wiberg_indices', 'npa_spin_densities'])
            if graph_type in ['NBO', 'QM']:
                json_cols.extend(['natural_spin_densities', 'lone_pairs', 'natural_bond_orbitals'])
            
            for col in json_cols:
                if col in node_row and pd.notna(node_row[col]):
                    setattr(node, col, json.loads(node_row[col]))
            
            molecular_graph.nodes[node.node_id] = node
        
        # Reconstruct edges
        molecular_graph.edges = {}
        for _, edge_row in edges_data.iterrows():
            edge = Edge()
            edge.source_id = edge_row['source_id']
            edge.target_id = edge_row['target_id']
            
            # Set numeric properties
            if pd.notna(edge_row['distance']):
                edge.distance = edge_row['distance']
            
            # Set graph-type specific properties
            if graph_type in ['NPA', 'QM'] and 'wiberg_bond_order' in edge_row:
                if pd.notna(edge_row['wiberg_bond_order']):
                    edge.wiberg_bond_order = edge_row['wiberg_bond_order']
            
            if graph_type in ['NBO', 'QM']:
                if 'natural_bond_order' in edge_row and pd.notna(edge_row['natural_bond_order']):
                    edge.natural_bond_order = edge_row['natural_bond_order']
                if 'bond_index' in edge_row and pd.notna(edge_row['bond_index']):
                    edge.bond_index = edge_row['bond_index']
            
            # Parse JSON attributes for complex structures
            json_cols = []
            if graph_type in ['NBO', 'QM']:
                json_cols.extend(['natural_localized_orbitals'])
            
            for col in json_cols:
                if col in edge_row and pd.notna(edge_row[col]):
                    setattr(edge, col, json.loads(edge_row[col]))
            
            edge_key = (edge.source_id, edge.target_id)
            molecular_graph.edges[edge_key] = edge
        
        # Reconstruct targets
        if not targets_data.empty:
            molecular_graph.targets = Targets()
            targets_row = targets_data.iloc[0]
            
            # Set numeric target properties
            numeric_targets = ['energy', 'homo', 'lumo', 'gap', 'dipole_magnitude']
            for col in numeric_targets:
                if col in targets_row and pd.notna(targets_row[col]):
                    setattr(molecular_graph.targets, col, targets_row[col])
            
            # Parse JSON targets for complex structures
            json_targets = ['dipole_vector', 'vibrational_frequencies', 'ir_intensities', 'thermodynamic_properties']
            for col in json_targets:
                if col in targets_row and pd.notna(targets_row[col]):
                    setattr(molecular_graph.targets, col, json.loads(targets_row[col]))
        
        return molecular_graph
    
    def load_labels(self, db_path: Union[str, Path], label_names: Optional[List[str]] = None) -> Dict:
        """
        Load labels for this graph from database.
        
        Args:
            db_path: Path to SQLite database file
            label_names: Specific label names to load, or None for all
            
        Returns:
            Dict: Label name to value mapping
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            if label_names:
                placeholders = ','.join('?' * len(label_names))
                cursor.execute(f'''
                    SELECT label_name, label_value FROM labels 
                    WHERE graph_id = ? AND label_name IN ({placeholders})
                ''', [self.id] + label_names)
            else:
                cursor.execute('SELECT label_name, label_value FROM labels WHERE graph_id = ?', (self.id,))
            
            return dict(cursor.fetchall())
        finally:
            conn.close()
    
    @classmethod
    def query_database(cls, db_path: Union[str, Path], 
                      conditions: Optional[Dict] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Query molecular graphs from database with conditions.
        
        Args:
            db_path: Path to SQLite database file
            conditions: Dict of column:value conditions to filter by
            limit: Maximum number of results to return
            
        Returns:
            pd.DataFrame: Query results
        """
        conn = sqlite3.connect(db_path)
        
        query = "SELECT * FROM graphs"
        params = []
        
        if conditions:
            where_clauses = []
            for col, val in conditions.items():
                where_clauses.append(f"{col} = ?")
                params.append(val)
            query += " WHERE " + " AND ".join(where_clauses)
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    @classmethod
    def get_ml_data(cls, db_path: Union[str, Path], 
                   graph_type: Optional[str] = None,
                   graph_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Get ML-ready data (quantitative features only) from database views.
        
        Args:
            db_path: Path to SQLite database file
            graph_type: Filter by specific graph type
            graph_ids: Filter by specific graph IDs
            
        Returns:
            Dict containing DataFrames for graphs, nodes, edges, and targets
        """
        conn = sqlite3.connect(db_path)
        
        try:
            where_conditions = []
            params = []
            
            if graph_type:
                where_conditions.append("graph_type = ?")
                params.append(graph_type)
            
            if graph_ids:
                placeholders = ','.join('?' * len(graph_ids))
                where_conditions.append(f"graph_id IN ({placeholders})")
                params.extend(graph_ids)
            
            where_clause = ""
            if where_conditions:
                where_clause = " WHERE " + " AND ".join(where_conditions)
            
            # Query ML views
            ml_data = {}
            
            # Get graph-level ML features
            ml_data['graphs'] = pd.read_sql_query(
                f"SELECT * FROM ml_graphs{where_clause}", 
                conn, params=params
            )
            
            # Get node-level ML features
            if graph_ids or graph_type:
                ml_data['nodes'] = pd.read_sql_query(
                    f"SELECT * FROM ml_nodes{where_clause}", 
                    conn, params=params
                )
            else:
                ml_data['nodes'] = pd.read_sql_query("SELECT * FROM ml_nodes", conn)
            
            # Get edge-level ML features  
            if graph_ids or graph_type:
                ml_data['edges'] = pd.read_sql_query(
                    f"SELECT * FROM ml_edges{where_clause}", 
                    conn, params=params
                )
            else:
                ml_data['edges'] = pd.read_sql_query("SELECT * FROM ml_edges", conn)
            
            # Get target ML features
            ml_data['targets'] = pd.read_sql_query(
                f"SELECT * FROM ml_targets{where_clause}", 
                conn, params=params
            )
            
            return ml_data
            
        finally:
            conn.close()
    
    @classmethod 
    def get_feature_summary(cls, db_path: Union[str, Path]) -> Dict:
        """
        Get summary statistics of features in the database.
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            Dict: Feature summary statistics
        """
        conn = sqlite3.connect(db_path)
        
        try:
            summary = {}
            
            # Graph type distribution
            graph_types = pd.read_sql_query(
                "SELECT graph_type, COUNT(*) as count FROM graphs GROUP BY graph_type", 
                conn
            )
            summary['graph_types'] = dict(zip(graph_types['graph_type'], graph_types['count']))
            
            # Molecular mass distribution
            mass_stats = pd.read_sql_query(
                "SELECT MIN(molecular_mass) as min_mass, MAX(molecular_mass) as max_mass, AVG(molecular_mass) as avg_mass FROM graphs", 
                conn
            ).iloc[0]
            summary['molecular_mass'] = mass_stats.to_dict()
            
            # Atom count distribution
            atom_stats = pd.read_sql_query(
                "SELECT MIN(num_atoms) as min_atoms, MAX(num_atoms) as max_atoms, AVG(num_atoms) as avg_atoms FROM graphs", 
                conn
            ).iloc[0]
            summary['atom_counts'] = atom_stats.to_dict()
            
            # Atomic number distribution
            element_dist = pd.read_sql_query(
                "SELECT atomic_number, COUNT(*) as count FROM nodes GROUP BY atomic_number ORDER BY count DESC", 
                conn
            )
            summary['element_distribution'] = dict(zip(element_dist['atomic_number'], element_dist['count']))
            
            # Bond order distribution (if available)
            try:
                bond_stats = pd.read_sql_query(
                    "SELECT MIN(bond_order) as min_bo, MAX(bond_order) as max_bo, AVG(bond_order) as avg_bo FROM edges WHERE bond_order IS NOT NULL", 
                    conn
                ).iloc[0]
                summary['bond_orders'] = bond_stats.to_dict()
            except:
                summary['bond_orders'] = None
                
            return summary
            
        finally:
            conn.close()
    
    @classmethod
    def export_ml_features(cls, db_path: Union[str, Path], 
                          output_dir: Union[str, Path],
                          graph_type: Optional[str] = None) -> None:
        """
        Export ML-ready features to CSV files.
        
        Args:
            db_path: Path to SQLite database file
            output_dir: Directory to save CSV files
            graph_type: Optional graph type filter
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ml_data = cls.get_ml_data(db_path, graph_type=graph_type)
        
        # Export each feature level
        for level, df in ml_data.items():
            if not df.empty:
                filename = f"ml_{level}"
                if graph_type:
                    filename += f"_{graph_type}"
                filename += ".csv"
                
                filepath = output_dir / filename
                df.to_csv(filepath, index=False)
                logger.info(f"\tExported {len(df)} {level} features to {filepath}")
        
        # Export combined dataset for easy loading
        if all(not df.empty for df in ml_data.values()):
            combined_filename = "ml_combined"
            if graph_type:
                combined_filename += f"_{graph_type}"
            combined_filename += ".pkl"
            
            combined_path = output_dir / combined_filename
            with open(combined_path, 'wb') as f:
                import pickle
                pickle.dump(ml_data, f)
            logger.info(f"\tExported combined ML dataset to {combined_path}")
    
    def to_pytorch_geometric(self, labels_df: Optional[pd.DataFrame] = None) -> Optional['Data']:
        """
        Convert to PyTorch Geometric Data object for GNN training.
        
        Args:
            labels_df: DataFrame with graph_id and target columns
            
        Returns:
            Data: PyTorch Geometric data object or None if torch unavailable
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, cannot create Data object")
            return None
        
        ml_data = self.get_ml_features()
        
        # Node features tensor
        node_features = []
        for node in ml_data['node_features']:
            features = [float(v) if not np.isnan(v) else 0.0 for v in node.values()]
            node_features.append(features)
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge connectivity and features
        edge_indices = []
        edge_features = []
        for edge in ml_data['edge_features']:
            edge_copy = edge.copy()  # Don't modify original
            i, j = int(edge_copy.pop('atom_i', 0)), int(edge_copy.pop('atom_j', 0))
            features = [float(v) if not np.isnan(v) else 0.0 for v in edge_copy.values()]
            
            # Add both directions for undirected graph
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([features, features])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).T if edge_indices else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.zeros((0, 1))
        
        # Graph-level features
        graph_features = [float(v) if not np.isnan(v) else 0.0 for v in ml_data['graph_features'].values()]
        graph_attr = torch.tensor(graph_features, dtype=torch.float) if graph_features else None
        
        # Target labels if provided
        y = None
        if labels_df is not None and self.id in labels_df['graph_id'].values:
            target_row = labels_df[labels_df['graph_id'] == self.id]
            if not target_row.empty:
                y = torch.tensor([float(target_row.iloc[0]['target'])], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_attr=graph_attr)
    
    def to_networkx(self) -> Optional['nx.Graph']:
        """
        Convert to NetworkX graph for analysis.
        
        Returns:
            nx.Graph: NetworkX graph object or None if networkx unavailable
        """
        if not HAS_VISUALIZATION:
            logger.warning("NetworkX not available")
            return None
        
        G = nx.Graph()
        
        # Add nodes with features
        for node in self.get_node_features():
            node_id = node.get('atom_index', 0)
            G.add_node(node_id, **node)
        
        # Add edges with features  
        for edge in self.get_edge_features():
            i, j = edge.get('atom_i', 0), edge.get('atom_j', 0)
            G.add_edge(i, j, **edge)
        
        # Add graph-level attributes
        G.graph.update(self.get_graph_features())
        
        return G
        
    def summary(self) -> Dict:
        """
        Get molecular graph summary statistics.
        
        Returns:
            Dict: Summary statistics
        """
        node_features = self.get_node_features()
        edge_features = self.get_edge_features()
        
        summary = {
            'graph_id': self.id,
            'num_atoms': len(node_features),
            'num_bonds': len(edge_features),
            'graph_type': self._qm_data.graph_type,
            'formula': self.get_graph_features().get('formula'),
            'molecular_mass': self.get_graph_features().get('molecular_mass')
        }
        
        if node_features:
            atom_types = {}
            for node in node_features:
                atom_label = node.get('atom_label', 'Unknown')
                atom_types[atom_label] = atom_types.get(atom_label, 0) + 1
            summary['atom_composition'] = atom_types
        
        if edge_features:
            bond_orders = [edge.get('wiberg_bond_order', 0) for edge in edge_features if edge.get('wiberg_bond_order') is not None]
            if bond_orders:
                summary['avg_bond_order'] = np.mean(bond_orders)
                summary['max_bond_order'] = np.max(bond_orders)
        
        return summary
    
    @classmethod
    def delete_graph(cls, graph_id: str, db_path: Union[str, Path]) -> bool:
        """
        Delete a specific molecular graph and all its associated data from the database.
        
        Args:
            graph_id: ID of the graph to delete
            db_path: Path to SQLite database file
            
        Returns:
            bool: True if graph was deleted, False if graph was not found
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Check if graph exists
            cursor.execute("SELECT COUNT(*) FROM graphs WHERE graph_id = ?", (graph_id,))
            count = cursor.fetchone()[0]
            
            if count == 0:
                logger.warning(f"Graph '{graph_id}' not found in database")
                return False
            
            # Delete from all tables (foreign key constraints will handle cascading)
            # Delete edges first
            cursor.execute("DELETE FROM edges WHERE graph_id = ?", (graph_id,))
            edges_deleted = cursor.rowcount
            
            # Delete nodes
            cursor.execute("DELETE FROM nodes WHERE graph_id = ?", (graph_id,))
            nodes_deleted = cursor.rowcount
            
            # Delete target features
            cursor.execute("DELETE FROM targets WHERE graph_id = ?", (graph_id,))
            targets_deleted = cursor.rowcount
            
            # Delete graph
            cursor.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
            graphs_deleted = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"Successfully deleted graph '{graph_id}': "
                       f"{graphs_deleted} graph, {nodes_deleted} nodes, "
                       f"{edges_deleted} edges, {targets_deleted} targets")
            
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting graph '{graph_id}': {e}")
            raise
        finally:
            conn.close()
    
    @classmethod
    def delete_graphs_by_criteria(cls, db_path: Union[str, Path], 
                                 where_clause: str = None,
                                 parameters: tuple = None) -> int:
        """
        Delete graphs matching specific criteria.
        
        Args:
            db_path: Path to SQLite database file
            where_clause: SQL WHERE clause (without 'WHERE' keyword)
            parameters: Parameters for the WHERE clause
            
        Returns:
            int: Number of graphs deleted
            
        Examples:
            # Delete graphs with energy > -100
            delete_graphs_by_criteria(db_path, "json_extract(graph_features, '$.total_energy') > ?", (-100,))
            
            # Delete graphs of specific molecule type
            delete_graphs_by_criteria(db_path, "json_extract(graph_features, '$.molecule_type') = ?", ('alkane',))
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        if not where_clause:
            raise ValueError("where_clause is required for safety. Use clear_database() to delete all.")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # First get the graph IDs that match the criteria
            query = f"SELECT graph_id FROM graphs WHERE {where_clause}"
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            graph_ids = [row[0] for row in cursor.fetchall()]
            
            if not graph_ids:
                logger.info("No graphs match the specified criteria")
                return 0
            
            # Delete each graph
            deleted_count = 0
            for graph_id in graph_ids:
                # Delete edges
                cursor.execute("DELETE FROM edges WHERE graph_id = ?", (graph_id,))
                
                # Delete nodes
                cursor.execute("DELETE FROM nodes WHERE graph_id = ?", (graph_id,))
                
                # Delete target features
                cursor.execute("DELETE FROM targets WHERE graph_id = ?", (graph_id,))
                
                # Delete graph
                cursor.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
                deleted_count += 1
            
            conn.commit()
            
            logger.info(f"Successfully deleted {deleted_count} graphs matching criteria")
            return deleted_count
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting graphs by criteria: {e}")
            raise
        finally:
            conn.close()
    
    @classmethod
    def clear_database(cls, db_path: Union[str, Path], confirm: bool = False) -> int:
        """
        Remove all data from the database while keeping the schema.
        
        Args:
            db_path: Path to SQLite database file
            confirm: Must be True to actually clear the database (safety check)
            
        Returns:
            int: Total number of records deleted
        """
        if not confirm:
            raise ValueError("confirm=True is required to clear the database")
        
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Count total records before deletion
            cursor.execute("SELECT COUNT(*) FROM graphs")
            graph_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM targets")
            target_count = cursor.fetchone()[0]
            
            total_records = graph_count + node_count + edge_count + target_count
            
            if total_records == 0:
                logger.info("Database is already empty")
                return 0
            
            # Delete all data
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM nodes") 
            cursor.execute("DELETE FROM targets")
            cursor.execute("DELETE FROM graphs")
            
            conn.commit()
            
            logger.info(f"Successfully cleared database: {graph_count} graphs, "
                       f"{node_count} nodes, {edge_count} edges, {target_count} targets")
            
            return total_records
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error clearing database: {e}")
            raise
        finally:
            conn.close()
    
    def __repr__(self) -> str:
        """String representation of molecular graph."""
        return f"MolecularGraph(id='{self.id}', atoms={self.num_atoms}, bonds={self.num_bonds})"
    
    def __str__(self) -> str:
        """Detailed string representation."""
        summary = self.summary()
        composition = summary.get('atom_composition', {})
        comp_str = ', '.join([f"{count}{atom}" for atom, count in composition.items()])
        return f"MolecularGraph {self.id}: {comp_str} ({self.num_atoms} atoms, {self.num_bonds} bonds)"


class GraphDatabase:
    """Utility class for batch operations on molecular graph databases."""
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize database interface.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            MolecularGraph.create_database(self.db_path)
    
    def add_labels_from_csv(self, csv_path: Union[str, Path], 
                           id_column: str = 'graph_id',
                           target_column: str = 'target',
                           label_name: str = 'target',
                           label_type: str = 'regression') -> None:
        """
        Add labels to database from CSV file.
        
        Args:
            csv_path: Path to CSV file with labels
            id_column: Column name containing graph IDs
            target_column: Column name containing target values  
            label_name: Name for this label type
            label_type: 'regression' or 'classification'
        """
        # Use polars if available, otherwise pandas
        if HAS_POLARS:
            df = pl.scan_csv(csv_path).select([id_column, target_column]).collect()
            rows = df.rows()
        else:
            df = pd.read_csv(csv_path)[[id_column, target_column]]
            rows = df.itertuples(index=False, name=None)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            count = 0
            for row in rows:
                graph_id, target_value = row
                cursor.execute('''
                    INSERT OR REPLACE INTO labels (graph_id, label_name, label_value, label_type)
                    VALUES (?, ?, ?, ?)
                ''', (graph_id, label_name, float(target_value), label_type))
                count += 1
            
            conn.commit()
            logger.info(f"\tAdded {count} labels from {csv_path}")
        finally:
            conn.close()
    
    def get_ml_dataset(self, label_name: str = 'target',
                      graph_type: Optional[str] = None) -> Tuple[List['MolecularGraph'], List[float]]:
        """
        Get molecular graphs and labels for ML training.
        
        Args:
            label_name: Name of label to retrieve
            graph_type: Filter by graph type (optional)
            
        Returns:
            Tuple of (graphs, labels) lists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Query for graphs with labels
            query = '''
                SELECT g.graph_id, l.label_value 
                FROM graphs g 
                JOIN labels l ON g.graph_id = l.graph_id 
                WHERE l.label_name = ?
            '''
            params = [label_name]
            
            if graph_type:
                query += ' AND g.graph_type = ?'
                params.append(graph_type)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            graphs = []
            labels = []
            
            for graph_id, label_value in results:
                try:
                    graph = MolecularGraph.load_from_database(graph_id, self.db_path)
                    graphs.append(graph)
                    labels.append(float(label_value))
                except Exception as e:
                    logger.warning(f"Failed to load graph {graph_id}: {e}")
            
            return graphs, labels
            
        finally:
            conn.close()
    
    @classmethod
    def export_features_csv(cls, db_path: Union[str, Path], output_path: Union[str, Path],
                           feature_level: str = 'graph') -> None:
        """
        Export features to CSV for external ML tools.
        
        Args:
            db_path: Path to SQLite database file
            output_path: Path for output CSV file
            feature_level: 'graph', 'node', or 'edge'
        """
        conn = sqlite3.connect(db_path)
        
        if feature_level == 'graph':
            query = "SELECT graph_id, graph_features FROM graphs"
        elif feature_level == 'node':
            query = "SELECT graph_id, atom_index, node_features FROM nodes"
        elif feature_level == 'edge':
            query = "SELECT graph_id, atom_i, atom_j, edge_features FROM edges"
        else:
            raise ValueError("feature_level must be 'graph', 'node', or 'edge'")
        
        df = pd.read_sql_query(query, conn)
        
        # Expand JSON features into columns
        if not df.empty:
            feature_col = f"{feature_level}_features"
            features_expanded = pd.json_normalize(df[feature_col].apply(json.loads))
            df = pd.concat([df.drop(columns=[feature_col]), features_expanded], axis=1)
        
        df.to_csv(output_path, index=False)
        logger.info(f"\txported {len(df)} {feature_level} features to {output_path}")
        
        conn.close()

# backward compatibility
Graph = MolecularGraph
