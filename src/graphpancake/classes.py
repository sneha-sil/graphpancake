from graphpancake.functions import *
import numpy as np

class DictData:
    """Memory-optimized data container using __slots__."""
    
    __slots__ = [
        # Basic molecular properties
        'graph_type', 'id', 'smiles', 'formula', 'molecular_mass', 'num_atoms',
        'xyz_coordinates', 'atomic_numbers', 'atom_labels', 'polarizability',
        'moments_of_inertia', 'rotational_constants', 'rotational_temperatures',
        'heat_capacity_Cv', 'heat_capacity_Cp', 'entropy', 'ZPE', 'electronic_energy',
        'potential_energy', 'enthalpy', 'gibbs_free_energy', 'frequencies',
        
        # Quantum chemical analysis data
        'wiberg_bond_order_matrix', 'number_of_2C_BDs_matrix', 'natural_population_analysis_charges',
        'natural_electron_configuration', 'CLPO_data', 'natural_population_analysis',
        'natural_population_totals', 'core_NBOs', 'lone_pair_NBOs', 'bonding_NBOs',
        'antibonding_NBOs',
        
        # Pre-computed derived properties
        'bond_distance_matrix', 'atomic_masses', 'electronegativities', 'covalent_radii',
        'wiberg_bond_order_totals', 'bound_hydrogens', 'node_degrees',
        
        # CLPO-specific features
        'lone_pair_CLPOs', 'bonding_CLPOs', 'antibonding_CLPOs',
        
        # Feature computation cache
        '_precomputed_node_features', '_precomputed_edge_features',
        '_node_features_computed', '_edge_features_computed',
        
        # Additional derived features that might be computed
        'npa_charges', 'wiberg_matrix', 'bond_matrix', 'orbital_occupancies'
    ]
    
    def __init__(self, qm_data_dict: dict):
        """
        Initializes DictData with all datapoints from the provided QM data dictionary.
        Values return None if the key is not present in the qm_data dictionary generated using file inputs.

        Args:
            qm_data_dict (dict): Dictionary containing all quantum mechanical data fields.
        """
        # Single comprehensive key extraction to minimize dictionary access operations
        all_keys = {
            'graph_type': 'graph_type',
            'id': 'id', 
            'smiles': 'smiles',
            'formula': 'formula',
            'molecular_mass': 'molecular_mass',
            'num_atoms': 'num_atoms',
            
            'xyz_coordinates': 'xyz_coordinates',
            'atomic_numbers': 'atomic_numbers', 
            'atom_labels': 'atom_labels',
            
            'polarizability': 'polarizability',
            'moments_of_inertia': 'principal_moments',
            'rotational_constants': 'rotational_constants', 
            'rotational_temperatures': 'rotational_temperatures',
            'heat_capacity_Cv': 'cv',
            'heat_capacity_Cp': 'cp',
            'entropy': 'entropy',
            'ZPE': 'zpe',
            'electronic_energy': 'electronic_energy',
            'potential_energy': 'potential_energy',
            'enthalpy': 'enthalpy',
            'gibbs_free_energy': 'gibbs_free_energy',
            'frequencies': 'frequencies',
            
            'wiberg_bond_order_matrix': 'wiberg_matrix',
            'number_of_2C_BDs_matrix': 'bond_matrix',
            'natural_population_analysis_charges': 'npa_charges',
            'natural_electron_configuration': 'natural_electron_configuration',
            'CLPO_data': 'CLPO_data',
            'natural_population_analysis': 'natural_population_analysis',
            'natural_population_totals': 'natural_population_totals',
            'core_NBOs': 'core_NBOs',
            'lone_pair_NBOs': 'lone_pair_NBOs',
            'bonding_NBOs': 'bonding_NBOs',
            'antibonding_NBOs': 'antibonding_NBOs'
        }
        
        extracted_data = {key: qm_data_dict.get(dict_key) for key, dict_key in all_keys.items()}
        for attr, value in extracted_data.items():
            setattr(self, attr, value)
        
        # Pre-compute distance matrix
        self.bond_distance_matrix = calculate_distance_matrix(self.xyz_coordinates) if self.xyz_coordinates is not None else None
        
        # Pre-compute atomic properties
        if self.atomic_numbers is not None:
            self.atomic_masses, self.electronegativities, self.covalent_radii = get_atomic_properties(self.atomic_numbers)
        else:
            self.atomic_masses = None
            self.electronegativities = None
            self.covalent_radii = None
        
        # Pre-compute all node and edge features
        self._precomputed_node_features = None
        self._precomputed_edge_features = None
        self._node_features_computed = False
        self._edge_features_computed = False
        
        # Pre-compute graph-type specific derived features
        self._precompute_graph_type_features()
        
    def to_dict(self):
        """
        Convert the DictData object to a dictionary for compatibility with __dict__ access.
        This is needed because __slots__ objects don't have __dict__.
        
        Returns:
            dict: Dictionary representation of all attributes
        """
        result = {}
        for attr in self.__slots__:
            if hasattr(self, attr):
                result[attr] = getattr(self, attr)
        return result
        
    def _precompute_graph_type_features(self):
        """Pre-compute derived features based on graph type - all base data already extracted."""
        if self.graph_type in ["NPA", "QM"]:
            if self.wiberg_bond_order_matrix is not None:
                self.wiberg_bond_order_totals = get_wiberg_bond_order_totals(self.wiberg_bond_order_matrix)
                self.bound_hydrogens = get_bound_hydrogens_per_atom(self.atomic_numbers, self.wiberg_bond_order_matrix) if self.atomic_numbers else None
                self.node_degrees = get_node_degrees(self.wiberg_bond_order_matrix)
            else:
                self.wiberg_bond_order_totals = None
                self.bound_hydrogens = None
                self.node_degrees = None
                
        if self.graph_type == "NPA":
            if self.CLPO_data is not None:
                self.lone_pair_CLPOs = get_LP_CLPOs(self.CLPO_data)
                self.bonding_CLPOs = get_BD_CLPOs(self.CLPO_data)
                self.antibonding_CLPOs = get_AB_CLPOs(self.CLPO_data)
            else:
                self.lone_pair_CLPOs = None
                self.bonding_CLPOs = None
                self.antibonding_CLPOs = None
            
    def _precompute_all_node_features(self):
        """Pre-compute all node features once, storing structured data for flexible formatting."""
        if self._node_features_computed or self.atomic_numbers is None or self.num_atoms is None:
            return
            
        # Create structured node data - each element corresponds to one atom
        node_data = {
            'atom_indices': list(range(self.num_atoms)),
            'atomic_numbers': self.atomic_numbers,
            'atom_labels': self.atom_labels,
            'positions': self.xyz_coordinates,
            'atomic_masses': self.atomic_masses,
            'electronegativities': self.electronegativities,
            'covalent_radii': self.covalent_radii,
        }
        
        if self.graph_type == "NPA":
            if self.natural_population_analysis_charges:
                electron_pops, nmb_pops, npa_charges = extract_npa_charges(self.natural_population_analysis_charges)
            else:
                electron_pops = nmb_pops = npa_charges = [None] * self.num_atoms
                
            # Convert numpy arrays to Python lists to avoid binary storage
            wiberg_totals = self.wiberg_bond_order_totals.tolist() if self.wiberg_bond_order_totals is not None else [None] * self.num_atoms
            bound_h = self.bound_hydrogens.tolist() if self.bound_hydrogens is not None else [None] * self.num_atoms
            node_deg = self.node_degrees.tolist() if self.node_degrees is not None else [None] * self.num_atoms
                
            node_data.update({
                'wiberg_bond_order_totals': wiberg_totals,
                'bound_hydrogens': bound_h,
                'node_degrees': node_deg,
                'electron_populations': electron_pops,
                'nmb_populations': nmb_pops,
                'npa_charges': npa_charges,
            })
            
        elif self.graph_type == "NBO":
            num_atoms = self.num_atoms
            
            natural_charges = np.full(num_atoms, np.nan)
            core_populations = np.full(num_atoms, np.nan)
            valence_populations = np.full(num_atoms, np.nan)
            rydberg_populations = np.full(num_atoms, np.nan)
            total_populations = np.full(num_atoms, np.nan)
            core_orbital_occupancies = np.full(num_atoms, np.nan)
            core_orbital_energies = np.full(num_atoms, np.nan)
            
            if self.natural_population_analysis:
                for i, npa in enumerate(self.natural_population_analysis[:num_atoms]):
                    if npa:
                        natural_charges[i] = npa.get('natural_charge', np.nan)
                        core_populations[i] = npa.get('core_population', np.nan)
                        valence_populations[i] = npa.get('valence_population', np.nan)
                        rydberg_populations[i] = npa.get('rydberg_population', np.nan)
                        total_populations[i] = npa.get('total_population', np.nan)
            
            if self.core_NBOs:
                for i in range(num_atoms):
                    core_nbo = get_NBO_entry_by_indices(self.core_NBOs, [i+1])
                    if core_nbo:
                        core_orbital_occupancies[i] = get_NBO_information(core_nbo, "NBO_occupancy") or np.nan
                        core_orbital_energies[i] = get_NBO_information(core_nbo, "energy") or np.nan
            
            lone_pair_occupancies = []
            lone_pair_energies = []
            max_lone_pairs = 2
            
            for i in range(num_atoms):
                lone_pair_nbos = get_NBO_entry_by_indices(self.lone_pair_NBOs, [i+1]) if self.lone_pair_NBOs else None
                occupancies = get_NBO_information(lone_pair_nbos, "NBO_occupancy") if lone_pair_nbos else []
                energies = get_NBO_information(lone_pair_nbos, "energy") if lone_pair_nbos else []
                
                if not isinstance(occupancies, (list, tuple)):
                    occupancies = [occupancies] if occupancies is not None else []
                if not isinstance(energies, (list, tuple)):
                    energies = [energies] if energies is not None else []
                    
                atom_lp_occ = list(occupancies[:max_lone_pairs]) + [None] * max(0, max_lone_pairs - len(occupancies))
                atom_lp_en = list(energies[:max_lone_pairs]) + [None] * max(0, max_lone_pairs - len(energies))
                        
                lone_pair_occupancies.append(atom_lp_occ)
                lone_pair_energies.append(atom_lp_en)
            
            # Convert nan arrays back to lists with None for compatibility
            node_data.update({
                'natural_charges': [None if np.isnan(x) else x for x in natural_charges],
                'core_populations': [None if np.isnan(x) else x for x in core_populations],
                'valence_populations': [None if np.isnan(x) else x for x in valence_populations],
                'rydberg_populations': [None if np.isnan(x) else x for x in rydberg_populations],
                'total_populations': [None if np.isnan(x) else x for x in total_populations],
                'core_orbital_occupancies': [None if np.isnan(x) else x for x in core_orbital_occupancies],
                'core_orbital_energies': [None if np.isnan(x) else x for x in core_orbital_energies],
                'lone_pair_occupancies': lone_pair_occupancies,
                'lone_pair_energies': lone_pair_energies,
            })
            
        elif self.graph_type == "QM":
            num_atoms = self.num_atoms
            
            natural_charges = np.full(num_atoms, np.nan)
            core_populations = np.full(num_atoms, np.nan)
            valence_populations = np.full(num_atoms, np.nan)
            rydberg_populations = np.full(num_atoms, np.nan)
            total_populations = np.full(num_atoms, np.nan)
            core_orbital_occupancies = np.full(num_atoms, np.nan)
            core_orbital_energies = np.full(num_atoms, np.nan)
            
            if self.natural_population_analysis:
                for i, npa in enumerate(self.natural_population_analysis[:num_atoms]):
                    if npa:
                        natural_charges[i] = npa.get('natural_charge', np.nan)
                        core_populations[i] = npa.get('core_population', np.nan)
                        valence_populations[i] = npa.get('valence_population', np.nan)
                        rydberg_populations[i] = npa.get('rydberg_population', np.nan)
                        total_populations[i] = npa.get('total_population', np.nan)
            
            if self.core_NBOs:
                for i in range(num_atoms):
                    core_nbo = get_NBO_entry_by_indices(self.core_NBOs, [i+1])
                    if core_nbo:
                        core_orbital_occupancies[i] = get_NBO_information(core_nbo, "NBO_occupancy") or np.nan
                        core_orbital_energies[i] = get_NBO_information(core_nbo, "energy") or np.nan
            
            lone_pair_occupancies = []
            lone_pair_energies = []
            max_lone_pairs = 2
            
            for i in range(num_atoms):
                lone_pair_nbos = get_NBO_entry_by_indices(self.lone_pair_NBOs, [i+1]) if self.lone_pair_NBOs else None
                occupancies = get_NBO_information(lone_pair_nbos, "NBO_occupancy") if lone_pair_nbos else []
                energies = get_NBO_information(lone_pair_nbos, "energy") if lone_pair_nbos else []
                
                # Ensure lists
                if not isinstance(occupancies, (list, tuple)):
                    occupancies = [occupancies] if occupancies is not None else []
                if not isinstance(energies, (list, tuple)):
                    energies = [energies] if energies is not None else []
                    
                # Pad to exactly max_lone_pairs
                atom_lp_occ = list(occupancies[:max_lone_pairs]) + [None] * max(0, max_lone_pairs - len(occupancies))
                atom_lp_en = list(energies[:max_lone_pairs]) + [None] * max(0, max_lone_pairs - len(energies))
                        
                lone_pair_occupancies.append(atom_lp_occ)
                lone_pair_energies.append(atom_lp_en)
            
            # Combine all QM features including NPA charges
            if self.natural_population_analysis_charges:
                electron_pops, nmb_pops, npa_charges = extract_npa_charges(self.natural_population_analysis_charges)
            else:
                electron_pops = nmb_pops = npa_charges = [None] * self.num_atoms
            
            # Convert numpy arrays to Python lists to avoid binary storage
            wiberg_totals = self.wiberg_bond_order_totals.tolist() if self.wiberg_bond_order_totals is not None else [None] * self.num_atoms
            bound_h = self.bound_hydrogens.tolist() if self.bound_hydrogens is not None else [None] * self.num_atoms
            node_deg = self.node_degrees.tolist() if self.node_degrees is not None else [None] * self.num_atoms
            
            node_data.update({
                'wiberg_bond_order_totals': wiberg_totals,
                'bound_hydrogens': bound_h,
                'node_degrees': node_deg,
                'electron_populations': electron_pops,
                'nmb_populations': nmb_pops,
                'npa_charges': npa_charges,
                'natural_charges': [None if np.isnan(x) else x for x in natural_charges],
                'core_populations': [None if np.isnan(x) else x for x in core_populations],
                'valence_populations': [None if np.isnan(x) else x for x in valence_populations],
                'rydberg_populations': [None if np.isnan(x) else x for x in rydberg_populations],
                'total_populations': [None if np.isnan(x) else x for x in total_populations],
                'core_orbital_occupancies': [None if np.isnan(x) else x for x in core_orbital_occupancies],
                'core_orbital_energies': [None if np.isnan(x) else x for x in core_orbital_energies],
                'lone_pair_occupancies': lone_pair_occupancies,
                'lone_pair_energies': lone_pair_energies,
            })
            
        elif self.graph_type == "DFT":
            # DFT only has basic atomic properties
            pass
            
        self._precomputed_node_features = node_data
        self._node_features_computed = True
        
    def _precompute_all_edge_features(self, bond_order_threshold: float = 0.4, max_distance: float = 3.0):
        """Pre-compute all edge features once, storing structured data for flexible formatting."""
        if self._edge_features_computed or self.bond_distance_matrix is None:
            return
            
        edge_data = {
            'atom_i_list': [],
            'atom_j_list': [],
            'distances': [],
            'edge_types': [],
            'bond_orders': [],
            'conventional_bond_orders': [],
            'num_2C_BDs': [],
            'bonding_orbital_occupancies': [],
            'bonding_orbital_energies': [],
            'antibonding_orbital_occupancies': [],
            'antibonding_orbital_energies': [],
        }
        
        if self.graph_type == "NPA":
            atom_i_indices, atom_j_indices, distances, bond_orders = analyze_bond_orders(
                self.bond_distance_matrix, self.wiberg_bond_order_matrix, 
                bond_order_threshold, max_distance
            )
            
            if atom_i_indices: 
                conventional_bond_orders = get_conventional_bond_orders(bond_orders)
                
                bonding_occupancies = []
                antibonding_occupancies = []
                num_2c_bds = []
                
                for i, j in zip(atom_i_indices, atom_j_indices):
                    atom_index_i, atom_index_j = i + 1, j + 1
                    
                    num_2c_bd = self.number_of_2C_BDs_matrix[i][j] if self.number_of_2C_BDs_matrix is not None else None
                    num_2c_bds.append(num_2c_bd)
                    
                    bd_occupancy = None
                    ab_occupancy = None
                    
                    if self.bonding_CLPOs is not None:
                        bd_clpo = get_CLPO_entry_by_indices(self.bonding_CLPOs, [atom_index_i, atom_index_j])
                        if bd_clpo:
                            bd_occupancy = get_CLPO_information(bd_clpo, "occupancy")
                            
                    if self.antibonding_CLPOs is not None:
                        ab_clpo = get_CLPO_entry_by_indices(self.antibonding_CLPOs, [atom_index_i, atom_index_j])
                        if ab_clpo:
                            ab_occupancy = get_CLPO_information(ab_clpo, "occupancy")
                    
                    bonding_occupancies.append(bd_occupancy)
                    antibonding_occupancies.append(ab_occupancy)
                
                edge_data['atom_i_list'] = atom_i_indices
                edge_data['atom_j_list'] = atom_j_indices
                edge_data['distances'] = distances
                edge_data['edge_types'] = ["NPA"] * len(atom_i_indices)
                edge_data['bond_orders'] = bond_orders
                edge_data['conventional_bond_orders'] = conventional_bond_orders
                edge_data['num_2C_BDs'] = num_2c_bds
                edge_data['bonding_orbital_occupancies'] = bonding_occupancies
                edge_data['bonding_orbital_energies'] = [None] * len(atom_i_indices)
                edge_data['antibonding_orbital_occupancies'] = antibonding_occupancies
                edge_data['antibonding_orbital_energies'] = [None] * len(atom_i_indices)
                        
        elif self.graph_type == "NBO":
            atom_i_indices, atom_j_indices, distances, _ = analyze_bond_orders(
                self.bond_distance_matrix, None, 0.0, max_distance
            )
            
            valid_bonds = []
            for idx, (i, j) in enumerate(zip(atom_i_indices, atom_j_indices)):
                atom_index_i, atom_index_j = i + 1, j + 1
                bonding_nbo = get_NBO_entry_by_indices(self.bonding_NBOs, [atom_index_i, atom_index_j])
                if bonding_nbo:
                    valid_bonds.append(idx)
            
            if valid_bonds:
                valid_i = [atom_i_indices[idx] for idx in valid_bonds]
                valid_j = [atom_j_indices[idx] for idx in valid_bonds]
                valid_distances = [distances[idx] for idx in valid_bonds]
                
                bonding_occupancies = []
                bonding_energies = []
                antibonding_occupancies = []
                antibonding_energies = []
                
                for i, j in zip(valid_i, valid_j):
                    atom_index_i, atom_index_j = i + 1, j + 1
                    bonding_nbo = get_NBO_entry_by_indices(self.bonding_NBOs, [atom_index_i, atom_index_j])
                    antibonding_nbo = get_NBO_entry_by_indices(self.antibonding_NBOs, [atom_index_i, atom_index_j])
                    
                    bonding_occupancies.append(get_NBO_information(bonding_nbo, "occupancy"))
                    bonding_energies.append(get_NBO_information(bonding_nbo, "energy"))
                    antibonding_occupancies.append(get_NBO_information(antibonding_nbo, "occupancy") if antibonding_nbo else None)
                    antibonding_energies.append(get_NBO_information(antibonding_nbo, "energy") if antibonding_nbo else None)
                
                edge_data['atom_i_list'] = valid_i
                edge_data['atom_j_list'] = valid_j
                edge_data['distances'] = valid_distances
                edge_data['edge_types'] = ["NBO"] * len(valid_i)
                edge_data['bond_orders'] = [None] * len(valid_i)
                edge_data['conventional_bond_orders'] = [None] * len(valid_i)
                edge_data['num_2C_BDs'] = [None] * len(valid_i)
                edge_data['bonding_orbital_occupancies'] = bonding_occupancies
                edge_data['bonding_orbital_energies'] = bonding_energies
                edge_data['antibonding_orbital_occupancies'] = antibonding_occupancies
                edge_data['antibonding_orbital_energies'] = antibonding_energies
                        
        elif self.graph_type == "QM":
            atom_i_indices, atom_j_indices, distances, bond_orders = analyze_bond_orders(
                self.bond_distance_matrix, self.wiberg_bond_order_matrix, 
                bond_order_threshold, max_distance
            )
            
            if atom_i_indices:
                conventional_bond_orders = get_conventional_bond_orders(bond_orders)
                
                edge_types = []
                num_2c_bds = []
                bonding_occupancies = []
                bonding_energies = []
                antibonding_occupancies = []
                antibonding_energies = []
                
                for i, j in zip(atom_i_indices, atom_j_indices):
                    atom_index_i, atom_index_j = i + 1, j + 1
                    
                    num_2c_bd = self.number_of_2C_BDs_matrix[i][j] if self.number_of_2C_BDs_matrix is not None else None
                    num_2c_bds.append(num_2c_bd)
                    
                    bonding_nbo = get_NBO_entry_by_indices(self.bonding_NBOs, [atom_index_i, atom_index_j])
                    antibonding_nbo = get_NBO_entry_by_indices(self.antibonding_NBOs, [atom_index_i, atom_index_j])
                    
                    edge_type = "NBO" if bonding_nbo else "BO"
                    edge_types.append(edge_type)
                    
                    bd_occupancy = get_NBO_information(bonding_nbo, "occupancy") if bonding_nbo else None
                    bd_energy = get_NBO_information(bonding_nbo, "energy") if bonding_nbo else None
                    ab_occupancy = get_NBO_information(antibonding_nbo, "occupancy") if antibonding_nbo else None
                    ab_energy = get_NBO_information(antibonding_nbo, "energy") if antibonding_nbo else None
                    
                    bonding_occupancies.append(bd_occupancy)
                    bonding_energies.append(bd_energy)
                    antibonding_occupancies.append(ab_occupancy)
                    antibonding_energies.append(ab_energy)
                
                edge_data['atom_i_list'] = atom_i_indices
                edge_data['atom_j_list'] = atom_j_indices
                edge_data['distances'] = distances
                edge_data['edge_types'] = edge_types
                edge_data['bond_orders'] = bond_orders
                edge_data['conventional_bond_orders'] = conventional_bond_orders
                edge_data['num_2C_BDs'] = num_2c_bds
                edge_data['bonding_orbital_occupancies'] = bonding_occupancies
                edge_data['bonding_orbital_energies'] = bonding_energies
                edge_data['antibonding_orbital_occupancies'] = antibonding_occupancies
                edge_data['antibonding_orbital_energies'] = antibonding_energies
                        
        elif self.graph_type == "DFT":
            atom_i_indices, atom_j_indices, distances, edge_matrix = get_dft_edges(self.bond_distance_matrix, self.smiles, self.num_atoms)
            
            if atom_i_indices:
                edge_data['atom_i_list'] = atom_i_indices
                edge_data['atom_j_list'] = atom_j_indices
                edge_data['distances'] = distances
                edge_data['edge_types'] = ["distance"] * len(atom_i_indices)
                edge_data['bond_orders'] = [None] * len(atom_i_indices)
                edge_data['conventional_bond_orders'] = [None] * len(atom_i_indices)
                edge_data['num_2C_BDs'] = [None] * len(atom_i_indices)
                edge_data['bonding_orbital_occupancies'] = [None] * len(atom_i_indices)
                edge_data['bonding_orbital_energies'] = [None] * len(atom_i_indices)
                edge_data['antibonding_orbital_occupancies'] = [None] * len(atom_i_indices)
                edge_data['antibonding_orbital_energies'] = [None] * len(atom_i_indices)
        
        self._precomputed_edge_features = edge_data
        self._edge_features_computed = True
        
    def _get_conventional_bond_order(self, bond_order: float) -> float:
        """Convert Wiberg bond order to conventional bond order."""
        if bond_order < 1.11:
            return 1.0
        elif 1.11 <= bond_order < 1.54:
            return 1.5
        elif 1.54 <= bond_order < 2.0:
            return 2.0
        else:
            return 3.0
            
    def get_precomputed_node_features(self):
        """Get pre-computed node features (call this instead of iterating through atoms)."""
        if not self._node_features_computed:
            self._precompute_all_node_features()
        return self._precomputed_node_features
        
    def get_precomputed_edge_features(self, bond_order_threshold: float = 0.4, max_distance: float = 3.0):
        """Get pre-computed edge features (call this instead of nested loops)."""
        if not self._edge_features_computed:
            self._precompute_all_edge_features(bond_order_threshold, max_distance)
        return self._precomputed_edge_features
    
    def vectorized_extract_charges(self, node_data: dict, indices: list, graph_type: str) -> list:
        """
        Vectorized extraction of charge/population data for specific atoms.
        
        Args:
            node_data (dict): Pre-computed node data dictionary
            indices (list): List of atom indices to extract data for
            graph_type (str): Type of graph data to extract
            
        Returns:
            list: List of dictionaries containing extracted features for each index
        """
        def safe_get(data_dict, key, index):
            """Safely get data from dict with bounds checking."""
            if key not in data_dict:
                return None
            data = data_dict[key]
            if data is None:
                return None
            try:
                if index >= len(data):
                    return None
                return data[index]
            except (TypeError, IndexError):
                return None
        
        results = []
        for i in indices:
            if graph_type == "NPA":
                result = {
                    "wiberg_bond_order_total": safe_get(node_data, 'wiberg_bond_order_totals', i),
                    "bound_hydrogens": safe_get(node_data, 'bound_hydrogens', i),
                    "node_degree": safe_get(node_data, 'node_degrees', i),
                    "electron_population": safe_get(node_data, 'electron_populations', i),
                    "nmb_population": safe_get(node_data, 'nmb_populations', i),
                    "npa_charge": safe_get(node_data, 'npa_charges', i),
                }
            elif graph_type == "NBO":
                result = {
                    "natural_charge": safe_get(node_data, 'natural_charges', i),
                    "core_population": safe_get(node_data, 'core_populations', i),
                    "valence_population": safe_get(node_data, 'valence_populations', i),
                    "rydberg_population": safe_get(node_data, 'rydberg_populations', i),
                    "total_population": safe_get(node_data, 'total_populations', i),
                    "core_orbital_occupancy": safe_get(node_data, 'core_orbital_occupancies', i),
                    "core_orbital_energy": safe_get(node_data, 'core_orbital_energies', i),
                }
            else:
                result = {}
            
            results.append(result)
        
        return results
        
    def get_node_data_for_database(self, molecule_id: str = None):
        """
        Get node data formatted for database insertion.
        
        Args:
            molecule_id (str): Optional molecule ID to include in each record
            
        Returns:
            list[dict]: List of flat dictionaries, one per atom, ready for database insertion
        """
        node_data = self.get_precomputed_node_features()
        if not node_data or not node_data['atom_indices']:
            return []
        
        db_records = []
        num_nodes = len(node_data['atom_indices'])
        
        for i in range(num_nodes):
            record = {
                'molecule_id': molecule_id,
                'atom_index': node_data['atom_indices'][i],
                'atomic_number': node_data['atomic_numbers'][i],
                'atom_label': node_data['atom_labels'][i],
                'x_position': node_data['positions'][i][0] if node_data['positions'][i] else None,
                'y_position': node_data['positions'][i][1] if node_data['positions'][i] else None,
                'z_position': node_data['positions'][i][2] if node_data['positions'][i] else None,
                'atomic_mass': node_data['atomic_masses'][i],
                'electronegativity': node_data['electronegativities'][i],
                'covalent_radius': node_data['covalent_radii'][i],
            }
            
            if self.graph_type in ["NPA", "QM"]:
                record.update({
                    'wiberg_bond_order_total': node_data.get('wiberg_bond_order_totals', [None])[i],
                    'bound_hydrogens': node_data.get('bound_hydrogens', [None])[i],
                    'node_degree': node_data.get('node_degrees', [None])[i],
                    'electron_population': node_data.get('electron_populations', [None])[i],
                    'nmb_population': node_data.get('nmb_populations', [None])[i],
                    'npa_charge': node_data.get('npa_charges', [None])[i],
                })
                
            if self.graph_type in ["NBO", "QM"]:
                record.update({
                    'natural_charge': node_data.get('natural_charges', [None])[i],
                    'core_population': node_data.get('core_populations', [None])[i],
                    'valence_population': node_data.get('valence_populations', [None])[i],
                    'rydberg_population': node_data.get('rydberg_populations', [None])[i],
                    'total_population': node_data.get('total_populations', [None])[i],
                    'core_orbital_occupancy': node_data.get('core_orbital_occupancies', [None])[i],
                    'core_orbital_energy': node_data.get('core_orbital_energies', [None])[i],
                })
                
                if node_data.get('lone_pair_occupancies') and i < len(node_data['lone_pair_occupancies']):
                    lone_pair_occ = node_data['lone_pair_occupancies'][i]
                    lone_pair_en = node_data['lone_pair_energies'][i]
                    
                    for j in range(min(2, len(lone_pair_occ))):
                        record[f'lone_pair_{j+1}_occupancy'] = lone_pair_occ[j]
                        record[f'lone_pair_{j+1}_energy'] = lone_pair_en[j]
                        
            if self.graph_type == "NPA":
                record.update({
                    'electron_population': node_data.get('electron_populations', [None])[i],
                    'nmb_population': node_data.get('nmb_populations', [None])[i],
                    'npa_charge': node_data.get('npa_charges', [None])[i],
                })
                
            db_records.append(record)
            
        return db_records
        
    def get_edge_data_for_database(self, molecule_id: str = None, bond_order_threshold: float = 0.4, max_distance: float = 3.0):
        """
        Get edge data formatted for database insertion.
        
        Args:
            molecule_id (str): Optional molecule ID to include in each record
            bond_order_threshold (float): Minimum bond order threshold
            max_distance (float): Maximum distance threshold
            
        Returns:
            list[dict]: List of flat dictionaries, one per edge, ready for database insertion
        """
        edge_data = self.get_precomputed_edge_features(bond_order_threshold, max_distance)
        if not edge_data or not edge_data['atom_i_list']:
            return []
        
        db_records = []
        num_edges = len(edge_data['atom_i_list'])
        
        for i in range(num_edges):
            record = {
                'molecule_id': molecule_id,
                'atom_i': edge_data['atom_i_list'][i],
                'atom_j': edge_data['atom_j_list'][i],
                'distance': edge_data['distances'][i],
                'edge_type': edge_data['edge_types'][i],
                'bond_order': edge_data['bond_orders'][i],
                'conventional_bond_order': edge_data['conventional_bond_orders'][i],
                'num_2C_BDs': edge_data['num_2C_BDs'][i],
                'bonding_orbital_occupancy': edge_data['bonding_orbital_occupancies'][i],
                'bonding_orbital_energy': edge_data['bonding_orbital_energies'][i],
                'antibonding_orbital_occupancy': edge_data['antibonding_orbital_occupancies'][i],
                'antibonding_orbital_energy': edge_data['antibonding_orbital_energies'][i],
            }
            db_records.append(record)
            
        return db_records
        
    def get_columnar_node_data(self):
        """
        Get node data in columnar format for efficient bulk operations.
        
        Returns:
            dict: Dictionary with column names as keys and lists of values
        """
        return self.get_precomputed_node_features()
        
    def get_columnar_edge_data(self, bond_order_threshold: float = 0.4, max_distance: float = 3.0):
        """
        Get edge data in columnar format for efficient bulk operations.
        
        Returns:
            dict: Dictionary with column names as keys and lists of values
        """
        return self.get_precomputed_edge_features(bond_order_threshold, max_distance)

class GraphBase:
    """Base class for graph components to share common functionality."""
    
    __slots__ = ['qm_data']
    
    def __init__(self, qm_data: DictData):
        self.qm_data = qm_data
        self.graph_type = qm_data.graph_type
        
    def _batch_extract(self, data_dict: dict, keys: list, default=None) -> dict:
        """
        Efficiently extract multiple keys from a dictionary in batch.
        
        Args:
            data_dict (dict): Dictionary to extract from
            keys (list): List of keys to extract
            default: Default value for missing keys
            
        Returns:
            dict: Dictionary with extracted key-value pairs
        """
        return {key: data_dict.get(key, default) for key in keys}
        
    def _batch_extract_to_attrs(self, data_dict: dict, key_mapping: dict, default=None):
        """
        Extract multiple keys from dictionary and set as object attributes.
        
        Args:
            data_dict (dict): Dictionary to extract from
            key_mapping (dict): Mapping of {attribute_name: dict_key}
            default: Default value for missing keys
        """
        for attr_name, dict_key in key_mapping.items():
            setattr(self, attr_name, data_dict.get(dict_key, default))
            
    def _extract_indexed_values(self, data_list: list, indices: list, prefix: str, default=None):
        """
        Extract indexed values from a list and set as numbered attributes.
        
        Args:
            data_list (list): List to extract from
            indices (list): List of indices to extract
            prefix (str): Prefix for attribute names (e.g., 'moment' -> 'moment_1', 'moment_2')
            default: Default value for missing indices
        """
        for i, idx in enumerate(indices, 1):
            attr_name = f"{prefix}_{i}"
            if data_list and idx < len(data_list):
                setattr(self, attr_name, data_list[idx])
            else:
                setattr(self, attr_name, default)
        
class GraphInfo(GraphBase):
    def __init__(self, qm_data: DictData):
        """
        Initializes GraphInfo with graph-level metadata from DictData.

        Args:
            qm_data (DictData): Quantum mechanical data object.
        """
        super().__init__(qm_data)
        self.id = qm_data.id
        self.smiles = qm_data.smiles
        self.formula = qm_data.formula
        self.molecular_mass = qm_data.molecular_mass
        self.num_atoms = qm_data.num_atoms
        
        if self.graph_type == "NBO" or self.graph_type == "QM":
            natural_population_totals = qm_data.natural_population_totals
            self.total_charge = natural_population_totals['total_charge'] if natural_population_totals and 'total_charge' in natural_population_totals else None
            self.num_electrons = sum(qm_data.atomic_numbers) - self.total_charge if self.total_charge is not None and qm_data.atomic_numbers else None
        else:
            self.total_charge = float(0)
            self.num_electrons = sum(qm_data.atomic_numbers) if qm_data.atomic_numbers is not None else None

    def get_graph_features(self) -> dict:
        """
        Returns graph-level metadata as a dictionary.

        Returns:
            dict: Dictionary containing graph-level metadata.
        """
        graph_info_dict = {
            "graph_type": self.graph_type,
            "id": self.id,
            "smiles": self.smiles,
            "formula": self.formula,
            "molecular_mass": self.molecular_mass,
            "num_atoms": self.num_atoms,
            "charge": self.total_charge,
            "num_electrons": self.num_electrons,
        }
        return graph_info_dict
        
    def get_graph_features_ML(self) -> dict:
        """
        Returns graph-level metadata as a dictionary for machine learning tasks.

        Returns:
            dict: Dictionary containing graph-level metadata.
        """
        ML_graph_info_dict = {
            # "graph_type": self.graph_type,
            "id": self.id,
            # "smiles": self.smiles,
            # "formula": self.formula,
            "molecular_mass": self.molecular_mass,
            "num_atoms": self.num_atoms,
            "charge": self.total_charge,
            "num_electrons": self.num_electrons,
        }
        return ML_graph_info_dict
            
class Node(GraphBase):
    def __init__(self, qm_data: DictData):
        """
        Initializes Node with broad atom-level features from DictData.
        Uses pre-computed data from DictData to avoid redundant calculations.

        Args:
            qm_data (DictData): Quantum mechanical data object with pre-computed features.
        """
        super().__init__(qm_data)
        self.num_nodes = qm_data.num_atoms
        self.xyz_coordinates = qm_data.xyz_coordinates
        self.atomic_numbers = qm_data.atomic_numbers
        self.atom_labels = qm_data.atom_labels
        
        self.atomic_masses = qm_data.atomic_masses
        self.electronegativities = qm_data.electronegativities
        self.covalent_radii = qm_data.covalent_radii
        
        # Batch assign graph-type specific features using mapping to minimize conditional statements
        graph_type_mappings = {
            "NPA": {
                'wiberg_matrix': 'wiberg_bond_order_matrix',
                'wiberg_bond_order_totals': 'wiberg_bond_order_totals',
                'bound_hydrogens': 'bound_hydrogens',
                'node_degrees': 'node_degrees',
                'lone_pair_CLPOs': 'lone_pair_CLPOs',
                'natural_population_analysis_charges': 'natural_population_analysis_charges',
                'natural_electron_configuration': 'natural_electron_configuration'
            },
            "NBO": {
                'core_NBOs': 'core_NBOs',
                'lone_pair_NBOs': 'lone_pair_NBOs',
                'natural_population_analysis': 'natural_population_analysis'
            },
            "QM": {
                'wiberg_matrix': 'wiberg_bond_order_matrix',
                'wiberg_bond_order_totals': 'wiberg_bond_order_totals',
                'bound_hydrogens': 'bound_hydrogens',
                'node_degrees': 'node_degrees',
                'core_NBOs': 'core_NBOs',
                'lone_pair_NBOs': 'lone_pair_NBOs',
                'natural_population_analysis': 'natural_population_analysis'
            }
        }
        
        if self.graph_type in graph_type_mappings:
            self._batch_extract_to_attrs(qm_data.to_dict(), graph_type_mappings[self.graph_type])

    def get_node_features(self) -> list[dict]:
        """
        Extracts atom-level features.

        Returns:
            list[dict]: List of dictionaries, one per atom, containing its features.
        """
        node_data = self.qm_data.get_precomputed_node_features()
        if not node_data or not node_data['atom_indices']:
            return []
        
        node_list = []
        num_nodes = len(node_data['atom_indices'])
        
        for i in range(num_nodes):
            node = {
                "atom_index": node_data['atom_indices'][i],
                "atomic_number": node_data['atomic_numbers'][i],
                "atom_label": node_data['atom_labels'][i],
                "position": node_data['positions'][i],
                "atomic_mass": node_data['atomic_masses'][i],
                "electronegativity": node_data['electronegativities'][i],
                "covalent_radius": node_data['covalent_radii'][i],
            }
            
            if self.graph_type == "NPA":
                node.update({
                    "wiberg_bond_order_total": node_data['wiberg_bond_order_totals'][i] if node_data['wiberg_bond_order_totals'] else None,
                    "bound_hydrogens": node_data['bound_hydrogens'][i] if node_data['bound_hydrogens'] else None,
                    "node_degree": node_data['node_degrees'][i] if node_data['node_degrees'] else None,
                    "electron_population": node_data['electron_populations'][i] if node_data['electron_populations'] else None,
                    "nmb_population": node_data['nmb_populations'][i] if node_data['nmb_populations'] else None,
                    "npa_charge": node_data['npa_charges'][i] if node_data['npa_charges'] else None,
                })
                
            elif self.graph_type == "NBO":
                nbo_features = self.qm_data.vectorized_extract_charges(node_data, [i], 'NBO')[0]
                node.update(nbo_features)
                
                if node_data['lone_pair_occupancies'] and i < len(node_data['lone_pair_occupancies']):
                    lone_pair_occ = node_data['lone_pair_occupancies'][i]
                    lone_pair_en = node_data['lone_pair_energies'][i]
                    
                    # Check that both lists exist and are not None before zip
                    if lone_pair_occ is not None and lone_pair_en is not None:
                        if len(lone_pair_occ) == 2 and lone_pair_occ[0] is not None and lone_pair_occ[1] is None:
                            node["lone_pair_occupancy"] = lone_pair_occ[0]
                            node["lone_pair_energy"] = lone_pair_en[0]
                        else:
                            for j, (occ, en) in enumerate(zip(lone_pair_occ, lone_pair_en), 1):
                                if occ is not None:
                                    node[f"lone_pair_{j}_occupancy"] = occ
                                    node[f"lone_pair_{j}_energy"] = en
                
            elif self.graph_type == "QM":
                npa_features = self.qm_data.vectorized_extract_charges(node_data, [i], 'NPA')[0]
                nbo_features = self.qm_data.vectorized_extract_charges(node_data, [i], 'NBO')[0]
                node.update(npa_features)
                node.update(nbo_features)
                
                if node_data['lone_pair_occupancies'] and i < len(node_data['lone_pair_occupancies']):
                    lone_pair_occ = node_data['lone_pair_occupancies'][i]
                    lone_pair_en = node_data['lone_pair_energies'][i]
                    
                    max_lone_pairs = 2
                    for j in range(1, max_lone_pairs + 1):
                        if j <= len(lone_pair_occ):
                            node[f"lone_pair_{j}_occupancy"] = lone_pair_occ[j-1]
                            node[f"lone_pair_{j}_energy"] = lone_pair_en[j-1]
                        else:
                            node[f"lone_pair_{j}_occupancy"] = None
                            node[f"lone_pair_{j}_energy"] = None
                            
            elif self.graph_type == "DFT":
                pass
                
            node_list.append(node)
            
        all_keys = set().union(*(node.keys() for node in node_list))
        for node in node_list:
            for key in all_keys:
                node.setdefault(key, None)
                
        return node_list
    
    def get_node_features_ML(self) -> list[dict]:
        """
        Extracts atom-level features for machine learning tasks.

        Returns:
            list[dict]: List of dictionaries, one per atom, containing its features.
        """
        node_data = self.qm_data.get_precomputed_node_features()
        if not node_data or not node_data['atom_indices']:
            return []
        
        ML_node_list = []
        num_nodes = len(node_data['atom_indices'])
        
        for i in range(num_nodes):
            node = {
                "atom_index": node_data['atom_indices'][i],
                "atomic_number": node_data['atomic_numbers'][i],
                # "atom_label": node_data['atom_labels'][i], # string value
                "position": node_data['positions'][i],
                "atomic_mass": node_data['atomic_masses'][i],
                "electronegativity": node_data['electronegativities'][i],
                "covalent_radius": node_data['covalent_radii'][i],
            }
            
            if self.graph_type == "NPA":
                node.update({
                    "wiberg_bond_order_total": node_data['wiberg_bond_order_totals'][i] if node_data['wiberg_bond_order_totals'] else None,
                    "bound_hydrogens": node_data['bound_hydrogens'][i] if node_data['bound_hydrogens'] else None,
                    "node_degree": node_data['node_degrees'][i] if node_data['node_degrees'] else None,
                    "electron_population": node_data['electron_populations'][i] if node_data['electron_populations'] else None,
                    "nmb_population": node_data['nmb_populations'][i] if node_data['nmb_populations'] else None,
                    "npa_charge": node_data['npa_charges'][i] if node_data['npa_charges'] else None,
                })
                
            elif self.graph_type == "NBO":
                nbo_features = self.qm_data.vectorized_extract_charges(node_data, [i], 'NBO')[0]
                node.update(nbo_features)
                
                if node_data['lone_pair_occupancies'] and i < len(node_data['lone_pair_occupancies']):
                    lone_pair_occ = node_data['lone_pair_occupancies'][i]
                    lone_pair_en = node_data['lone_pair_energies'][i]
                    
                    # Check that both lists exist and are not None before zip
                    if lone_pair_occ is not None and lone_pair_en is not None:
                        if len(lone_pair_occ) == 2 and lone_pair_occ[0] is not None and lone_pair_occ[1] is None:
                            node["lone_pair_occupancy"] = lone_pair_occ[0]
                            node["lone_pair_energy"] = lone_pair_en[0]
                        else:
                            for j, (occ, en) in enumerate(zip(lone_pair_occ, lone_pair_en), 1):
                                if occ is not None:
                                    node[f"lone_pair_{j}_occupancy"] = occ
                                    node[f"lone_pair_{j}_energy"] = en
                
            elif self.graph_type == "QM":
                npa_features = self.qm_data.vectorized_extract_charges(node_data, [i], 'NPA')[0]
                nbo_features = self.qm_data.vectorized_extract_charges(node_data, [i], 'NBO')[0]
                node.update(npa_features)
                node.update(nbo_features)
                
                if node_data['lone_pair_occupancies'] and i < len(node_data['lone_pair_occupancies']):
                    lone_pair_occ = node_data['lone_pair_occupancies'][i]
                    lone_pair_en = node_data['lone_pair_energies'][i]
                    
                    max_lone_pairs = 2
                    for j in range(1, max_lone_pairs + 1):
                        if j <= len(lone_pair_occ):
                            node[f"lone_pair_{j}_occupancy"] = lone_pair_occ[j-1]
                            node[f"lone_pair_{j}_energy"] = lone_pair_en[j-1]
                        else:
                            node[f"lone_pair_{j}_occupancy"] = None
                            node[f"lone_pair_{j}_energy"] = None
                            
            elif self.graph_type == "DFT":
                pass
                
            ML_node_list.append(node)

        all_keys = set().union(*(node.keys() for node in ML_node_list))
        for node in ML_node_list:
            for key in all_keys:
                node.setdefault(key, None)

        return ML_node_list

    def get_node_position_dict(self) -> dict:
        """
        Returns a mapping from node index to 3D position.

        Returns:
            dict: Mapping from node index to 3D position.
        """
        return {i: self.xyz_coordinates[i] for i in range(self.num_nodes)}

    def get_node_label_dict(self) -> dict:
        """
        Returns a mapping from node index to atomic label.

        Returns:
            dict: Mapping from node index to atomic label.
        """
        return {i: self.atom_labels[i] for i in range(self.num_nodes)}


class Edge(GraphBase):
    def __init__(self, qm_data: DictData):
        """
        Initializes Edge with bond-level features from DictData.
        Uses pre-computed distance matrix to avoid redundant calculations.

        Args:
            qm_data (DictData): Quantum mechanical data object with pre-computed distance matrix.
        """
        super().__init__(qm_data)
        self.bond_distance_matrix = qm_data.bond_distance_matrix
        
        edge_type_mappings = {
            "NPA": {
                'wiberg_matrix': 'wiberg_bond_order_matrix',
                'number_of_2C_BDs_matrix': 'number_of_2C_BDs_matrix',
                'bonding_CLPOs': 'bonding_CLPOs',
                'antibonding_CLPOs': 'antibonding_CLPOs'
            },
            "NBO": {
                'bonding_NBOs': 'bonding_NBOs',
                'antibonding_NBOs': 'antibonding_NBOs'
            },
            "QM": {
                'wiberg_matrix': 'wiberg_bond_order_matrix',
                'number_of_2C_BDs_matrix': 'number_of_2C_BDs_matrix',
                'bonding_NBOs': 'bonding_NBOs',
                'antibonding_NBOs': 'antibonding_NBOs'
            }
        }
        
        if self.graph_type in edge_type_mappings:
            self._batch_extract_to_attrs(qm_data.to_dict(), edge_type_mappings[self.graph_type])
            
    def get_edge_features(self, bond_order_threshold: float = 0.4, max_distance: float = 3.0) -> list[dict]:
        """
        Extracts edge features.

        Args:
            bond_order_threshold (float): Minimum Wiberg bond order to consider a bond. Default is 0.4.
            max_distance (float): Maximum allowed bond length (in Å) to include an edge. Default is 3.0.

        Returns:
            list[dict]: List of dictionaries containing edge features.
        """
        edge_data = self.qm_data.get_precomputed_edge_features(bond_order_threshold, max_distance)
        if not edge_data or not edge_data['atom_i_list']:
            return []
        
        edge_list = []
        num_edges = len(edge_data['atom_i_list'])
        
        for i in range(num_edges):
            distance_val = edge_data['distances'][i]
            if isinstance(distance_val, (tuple, list)):
                print(f"DEBUG: Tuple found in distances[{i}]: {distance_val} (type: {type(distance_val)})")
                distance_val = distance_val[0] if len(distance_val) > 0 else 0.0
            
            edge = {
                "atom_i": edge_data['atom_i_list'][i],
                "atom_j": edge_data['atom_j_list'][i],
                "distance": distance_val,
                "edge_type": edge_data['edge_types'][i],
                "bond_order": edge_data['bond_orders'][i],
                "conventional_bond_order": edge_data['conventional_bond_orders'][i],
                "num_2C_BDs": edge_data['num_2C_BDs'][i],
                "bonding_orbital_occupancy": edge_data['bonding_orbital_occupancies'][i],
                "bonding_orbital_energy": edge_data['bonding_orbital_energies'][i],
                "antibonding_orbital_occupancy": edge_data['antibonding_orbital_occupancies'][i],
                "antibonding_orbital_energy": edge_data['antibonding_orbital_energies'][i],
            }
            edge_list.append(edge)
            
        return edge_list
    
    def get_edge_features_ML(self, bond_order_threshold: float = 0.4, max_distance: float = 3.0) -> list[dict]:
        """
        Extracts edge features for machine learning tasks.

        Args:
            bond_order_threshold (float): Minimum Wiberg bond order to consider a bond. Default is 0.4.
            max_distance (float): Maximum allowed bond length (in Å) to include an edge. Default is 3.0.

        Returns:
            list[dict]: List of dictionaries containing edge features.
        """
        edge_data = self.qm_data.get_precomputed_edge_features(bond_order_threshold, max_distance)
        if not edge_data or not edge_data['atom_i_list']:
            return []
        
        ML_edge_list = []
        num_edges = len(edge_data['atom_i_list'])
        
        for i in range(num_edges):
            distance_val = edge_data['distances'][i]
            if isinstance(distance_val, (tuple, list)):
                print(f"DEBUG: Tuple found in distances[{i}]: {distance_val} (type: {type(distance_val)})")
                distance_val = distance_val[0] if len(distance_val) > 0 else 0.0
            
            edge = {
                "atom_i": edge_data['atom_i_list'][i],
                "atom_j": edge_data['atom_j_list'][i],
                "distance": distance_val,
                # "edge_type": edge_data['edge_types'][i], # string value
                "bond_order": edge_data['bond_orders'][i],
                "conventional_bond_order": edge_data['conventional_bond_orders'][i],
                "num_2C_BDs": edge_data['num_2C_BDs'][i],
                "bonding_orbital_occupancy": edge_data['bonding_orbital_occupancies'][i],
                "bonding_orbital_energy": edge_data['bonding_orbital_energies'][i],
                "antibonding_orbital_occupancy": edge_data['antibonding_orbital_occupancies'][i],
                "antibonding_orbital_energy": edge_data['antibonding_orbital_energies'][i],
            }
            ML_edge_list.append(edge)
            
        return ML_edge_list


class Targets(GraphBase):
    """
    Extracts and stores thermodynamic and spectroscopic target properties from QM data.

    Includes polarizability, vibrational frequencies, thermochemical corrections, and related properties.
    """
    def __init__(self, qm_data: DictData):
        """
        Initializes Targets with thermodynamic and spectroscopic properties from DictData.

        Args:
            qm_data (DictData): Quantum mechanical data object.
        """
        super().__init__(qm_data)
        
        if self.graph_type in ["NBO", "QM"]:
            npa_totals = qm_data.natural_population_totals or {}
            npa_total_keys = {
                'natural_minimal_basis': 'natural_minimal_basis',
                'natural_rydberg_basis': 'natural_rydberg_basis',
                'total_core_population': 'total_core_population',
                'total_valence_population': 'total_valence_population',
                'total_rydberg_population': 'total_rydberg_population',
                'total_population': 'total_population'
            }
            npa_total_data = {key: npa_totals.get(dict_key) for key, dict_key in npa_total_keys.items()}
            for attr, value in npa_total_data.items():
                setattr(self, attr, value)
            
        self.polarizability = qm_data.polarizability

        self.frequencies = qm_data.frequencies
        self.num_frequencies = len(qm_data.frequencies) if qm_data.frequencies else 0
        self.lowest_frequency = get_lowest_vibrational_frequency(qm_data.frequencies) if qm_data.frequencies else None
        self.highest_frequency = get_highest_vibrational_frequency(qm_data.frequencies) if qm_data.frequencies else None

        self._extract_indexed_values(qm_data.moments_of_inertia, [0, 1, 2], 'moment')
        self._extract_indexed_values(qm_data.rotational_constants, [0, 1, 2], 'rot')  
        self._extract_indexed_values(qm_data.rotational_temperatures, [0, 1, 2], 'rot_temp')

        thermo_attrs = {
            'heat_capacity_Cv': qm_data.heat_capacity_Cv,
            'heat_capacity_Cp': qm_data.heat_capacity_Cp,
            'entropy': qm_data.entropy,
            'ZPE': qm_data.ZPE,
            'electronic_energy': qm_data.electronic_energy,
            'potential_energy': qm_data.potential_energy,
            'enthalpy': qm_data.enthalpy,
            'gibbs_free_energy': qm_data.gibbs_free_energy
        }
        for attr, value in thermo_attrs.items():
            setattr(self, attr, value)
            
        if self.electronic_energy is not None:
            self.potential_energy_correction = (self.potential_energy - self.electronic_energy) if self.potential_energy else None
            self.enthalpy_correction = (self.enthalpy - self.electronic_energy) if self.enthalpy else None  
            self.gibbs_free_energy_correction = (self.gibbs_free_energy - self.electronic_energy) if self.gibbs_free_energy else None
        else:
            self.potential_energy_correction = None
            self.enthalpy_correction = None
            self.gibbs_free_energy_correction = None
        
    def get_targets(self) -> dict:
        """
        Returns all target properties as a dictionary.

        Returns:
            dict: Dictionary of all target properties.
        """
        targets_dict = {
            "frequencies": self.frequencies,
            "num_frequencies": self.num_frequencies,
            "lowest_frequency": self.lowest_frequency,
            "highest_frequency": self.highest_frequency,
            "moment_1": self.moment_1,
            "moment_2": self.moment_2,
            "moment_3": self.moment_3,
            "rot_1": self.rot_1,
            "rot_2": self.rot_2,
            "rot_3": self.rot_3,
            "rot_temp_1": self.rot_temp_1,
            "rot_temp_2": self.rot_temp_2,
            "rot_temp_3": self.rot_temp_3,
            "heat_capacity_Cv": self.heat_capacity_Cv,
            "heat_capacity_Cp": self.heat_capacity_Cp,
            "entropy": self.entropy,
            "ZPE": self.ZPE,
            "electronic_energy": self.electronic_energy,
            "potential_energy": self.potential_energy,
            "potential_energy_correction": self.potential_energy_correction,
            "enthalpy": self.enthalpy,
            "enthalpy_correction": self.enthalpy_correction,
            "gibbs_free_energy": self.gibbs_free_energy,
            "gibbs_free_energy_correction": self.gibbs_free_energy_correction
        }
        
        if self.graph_type == "NBO" or self.graph_type == "QM":
            targets_dict["natural_minimal_basis"] = self.natural_minimal_basis
            targets_dict["natural_rydberg_basis"] = self.natural_rydberg_basis
            targets_dict["total_core_population"] = self.total_core_population
            targets_dict["total_valence_population"] = self.total_valence_population
            targets_dict["total_rydberg_population"] = self.total_rydberg_population
            targets_dict["total_population"] = self.total_population
            
        return targets_dict
    
    def get_targets_ML(self) -> dict:
        """
        Returns all target properties as a dictionary.

        Returns:
            dict: Dictionary of all target properties.
        """
        ML_targets_dict = {
            # "frequencies": self.frequencies, # Exclude full frequency list for ML to avoid variable-length issues
            "num_frequencies": self.num_frequencies,
            "lowest_frequency": self.lowest_frequency,
            "highest_frequency": self.highest_frequency,
            "moment_1": self.moment_1,
            "moment_2": self.moment_2,
            "moment_3": self.moment_3,
            "rot_1": self.rot_1,
            "rot_2": self.rot_2,
            "rot_3": self.rot_3,
            "rot_temp_1": self.rot_temp_1,
            "rot_temp_2": self.rot_temp_2,
            "rot_temp_3": self.rot_temp_3,
            "heat_capacity_Cv": self.heat_capacity_Cv,
            "heat_capacity_Cp": self.heat_capacity_Cp,
            "entropy": self.entropy,
            "ZPE": self.ZPE,
            "electronic_energy": self.electronic_energy,
            "potential_energy": self.potential_energy,
            "potential_energy_correction": self.potential_energy_correction,
            "enthalpy": self.enthalpy,
            "enthalpy_correction": self.enthalpy_correction,
            "gibbs_free_energy": self.gibbs_free_energy,
            "gibbs_free_energy_correction": self.gibbs_free_energy_correction
        }
        
        if self.graph_type == "NBO" or self.graph_type == "QM":
            ML_targets_dict["natural_minimal_basis"] = self.natural_minimal_basis
            ML_targets_dict["natural_rydberg_basis"] = self.natural_rydberg_basis
            ML_targets_dict["total_core_population"] = self.total_core_population
            ML_targets_dict["total_valence_population"] = self.total_valence_population
            ML_targets_dict["total_rydberg_population"] = self.total_rydberg_population
            ML_targets_dict["total_population"] = self.total_population

        return ML_targets_dict
        
