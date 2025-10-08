from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import regex as re
import mmap
from typing import Union, List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from pathlib import Path

# Pre-compiled regex patterns at module level with regex module over standard re
PATTERNS = {
    'principal_moments': re.compile(r"Principal moments of inertia", re.IGNORECASE),
    'rotational_constants': re.compile(r"Rotational constants relative to principal axes", re.IGNORECASE),
    'rotational_temperatures': re.compile(r"Rotational temperatures \(K\):", re.IGNORECASE),
    'cv': re.compile(r"Total CV:", re.IGNORECASE),
    'cp': re.compile(r"Total CP:", re.IGNORECASE),
    'entropy': re.compile(r"Total S:", re.IGNORECASE),
    'zpe': re.compile(r"Zero point energy \(ZPE\):", re.IGNORECASE),
    'electronic_energy': re.compile(r"Electronic energy:", re.IGNORECASE),
    'potential_energy': re.compile(r"Sum of electronic energy and thermal correction to U:", re.IGNORECASE),
    'enthalpy': re.compile(r"Sum of electronic energy and thermal correction to H:", re.IGNORECASE),
    'gibbs_free_energy': re.compile(r"Sum of electronic energy and thermal correction to G:", re.IGNORECASE),
    'frequencies': re.compile(r"frequencies \(cm\^-1\):", re.IGNORECASE),
    'wiberg_matrix': re.compile(r"Wiberg-Mayer bond indices", re.IGNORECASE),
    'bd_orbitals': re.compile(r"Number of two-center\(2C\) BD orbitals", re.IGNORECASE),
    'npa_charges': re.compile(r"Nuclear.*Electron.*NPA", re.IGNORECASE),
    'clpo_summary': re.compile(r"\*\*\* Summary of CLPO results", re.IGNORECASE),
    'nbo_summary': re.compile(r"NATURAL BOND ORBITALS \(Summary\):", re.IGNORECASE),
    'lewis_start': re.compile(r"------ Lewis", re.IGNORECASE),
    'nonlewis_start': re.compile(r"------ non-Lewis", re.IGNORECASE),
    'npa_summary': re.compile(r"Summary of Natural Population Analysis:", re.IGNORECASE),
    'npa_totals': re.compile(r"\* Total \*", re.IGNORECASE),
    'centr_header': re.compile(r"\s*Centr\. A/B", re.IGNORECASE),
    'digit_line': re.compile(r"\s*\d+"),
    'asterisk_end': re.compile(r'^\*{2,}'),
    # CLPO parsing patterns
    'clpo_bd_lp_io': re.compile(r"^\d+\s+\((BD|LP)\)\s+(.+?)\s+Io\s+=\s+[\d.]+\s+([\d.]+)", re.IGNORECASE),
    'clpo_lp_simple': re.compile(r"^\d+\s+\(LP\)\s+(.+?)\s+([\d.]+)", re.IGNORECASE),
    'clpo_nb': re.compile(r"^\d+\s+(.+?), antibonding \(NB\)\s+([\d.]+)", re.IGNORECASE),
    # NBO parsing patterns
    'cr_nbo': re.compile(r"^\s*\d+\.\s+CR\s*\(\s*\d+\)\s+([A-Za-z]+)\s+(\d+)\s+([\d.\-]+)\s+([\d.\-]+)", re.IGNORECASE),
    'lp_nbo': re.compile(r"^\s*\d+\.\s+LP\s*\(\s*\d+\)\s+([A-Za-z]+)\s+(\d+)\s+([\d.\-]+)\s+([\d.\-]+)", re.IGNORECASE),
}

# Element regex for NBO BD patterns
ELEMENT_REGEX = r"H|B|C|N|O|F|Si|P|S|Cl|Br|I"

BD_PATTERN = re.compile(
    rf"""
    ^\s*\d+\.\s+BD\s*\(\s*\d+\)\s*
    (?:({ELEMENT_REGEX})\s*(\d+)[- ]\s*({ELEMENT_REGEX})\s*(\d+)|({ELEMENT_REGEX})\s*(\d+)[- ]\s*({ELEMENT_REGEX})\s*(\d+))
    \s+([\d.\-]+)\s+([\d.\-]+)
    """,
    re.VERBOSE | re.IGNORECASE
)

BDSTAR_PATTERN = re.compile(
    rf"""
    ^\s*\d+\.\s+BD\*\s*\(\s*\d+\)\s*
    (?:({ELEMENT_REGEX})\s*(\d+)[- ]\s*({ELEMENT_REGEX})\s*(\d+)|({ELEMENT_REGEX})\s*(\d+)[- ]\s*({ELEMENT_REGEX})\s*(\d+))
    \s+([\d.\-]+)\s+([\d.\-]+)
    """,
    re.VERBOSE | re.IGNORECASE
)

@contextmanager
def mmap_file(file_path: Union[str, Path]):
    """Context manager for memory-mapped file reading."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Convert bytes to string for easier processing
            content = mm.read().decode('utf-8')
            yield content

# Global file cache
_file_cache = {}

def get_file_content(file_path: Union[str, Path]) -> str:
    """
    Returns file content with caching.

    Args:
        file_path (Union[str, Path]): Path to the file.
    
    Returns:
        str: Content of the file.
    """
    file_path = Path(file_path)
    file_key = str(file_path)
    if file_key not in _file_cache:
        with mmap_file(file_path) as content:
            _file_cache[file_key] = content
    return _file_cache[file_key]

def clear_file_cache():
    """
    Clears the file content cache in order to free memory.
    """
    global _file_cache
    _file_cache.clear()

def get_atomic_data_dicts():
    """
    Returns dictionaries mapping atomic numbers to atomic masses (in amu) and atomic symbols.

    Returns:
        tuple: (atomic_masses, atoms)
            - atomic_masses (dict): {atomic_number (int): atomic_mass (float)}
            - atoms (dict): {atomic_number (int): atomic_symbol (str)}
    
    References:
        Lide, D. R.; Baysinger, G.; Chemistry, S.; Berger, L. I.; Goldberg, R. N.; Kehiaian, H. V. CRC Handbook of Chemistry and Physics 2005.
    """
    atomic_masses = {
        6: 12.0107, 17: 35.453, 9: 18.9984032, 1: 1.00794, 7: 14.0067, 8: 15.9994,
        16: 32.065, 5: 10.811, 14: 28.0855, 15: 30.973761, 35: 79.904, 53: 126.90447
    }
    atomic_symbols = {
        6: "C", 17: "Cl", 9: "F", 1: "H", 7: "N", 8: "O", 16: "S", 5: "B",
        14: "Si", 15: "P", 35: "Br", 53: "I"
    }
    return atomic_masses, atomic_symbols

# Parsing functions for .xyz files and SMILES strings
def read_xyz_file(xyz_file: Union[str, Path]):
    """
    Given an .xyz file, this function reads the atomic numbers (1, 2, etc.) and the xyz coordinates for each. 
    
    Args: 
        xyz_file (str): Path to a .xyz file
        
    Returns: 
        tuple: A tuple containing:
            - atom_list (List[List[float]]): Each sublist contains [atomic_number, x, y, z].
            - atomic_numbers (List[int]): Atomic numbers of all atoms in the molecule [1, 2, ..., 11].
            - xyz_coordinates (List[List[float]]): Each sublist contains [x, y, z].
            - num_atoms (int): Total number of atoms in the molecule (11).
            - atom_labels (List[str]): Atom labels, e.g., ['C1', 'C2', ..., 'H11'].
            
    Raises:
        ValueError: If an unknown atomic element is encountered in the file.

    References:
        Rasmussen, M.H., Strandgaard, M., Seumer, J., Hemmingsen, L. K., Frei, A., Balcells, D., Jensen, J. H. SMILES all around: Structure to SMILES conversion for transition metal complexes. Journal of Cheminformatics 2025, 17 (63). https://doi.org/10.1186/s13321-025-01008-1
    """
    xyz_file = Path(xyz_file)
    atom_list = []
    atomic_numbers = []
    xyz_coordinates = []
    atom_labels = []
    
    atomic_masses, atomic_symbols = get_atomic_data_dicts()

    elements = {atomic_symbols[k]: k for k in atomic_symbols}
    
    lowercase_elements = {symbol.lower(): atomic_num for symbol, atomic_num in elements.items()}

    with xyz_file.open("r", encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number < 2:
                continue
            else:
                atomic_symbol, x, y, z = line.split()
                key = atomic_symbol.lower()
                if key not in lowercase_elements:
                    raise ValueError(f"Unknown element: {atomic_symbol}")
                atomic_numbers.append(lowercase_elements[key])
                atom = lowercase_elements[key]
                atom_list.append([atom, float(x), float(y), float(z)])
                xyz_coordinates.append([float(x), float(y), float(z)])
                atom_labels.append(f"{atomic_symbol}{line_number - 1}")

    return atom_list, atomic_numbers, xyz_coordinates, num_atoms, atom_labels

def get_molecular_weight(smiles_string):
    """
    Returns the molecular weight of a molecule given its SMILES string.
    
    Args:
        smiles_string (str): The SMILES string of the molecule
        
    Returns:
        float: The molecular weight of a molecule (in amu)
        
    Raises:
        ValueError: If the SMILES string cannot be parsed into an RDKit molecule.
    """
    molecule = Chem.MolFromSmiles(smiles_string)
    mol_weight = Chem.Descriptors.ExactMolWt(molecule)
    if not mol_weight:
        raise ValueError("The SMILES string does not represent a valid molecule.")
    return mol_weight

# Shermo output parsing functions
def get_principal_moments_of_inertia(shermo_output: Union[str, Path]):
    """
    Returns the principal moments of inertia from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.
        
    Returns: 
        - list[float]: A list of the 3 principal moments of inertia (in amu*Bohr^2).
        - None if no value is found.
    """
    content = get_file_content(shermo_output)
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if PATTERNS['principal_moments'].search(line):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                return [float(x) for x in next_line.split()]
    return None

def get_rotational_constants(shermo_output: Union[str, Path]):
    """
    Returns the rotational constants relative to the principal axes of a molecule from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.
        
    Returns: 
        - list[float]: A list of the 3 rotational constants relative to principal axes (in GHz).
        - None if no value is found.
    """
    content = get_file_content(shermo_output)
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if PATTERNS['rotational_constants'].search(line):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                return [float(x) for x in next_line.split()]
    return None

def get_rotational_temperatures(shermo_output: Union[str, Path]):
    """
    Returns the rotational temperatures of a molecule from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.
        
    Returns: 
        - list[float]: A list of the 3 rotational temperatures (in K).
        - None if no value is found.
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['rotational_temperatures'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        rot_temps = [float(x) for x in line.split(':')[1].split()]
        return rot_temps
    return None

def get_Cv(shermo_output: Union[str, Path]):
    """
    Returns the heat capacity at constant volume (Cv) from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.

    Returns: 
        float: The heat capacity at constant volume (Cv, in J/mol/K).
        
    Raises:
        ValueError: if no value is found in the file
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['cv'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        return float(line.split()[2])
    raise ValueError("Cv not found.")

def get_Cp(shermo_output: Union[str, Path]):
    """
    Returns the heat capacity at constant pressure (Cp) from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.

    Returns: 
        float: The heat capacity at constant pressure (Cp, in J/mol/K).
        
    Raises:
        ValueError: if no value is found in the file
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['cp'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        return float(line.split()[2])
    raise ValueError("Cp not found.")

def get_entropy(shermo_output: Union[str, Path]):
    """
    Returns the total entropy (S) from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.
        
    Returns: 
        float: The total entropy (S, in J/mol/K).
        
    Raises:
        ValueError: if no value is found in the file
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['entropy'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        return float(line.split()[2])
    raise ValueError("Entropy not found.")

def get_ZPE(shermo_output: Union[str, Path]):
    """
    Returns the zero point energy (ZPE) from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.
        
    Returns: 
        float: The zero point energy (ZPE, in Eh).
        
    Raises:
        ValueError: if no value is found in the file
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['zpe'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        return float(line.split()[-2])
    raise ValueError("Zero point energy (ZPE) not found.")

def get_electronic_energy(shermo_output: Union[str, Path]):
    """
    Returns the electronic energy from a Shermo output file.

    Args:
        shermo_output (str): Path to a Shermo output file.

    Returns:
        float: The electronic energy (in Eh).

    Raises:
        ValueError: if no value is found in the file.
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['electronic_energy'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        return float(line.split()[-2])
    raise ValueError("Electronic energy not found.")

def get_potential_energy(shermo_output: Union[str, Path]):
    """
    Returns the potential energy (U) from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.
        
    Returns: 
        float: The potential energy (U, in Eh).
        
    Raises:
        ValueError: if no value is found in the file.
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['potential_energy'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        return float(line.split()[-2])
    raise ValueError("Potential energy (U) not found.")

def get_enthalpy(shermo_output: Union[str, Path]):
    """
    Returns the enthalpy (H) from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.
        
    Returns: 
        float: The enthalpy (H, in Eh).
        
    Raises:
        ValueError: if no value is found in the file.
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['enthalpy'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        return float(line.split()[-2])
    raise ValueError("Enthalpy (H) not found.")

def get_Gibbs_free_energy(shermo_output: Union[str, Path]):
    """
    Returns the Gibbs free energy (G) from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.

    Returns: 
        float: The Gibbs free energy (G, in Eh).
        
    Raises:
        ValueError: if no value is found in the file
    """
    content = get_file_content(shermo_output)
    
    match = PATTERNS['gibbs_free_energy'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        return float(line.split()[-2])
    raise ValueError("Gibbs free energy (G) not found.")

def get_vibrational_frequencies(shermo_output: Union[str, Path]):
    """
    Returns the vibrational frequencies of a molecule from a Shermo output file.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.

    Returns: 
        list[float]: A list of vibrational frequencies present in the molecule (in cm^-1).
        
    Raises:
        ValueError: if no value is found in the file
    """
    content = get_file_content(shermo_output)
    frequencies = []
    
    match = PATTERNS['frequencies'].search(content)
    if match:
        start_pos = content.find('\n', match.end()) + 1
        lines = content[start_pos:].split('\n')
        
        for line in lines:
            stripped = line.strip()
            if not stripped or not any(char.isdigit() for char in stripped):
                break
            try:
                freqs = [float(val) for val in stripped.split()]
                frequencies.extend(freqs)
            except ValueError:
                break
    
    if not frequencies:
        raise ValueError("No vibrational frequencies found.")
    return frequencies

# Batch function for Shermo data
def get_all_shermo_data(shermo_output: Union[str, Path]):
    """
    Retrieve all Shermo thermodynamic data in one pass.
    
    Args: 
        shermo_output (str): Path to a Shermo output file.
    
    Returns:
        dict: A dictionary containing all Shermo data with keys:
            - 'principal_moments' (list[float] or None)
            - 'rotational_constants' (list[float] or None)
            - 'rotational_temperatures' (list[float] or None)
            - 'cv' (float or None)
            - 'cp' (float or None)
            - 'entropy' (float or None)
            - 'zpe' (float or None)
            - 'electronic_energy' (float or None)
            - 'potential_energy' (float or None)
            - 'enthalpy' (float or None)
            - 'gibbs_free_energy' (float or None)
            - 'frequencies' (list[float])
    """
    content = get_file_content(shermo_output)
    
    data = {}
    
    for key, pattern in PATTERNS.items():
        if key in ['principal_moments', 'rotational_constants', 'rotational_temperatures']:
            continue 
            
        match = pattern.search(content)
        if match:
            line = content[match.start():content.find('\n', match.start())]
            try:
                if key in ['cv', 'cp', 'entropy']:
                    data[key] = float(line.split()[2])
                elif key in ['zpe', 'electronic_energy', 'potential_energy', 'enthalpy', 'gibbs_free_energy']:
                    data[key] = float(line.split()[-2])
            except (ValueError, IndexError):
                data[key] = None
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if PATTERNS['principal_moments'].search(line) and i + 1 < len(lines):
            try:
                data['principal_moments'] = [float(x) for x in lines[i + 1].strip().split()]
            except (ValueError, IndexError):
                data['principal_moments'] = None
        
        elif PATTERNS['rotational_constants'].search(line) and i + 1 < len(lines):
            try:
                data['rotational_constants'] = [float(x) for x in lines[i + 1].strip().split()]
            except (ValueError, IndexError):
                data['rotational_constants'] = None
    
    match = PATTERNS['rotational_temperatures'].search(content)
    if match:
        try:
            line = content[match.start():content.find('\n', match.start())]
            data['rotational_temperatures'] = [float(x) for x in line.split(':')[1].split()]
        except (ValueError, IndexError):
            data['rotational_temperatures'] = None
    
    data['frequencies'] = get_vibrational_frequencies(shermo_output)
    
    return data

# JANPA output parsing functions
def get_wiberg_bond_order_matrix(janpa_output: Union[str, Path]):
    """
    Returns the Wiberg bond order matrix from a JANPA output file.
    
    Args:
        janpa_output (str): Path to a JANPA output file
        
    Returns:
        list[list[float]]: A symmetric 2D matrix of bond orders.
        
    Raises:
        ValueError if the matrix is not found or incomplete
    """
    content = get_file_content(janpa_output)
    lines = content.split('\n')
    
    matrix_started = False
    rows = []
    atom_count = 0

    for line in lines:
        if PATTERNS['wiberg_matrix'].search(line):
            matrix_started = True
            continue

        if matrix_started:
            if PATTERNS['centr_header'].match(line):
                continue
            if line.strip() == "" or "*" in line:
                break

            line_clean = re.sub(r"[()]", "", line)
            parts = line_clean.strip().split()

            if len(parts) < 2:
                continue

            try:
                values = list(map(float, parts[1:]))
                rows.append(values)
                atom_count += 1
            except ValueError:
                continue

    if not rows:
        raise ValueError("Wiberg bond order matrix not found.")
    
    full_matrix = [[0.0 for _ in range(atom_count)] for _ in range(atom_count)]
    
    for i in range(atom_count):
        for j, val in enumerate(rows[i]):
            full_matrix[i][i + j] = val
            full_matrix[i + j][i] = val

    return full_matrix

def get_number_of_2C_BDs_matrix(janpa_output: Union[str, Path]):
    """
    Returns the number of two-center (2C) bonding orbitals matrix from a JANPA output file.
    
    Args:
        janpa_output (str): Path to a JANPA output file.
        
    Returns:
        list[list[int]]: The symmetric 2D matrix of the number of two-center(2C) BD orbitals for each pair of atoms in the molecule.
        
    Raises:
        ValueError if the matrix is not found or incomplete
    """
    content = get_file_content(janpa_output)
    lines = content.split('\n')
    
    last_rows = []
    last_atom_count = 0
    matrix_started = False
    current_rows = []
    current_atom_count = 0

    for line in lines:
        if PATTERNS['bd_orbitals'].search(line):
            matrix_started = True
            current_rows = []
            current_atom_count = 0
            continue
            
        if matrix_started:
            if PATTERNS['centr_header'].match(line):
                continue
            if not line.strip() or not PATTERNS['digit_line'].match(line):
                if current_rows:
                    last_rows = current_rows
                    last_atom_count = current_atom_count
                matrix_started = False
                continue
                
            parts = line.strip().split()
            if not parts or len(parts) < 2:
                continue
                
            row_data = []
            for part in parts[2:]:
                part = part.strip()
                if not part:
                    continue
                try:
                    value = int(part.strip("()"))
                    row_data.append(value)
                except ValueError:
                    raise ValueError(f"Invalid entry '{part}' encountered in matrix.")
            current_rows.append(row_data)
            current_atom_count += 1

    if not last_rows:
        raise ValueError("Two-center BD orbital matrix not found.")

    full_matrix = [[0 for _ in range(last_atom_count)] for _ in range(last_atom_count)]
    
    for i in range(last_atom_count):
        row_data = last_rows[i]
        
        # Handle diagonal element (first element in row_data)
        if row_data:
            full_matrix[i][i] = row_data[0]  # This is the (4) value from parentheses
        
        # Handle off-diagonal elements
        for j, val in enumerate(row_data[1:], start=1):  # Skip first element (diagonal)
            col_idx = i + j  # Correct column indexing for upper triangular
            if col_idx >= last_atom_count:
                break
            full_matrix[i][col_idx] = val
            full_matrix[col_idx][i] = val  # Make symmetric
            
    return full_matrix

def get_natural_population_analysis_charges(janpa_output: Union[str, Path]):
    """
    Parses the natural population analysis charges from a JANPA output file.
    
    Args:
        janpa_output (str): Path to a JANPA output file.
        
    Returns:
        list[list[float]]: A list of lists containing:
            - (float): The electron population for each atom.
            - (float): The NMB population for each atom.
            - (float): The NPA charge for each atom.
            
    Raises:
        ValueError: If the NPA charge data is not found or is malformed.
    """
    content = get_file_content(janpa_output)
    lines = content.split('\n')
    
    charges = []
    parsing = False
    
    for line in lines:
        if PATTERNS['npa_charges'].search(line):
            parsing = True
            continue
        if parsing:
            if not line.strip() or not line.lstrip()[0].isalnum():
                break
            parts = line.split()
            try:
                electron_population = float(parts[2])
                nmb_population = float(parts[3])
                npa_charge = float(parts[4])
                charges.append([electron_population, nmb_population, npa_charge])
            except (ValueError, IndexError):
                continue
    
    if not charges:
        raise ValueError("No NPA charges found.")
    return charges

def get_orbital_occupancies(janpa_output: Union[str, Path]):
    """
    Extracts the s, p, d, and f orbital occupancies for each atom in a molecule from a JANPA output file.
    
    Args:
        janpa_output (str): Path to a JANPA output file.
        
    Returns:
        list[list[float]]: 4-element lists containing the orbital occupancies in the order of s, p, d, and f symmetries.
            
    Raises:
        ValueError: If the orbital occupancy data is not found or is malformed.
    """
    content = get_file_content(janpa_output)
    lines = content.split('\n')
    
    orbital_occupancies = []
    in_population_block = False
    
    for line in lines:
        if "Cntr" in line:
            in_population_block = True
            continue
        if in_population_block:
            if line.strip() == "" or "*" in line:
                break
            if "Cntr" in line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    s = float(parts[1])
                    p = float(parts[2])
                    d = float(parts[3])
                    f_ = float(parts[4])
                    orbital_occupancies.append([s, p, d, f_])
                except ValueError:
                    continue
    return orbital_occupancies

def analyze_CLPOs(janpa_output: Union[str, Path]):
    """
    Extracts the CLPO ID, type (BD, NB, LP), atom indices involved, and occupancy for each CLPO (Chemical Localized Property Orbitals) from a JANPA output file.
    
    Args: 
        janpa_output (str): Path to a JANPA output file.
        
    Returns:
        list[list[str, str, list[int], float]]: A list of lists, where each sublist contains:
        - CLPO_id (int): Unique identifier for the CLPO.
        - CLPO_type (str): The type of CLPO (BD, NB, LP).
        - atom_indices (list[int]): A list of atom indices involved in the CLPO.
        - occupancy (float): The occupancy of the CLPO.
    """
    content = get_file_content(janpa_output)
    lines = content.split('\n')
    
    results = []
    in_clpo_section = False
    clpo_id = 0
    
    for line in lines:
        line = line.strip()
        if PATTERNS['clpo_summary'].search(line):
            in_clpo_section = True
            continue
        if in_clpo_section and PATTERNS['asterisk_end'].match(line):
            break
        if in_clpo_section:
            if '(RY)' in line:
                continue
                
            # BD or LP CLPOs
            match_bd_lp = PATTERNS['clpo_bd_lp_io'].match(line)
            if match_bd_lp:
                clpo_type = match_bd_lp.group(1)
                atoms_raw = match_bd_lp.group(2)
                occupancy = float(match_bd_lp.group(3))
                matches = re.findall(r"([A-Z][a-z]?)(\d+)", atoms_raw)
                atom_indices = [int(index) for _, index in matches]
                results.append([clpo_id, clpo_type, atom_indices, occupancy])
                clpo_id += 1
                continue
                
            # LP CLPOs
            match_lp = PATTERNS['clpo_lp_simple'].match(line)
            if match_lp:
                clpo_type = "LP"
                atoms_raw = match_lp.group(1)
                occupancy = float(match_lp.group(2))
                matches = re.findall(r"([A-Z][a-z]?)(\d+)", atoms_raw)
                atom_indices = [int(index) for _, index in matches]
                results.append([clpo_id, clpo_type, atom_indices, occupancy])
                clpo_id += 1
                continue
                
            # NB CLPOs (antibonding)
            match_nb = PATTERNS['clpo_nb'].match(line)
            if match_nb:
                clpo_type = "AB"
                atoms_raw = match_nb.group(1)
                occupancy = float(match_nb.group(2))
                matches = re.findall(r"([A-Z][a-z]?)(\d+)", atoms_raw)
                atom_indices = [int(index) for _, index in matches]
                results.append([clpo_id, clpo_type, atom_indices, occupancy])
                clpo_id += 1
                continue
    return results

def get_all_janpa_data(janpa_output: Union[str, Path]):
    """
    Retrieves all JANPA data in one pass.
    
    Args:
        janpa_output (str): Path to a JANPA output file.
    
    Returns:
        dict: A dictionary containing all JANPA data with keys:
            - 'wiberg_matrix' (list[list[float]])
            - 'bond_matrix' (list[list[int]])
            - 'npa_charges' (list[list[float]])
            - 'orbital_occupancies' (list[list[float]])
            - 'clpo_data' (list[list[str, str, list[int], float]])
    """
    data = {}
    try:
        data['wiberg_matrix'] = get_wiberg_bond_order_matrix(janpa_output)
        data['bond_matrix'] = get_number_of_2C_BDs_matrix(janpa_output)
        data['npa_charges'] = get_natural_population_analysis_charges(janpa_output)
        data['orbital_occupancies'] = get_orbital_occupancies(janpa_output)
        data['clpo_data'] = analyze_CLPOs(janpa_output)
    except Exception as e:
        print(f"Warning: Error parsing JANPA data: {e}")
    return data

# NBO output parsing functions
def get_natural_population_analysis(nbo_output: Union[str, Path]):
    """
    Parses the natural population analysis summary from NBO analysis.
    
    Args:
        nbo_output (str): Path to an NBO output file.

    Returns:
        list[dict]: A list of dictionaries for each atom in the molecule, each containing:
            - 'atom_number': Atomic number of the atom.
            - 'element': Element symbol of the atom.
            - 'natural_charge': Natural charge of the atom.
            - 'core_population': Core population of the atom.
            - 'valence_population': Valence population of the atom.
            - 'rydberg_population': Rydberg population of the atom.
            - 'total_population': Total population of the atom.

    Raises:
        ValueError if the natural population analysis summary is not found or incomplete
    """
    content = get_file_content(nbo_output)
    
    start_marker = "Summary of Natural Population Analysis:"
    end_marker = "==============="
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        raise ValueError("No natural population analysis summary found.")
    
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        end_idx = len(content)
    
    section = content[start_idx:end_idx]
    lines = section.split('\n')[1:]
    
    atoms = []
    for line in lines:
        if not line.strip() or line.strip().startswith("-") or line.strip().startswith("Atom ") or line.strip().startswith("Natural") or "Total" in line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        atom = {
            'atom_number': int(parts[1]),
            'element': parts[0],
            'natural_charge': float(parts[2]),
            'core_population': float(parts[3]),
            'valence_population': float(parts[4]),
            'rydberg_population': float(parts[5]),
            'total_population': float(parts[6])
        }
        atoms.append(atom)
    
    if not atoms:
        raise ValueError("No natural population analysis summary found.")
    return atoms

def get_natural_population_totals(nbo_output: Union[str, Path]):
    """
    Parses the natural population totals from NBO analysis.

    Args:
        nbo_output (str): Path to an NBO output file.

    Returns:
        dict: A dictionary containing:
            - 'total_charge': Total charge of the molecule.
            - 'total_core_population': Total core population of the molecule.
            - 'total_valence_population': Total valence population of the molecule.
            - 'total_rydberg_population': Total Rydberg population of the molecule.
            - 'natural_minimal_basis': Total natural population in the minimal basis (core + valence).
            - 'natural_rydberg_basis': Total natural population in the Rydberg basis.
            - 'total_population': Total population of the molecule.

    Raises:
        ValueError if the natural population totals are not found or incomplete
    """
    content = get_file_content(nbo_output)
    
    match = PATTERNS['npa_totals'].search(content)
    if match:
        line = content[match.start():content.find('\n', match.start())]
        parts = line.split()
        if len(parts) < 7:
            raise ValueError("Malformed totals line in NBO output.")
        return {
            "total_charge": float(parts[3]),
            "total_core_population": float(parts[4]),
            "total_valence_population": float(parts[5]),
            "total_rydberg_population": float(parts[6]),
            "natural_minimal_basis": float(parts[4]) + float(parts[5]),
            "natural_rydberg_basis": float(parts[6]),
            "total_population": float(parts[7])
        }
    raise ValueError("No natural population totals found.")

def get_core_NBOs(nbo_output: Union[str, Path]):
    """
    Parses the core natural bond orbitals (NBOs) from NBO analysis.
    
    Args:
        nbo_output (str): Path to an NBO output file.

    Returns:
        list[dict]: A list of dictionaries for each core NBO, each containing:
            - 'atom_number': Atomic number of the atom.
            - 'element': Element symbol of the atom.
            - 'NBO_type': Type of NBO (always "CR" for core).
            - 'NBO_occupancy': Occupancy of the NBO.
            - 'energy': Energy of the NBO.

    Raises:
        ValueError if the core NBOs are not found or incomplete
    """
    content = get_file_content(nbo_output)
    lines = content.split('\n')
    
    core_NBOs = []
    in_summary = False
    in_lewis = False
    
    for line in lines:
        if not in_summary and PATTERNS['nbo_summary'].search(line):
            in_summary = True
            continue
        if in_summary and PATTERNS['lewis_start'].search(line):
            in_lewis = True
            continue
        if in_lewis and PATTERNS['nonlewis_start'].search(line):
            break
        if in_lewis:
            match = PATTERNS['cr_nbo'].match(line)
            if match:
                element = match.group(1)
                atom_number = int(match.group(2))
                occupancy = float(match.group(3))
                energy = float(match.group(4))
                core_NBOs.append({
                    "atom_number": atom_number,
                    "element": element,
                    "NBO_type": "CR",
                    "NBO_occupancy": occupancy,
                    "energy": energy
                })
    return core_NBOs

def get_lone_pair_NBOs(nbo_output: Union[str, Path]):
    """
    Parses the lone pair natural bond orbitals (NBOs) from NBO analysis.
    
    Args:
        nbo_output (str): Path to an NBO output file.

    Returns:
        list[dict]: A list of dictionaries for each lone pair NBO, each containing:
            - 'atom_number': Atomic number of the atom.
            - 'element': Element symbol of the atom.
            - 'NBO_type': Type of NBO (always "LP" for lone pairs).
            - 'NBO_occupancy': Occupancy of the NBO.
            - 'energy': Energy of the NBO.
        None if no lone pair NBOs are found.
    """
    content = get_file_content(nbo_output)
    lines = content.split('\n')
    
    lone_pair_NBOs = []
    in_summary = False
    in_lewis = False
    
    for line in lines:
        if not in_summary and PATTERNS['nbo_summary'].search(line):
            in_summary = True
            continue
        if in_summary and PATTERNS['lewis_start'].search(line):
            in_lewis = True
            continue
        if in_lewis and PATTERNS['nonlewis_start'].search(line):
            break
        if in_lewis:
            match = PATTERNS['lp_nbo'].match(line)
            if match:
                element = match.group(1)
                atom_number = int(match.group(2))
                occupancy = float(match.group(3))
                energy = float(match.group(4))
                lone_pair_NBOs.append({
                    "atom_number": atom_number,
                    "element": element,
                    "NBO_type": "LP",
                    "NBO_occupancy": occupancy,
                    "energy": energy
                })
    return lone_pair_NBOs

def get_bonding_NBOs(nbo_output: Union[str, Path]):
    """
    Parses the bonding natural bond orbitals (NBOs) from NBO analysis.
    
    Args:
        nbo_output (str): Path to an NBO output file.
        
    Returns:
        list[dict]: A list of dictionaries for each bonding NBO, each containing:
            - 'atom_number_1': Atomic number of the first atom in the bond.
            - 'atom_number_2': Atomic number of the second atom in the bond.
            - 'element_1': Element symbol of the first atom in the bond.
            - 'element_2': Element symbol of the second atom in the bond.
            - 'NBO_type': Type of NBO (always "BD" for bonding).
            - 'occupancy': Occupancy of the bonding NBO.
            - 'energy': Energy of the bonding NBO.
            
    Raises:
        ValueError: If no bonding NBOs are found in the file.
    """
    content = get_file_content(nbo_output)
    lines = content.split('\n')
    
    bonding_NBOs = []
    in_summary = False
    in_lewis = False
    
    for line in lines:
        if not in_summary and PATTERNS['nbo_summary'].search(line):
            in_summary = True
            continue
        if in_summary and PATTERNS['lewis_start'].search(line):
            in_lewis = True
            continue
        if in_lewis and PATTERNS['nonlewis_start'].search(line):
            break
        if in_lewis:
            match = BD_PATTERN.match(line)
            if match:
                if match.group(1) and match.group(3):
                    element_1 = match.group(1)
                    atom_number_1 = int(match.group(2))
                    element_2 = match.group(3)
                    atom_number_2 = int(match.group(4))
                    occupancy = float(match.group(9))
                    energy = float(match.group(10))
                elif match.group(5) and match.group(7):
                    element_1 = match.group(5)
                    atom_number_1 = int(match.group(6))
                    element_2 = match.group(7)
                    atom_number_2 = int(match.group(8))
                    occupancy = float(match.group(9))
                    energy = float(match.group(10))
                else:
                    continue
                    
                bonding_NBOs.append({
                    "atom_number_1": atom_number_1,
                    "atom_number_2": atom_number_2,
                    "element_1": element_1,
                    "element_2": element_2,
                    "NBO_type": "BD",
                    "occupancy": occupancy,
                    "energy": energy
                })
    return bonding_NBOs

def get_antibonding_NBOs(nbo_output: Union[str, Path]):
    """
    Parses the antibonding natural bond orbitals (NBOs) from NBO analysis.
    
    Args:
        nbo_output (str): Path to an NBO output file.
        
    Returns:
        list[dict]: A list of dictionaries for each antibonding NBO, each containing:
            - 'atom_number_1': Atomic number of the first atom in the antibonding orbital.
            - 'atom_number_2': Atomic number of the second atom in the antibonding  orbital.
            - 'element_1': Element symbol of the first atom in the antibonding orbital.
            - 'element_2': Element symbol of the second atom in the antibonding orbital.
            - 'NBO_type': Type of NBO ("BD*" for antibonding).
            - 'occupancy': Occupancy of the antibonding NBO.
            - 'energy': Energy of the antibonding NBO.
                          
    Note:
        Parses the non-Lewis section for BD* orbitals and, if present, parses the main section for hybridization/fragment lines. If fragment lines are missing, then those fields are set to None.
    """
    content = get_file_content(nbo_output)
    lines = content.split('\n')
    
    antibonding_NBOs = []
    in_summary = False
    in_nonlewis = False
    
    for line in lines:
        if not in_summary and PATTERNS['nbo_summary'].search(line):
            in_summary = True
            continue
        if in_summary and PATTERNS['nonlewis_start'].search(line):
            in_nonlewis = True
            continue
        if in_nonlewis and (line.strip().startswith('RY') or line.strip() == '' or line.strip().startswith('===')):
            break
        if in_nonlewis:
            match = BDSTAR_PATTERN.match(line)
            if match:
                if match.group(1) and match.group(3):
                    element_1 = match.group(1)
                    atom_number_1 = int(match.group(2))
                    element_2 = match.group(3)
                    atom_number_2 = int(match.group(4))
                    occupancy = float(match.group(9))
                    energy = float(match.group(10))
                elif match.group(5) and match.group(7):
                    element_1 = match.group(5)
                    atom_number_1 = int(match.group(6))
                    element_2 = match.group(7)
                    atom_number_2 = int(match.group(8))
                    occupancy = float(match.group(9))
                    energy = float(match.group(10))
                else:
                    continue
                    
                antibonding_NBOs.append({
                    "atom_number_1": atom_number_1,
                    "atom_number_2": atom_number_2,
                    "element_1": element_1,
                    "element_2": element_2,
                    "NBO_type": "AB",
                    "occupancy": occupancy,
                    "energy": energy
                })
    return antibonding_NBOs

def get_all_nbo_data(nbo_output: Union[str, Path]):
    """
    Retrieves all NBO data in one pass using memory-mapped file.
    
    Args:
        nbo_output (str): Path to an NBO output file.
    
    Returns:
        dict: A dictionary containing all NBO data with keys:
            - 'natural_population_analysis' (list[dict])
            - 'natural_population_totals' (dict)
            - 'core_NBOs' (list[dict])
            - 'lone_pair_NBOs' (list[dict])
            - 'bonding_NBOs' (list[dict])
            - 'antibonding_NBOs' (list[dict])
    """
    data = {}
    try:
        data['natural_population_analysis'] = get_natural_population_analysis(nbo_output)
        data['natural_population_totals'] = get_natural_population_totals(nbo_output)
        data['core_NBOs'] = get_core_NBOs(nbo_output)
        data['lone_pair_NBOs'] = get_lone_pair_NBOs(nbo_output)
        data['bonding_NBOs'] = get_bonding_NBOs(nbo_output)
        data['antibonding_NBOs'] = get_antibonding_NBOs(nbo_output)
    except Exception as e:
        print(f"Warning: Error parsing NBO data: {e}")
    return data

# Helper functions

def get_CLPO_entry_by_indices(CLPO_data: list, indices: list[int]) -> list | None:
    """
    Returns the CLPO entry that matches the atom pair, or None if not found.
    
    Args:
        CLPO_data (list): The CLPO data to search through.
        indices (list[int]): The atom indices to match.
        
    Returns:
        list | None: The CLPO entry that matches the atom pair, or None if not found.
    """
    input_pair = set(indices)
    for entry in CLPO_data:
        atom_pair = set(get_CLPO_information(entry, "indices"))
        if input_pair == atom_pair:
            return entry
    return None

def get_LP_occupancy_for_atom(lp_clpos: list, atom_index: int) -> float | None:
    """
    Returns the lone pair CLPO entry that matches the atom, or None if not found.
    
    Args:
        lp_clpos (list): The list of lone pair CLPOs.
        atom_index (int): The index of the atom to search for.
        
    Returns:
        float | None: The occupancy of the lone pair CLPO for the specified atom, or None if not found.
    """
    for entry in lp_clpos:
        if atom_index in get_CLPO_information(entry, "indices"):
            return get_CLPO_information(entry, "occupancy")
    return None

def get_CLPO_information(CLPO_entry: list, property: str):
    """
    Extracts a specified property from a CLPO (Chemical Localized Property Orbital) entry.
    
    Args:
        CLPO_entry (list): A list containing:
            - CLPO_id (int): Unique identifier for the CLPO.
            - CLPO_type (str): The type of CLPO (BD, NB, LP
            - atom_indices (list[int]): A list of atom indices involved in the CLPO.
            - occupancy (float): The occupancy of the CLPO.
        property (str): One of "id", "type", "indices", or "occupancy"
        
    Returns:
        The value of the requested property from the CLPO entry.
        
    Raises:
        ValueError: if the property is not valid. 
    """
    property_mapping = {
        "id": 0,
        "type": 1,
        "indices": 2,
        "occupancy": 3
    }
    if property not in property_mapping:
        raise ValueError(f"'{property}' is not a valid CLPO property. Choose from: {list(property_mapping.keys())}")
    return CLPO_entry[property_mapping[property]]

def get_NBO_entry_by_indices(nbo_list, indices, nbo_type=None):
    """
    Helper function to find an NBO entry by atom indices and optional NBO type.
    
    Args:
        nbo_list (list[dict]): List of NBO dictionaries.
        indices (list[int]): Atom indices to match.
        nbo_type (str, optional): NBO type to match (e.g., 'BD', 'LP', 'CR'). If None, matches any type.
        
    Returns:
        dict or None: The matching NBO entry, or None if not found.
    """
    indices_set = set(indices)
    for entry in nbo_list:
        if 'atom_number_1' in entry and 'atom_number_2' in entry:
            entry_indices = {entry['atom_number_1'], entry['atom_number_2']}
        elif 'atom_number' in entry:
            entry_indices = {entry['atom_number']}
        else:
            continue
        if indices_set == entry_indices:
            if nbo_type is None or entry.get('NBO_type') == nbo_type:
                return entry
    return None

def get_NBO_information(nbo_entry, property):
    """
    Extracts a specific property from an NBO entry or a list of NBO entries.

    Args:
        nbo_entry (dict or list): NBO dictionary or list of NBO dictionaries.
        property (str): Key to extract (e.g., 'NBO_type', 'NBO_occupancy', 'energy', etc.)

    Returns:
        The requested property value, or a list of values if nbo_entry is a list.
        Returns None if entry is None or property not found.
    """
    if nbo_entry is None:
        return None
    if isinstance(nbo_entry, list):
        return [entry.get(property, None) for entry in nbo_entry if property in entry]
    if property not in nbo_entry:
        return None
    return nbo_entry[property]

def calculate_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Helper function that calculates the distance matrix for a set of N-dimensional points using Euclidean distance, vectorized.

    Args:
        points (list[list[float]]): List of coordinates for each point, where each point is represented as a list of floats.

    Returns:
        list[list[float]]: Symmetric distance matrix.
        
    References: 
        Kneiding, H., Lukin, R., Lang, L., Reine, S., Pedersen, T. B., De Bin, R., Balcells, D. Deep learning metal complex properties with natural quantum graphs. Digital Discovery 2023, 2, 618-633, https://doi.org/10.1039/D2DD00129B
    """
    if points is None or len(points) == 0:
        return None
    
    points = np.array(points)
    n = points.shape[0]
    
    # Broadcast to create difference arrays: (n, 1, 3) - (1, n, 3) = (n, n, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # Calculate squared distances and take square root
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    return distances.tolist()  # Convert back to list for compatibility

def get_atomic_properties(atomic_numbers: list) -> tuple:
    """
    Vectorize atomic property lookups using NumPy operations.
    
    Args:
        atomic_numbers (list[int]): List of atomic numbers
        
    Returns:
        tuple: (masses, electronegativities, covalent_radii) as NumPy arrays
        
    References:
        Pauling, L. The Nature of the Chemical Bond. Cornell University Press 1960, 3rd ed.
        Cordero, B., Gomez, V., Platero-Prats, A. E., Reves, M., Echeverria, J., Cremades, E., Barragan, F., and Alvarez, S. Covalent radii revisited. Dalton Transactions 2008, 2832-2838. DOI https://doi.org/10.1039/B801115J
    """
    if atomic_numbers is None:
        return None, None, None
        
    atomic_numbers = np.array(atomic_numbers)
    
    # Create lookup dictionaries
    atomic_masses, atoms = get_atomic_data_dicts()
    electronegativity_dict = {
        1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
    }
    covalent_radii_dict = {
        1: 0.31, 5: 0.84, 6: 0.73, 7: 0.71, 8: 0.66, 9: 0.57,
        14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
    }
 
    masses = np.array([atomic_masses.get(z, 0.0) for z in atomic_numbers])
    electronegativities = np.array([electronegativity_dict.get(z, 0.0) for z in atomic_numbers])
    covalent_radii = np.array([covalent_radii_dict.get(z, 0.0) for z in atomic_numbers])
    
    return masses.tolist(), electronegativities.tolist(), covalent_radii.tolist()

def analyze_bond_orders(distance_matrix: np.ndarray, wiberg_matrix: np.ndarray = None, 
                           bond_order_threshold: float = 0.4, max_distance: float = 3.0) -> tuple:
    """
    Vectorized bond analysis to find valid edges.
    
    Args:
        distance_matrix (np.ndarray): Distance matrix
        wiberg_matrix (np.ndarray): Wiberg bond order matrix (optional)
        bond_order_threshold (float): Minimum bond order threshold
        max_distance (float): Maximum distance threshold
        
    Returns:
        tuple: (atom_i_indices, atom_j_indices, distances, bond_orders) where:
            - atom_i_indices (List[int]): List of atom i indices for each valid bond
            - atom_j_indices (List[int]): List of atom j indices for each valid bond
            - distances (List[float]): List of distances for each valid bond
            - bond_orders (List[Optional[float]]): List of bond orders for each valid bond (None if wiberg_matrix not provided)
    """
    if distance_matrix is None:
        return [], [], [], []
        
    distance_matrix = np.array(distance_matrix)
    n_atoms = distance_matrix.shape[0]
    
    # Create upper triangular mask (i < j)
    atom_i_indices, atom_j_indices = np.triu_indices(n_atoms, k=1)
    
    # Get distances for upper triangle
    distances = distance_matrix[atom_i_indices, atom_j_indices]
    
    # Apply distance filter
    distance_mask = distances < max_distance
    
    if wiberg_matrix is not None:
        wiberg_matrix = np.array(wiberg_matrix)
        bond_orders = wiberg_matrix[atom_i_indices, atom_j_indices]
        
        # Apply bond order filter
        bond_order_mask = bond_orders > bond_order_threshold
        
        # Combine filters
        valid_mask = distance_mask & bond_order_mask
        
        # Convert numpy arrays to Python native types to avoid tuple conversion issues
        filtered_distances = distances[valid_mask]
        # Ensure each distance is a proper Python float
        clean_distances = [float(d) for d in filtered_distances]
        
        filtered_bond_orders = bond_orders[valid_mask] 
        clean_bond_orders = [float(bo) if bo is not None else None for bo in filtered_bond_orders]
        
        return atom_i_indices[valid_mask].tolist(), atom_j_indices[valid_mask].tolist(), clean_distances, clean_bond_orders
    else:
        # No Wiberg matrix provided, only use distance filter
        valid_mask = distance_mask
        
        filtered_distances = distances[valid_mask]
        clean_distances = [float(d) for d in filtered_distances]
        
        return atom_i_indices[valid_mask].tolist(), atom_j_indices[valid_mask].tolist(), clean_distances, [None] * len(clean_distances)

def get_dft_edges(distance_matrix, SMILES, num_atoms):
    """
    Given a distance matrix and a SMILES string, this function returns a matrix listing the distances between bonded atoms, with non-bonded pairs set to 0.0. Bonds are determined based on the connectivity derived from the SMILES string using RDKit.
    
    Args:
        distance_matrix (list[list[float]]): Symmetric distance matrix.
        smiles (str): SMILES string of the molecule.
    
    Returns:
        tuple: (atom_i_indices, atom_j_indices, distances, edge_matrix) where:
            - atom_i_indices (list[int]): List of indices for the first atom in each bond.
            - atom_j_indices (list[int]): List of indices for the second atom in each bond.
            - distances (list[float]): List of distances corresponding to each bond.
            - edge_matrix (list[list[float]]): Matrix of distances between bonded atoms, with non-bonded pairs set to 0.0.
    """
    molecule = Chem.MolFromSmiles(SMILES)
    molecule = Chem.AddHs(molecule)
    
    atom_i_indices = []
    atom_j_indices = []
    distances = []
    
    # get connectivity, i.e., which atoms are bonded to which
    connectivity = np.zeros((num_atoms, num_atoms), dtype=int)
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        connectivity[i][j] = 1
        connectivity[j][i] = 1
        
        if i < j:
            atom_i_indices.append(i)
            atom_j_indices.append(j)
            distances.append(distance_matrix[i][j])
        
    # for each pair of atoms, if they are bonded in connectivity, use distance from distance_matrix, else use 0
    edge_matrix = np.zeros((num_atoms, num_atoms), dtype=float)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if connectivity[i][j] == 1:
                edge_matrix[i][j] = distance_matrix[i][j]
            else:
                edge_matrix[i][j] = 0.0

    return atom_i_indices, atom_j_indices, distances, edge_matrix

def get_conventional_bond_orders(bond_orders: list) -> list:
    """
    Helper function to retrieve the conventional bond order from Wiberg bond orders, vectorized.
    
    Args:
        bond_orders (list): List of Wiberg bond orders (may contain None)
        
    Returns:
        list: List of conventional bond orders (preserving None positions)
    """
    if not bond_orders:
        return []
    
    result = []
    for bond_order in bond_orders:
        if bond_order is None:
            result.append(None)
        else:
            if bond_order < 1.11:
                result.append(1.0)
            elif bond_order < 1.54:
                result.append(1.5)
            elif bond_order < 2.0:
                result.append(2.0)
            else:
                result.append(3.0)
    
    return result

def extract_npa_charges(charge_tuples: list) -> tuple:
    """
    Vectorized extraction of charge components from NPA charge tuples.
    
    Args:
        charge_tuples (list): List of (electron_pop, nmb_pop, npa_charge) tuples
        
    Returns:
        tuple: (electron_populations, nmb_populations, npa_charges) as lists
    """
    if not charge_tuples:
        return [], [], []
        
    charge_array = np.array(charge_tuples)
    return (charge_array[:, 0].tolist(), 
            charge_array[:, 1].tolist(), 
            charge_array[:, 2].tolist())

def get_wiberg_bond_order_totals(wiberg_matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Vectorized helper function that returns the bond order totals (row sums) from a Wiberg bond order matrix using NumPy sum.
    
    Args:
        wiberg_matrix (list[list[float]]): A 2-dimensional list corresponding to the Wiberg bond order matrix.
        
    Returns:
        np.ndarray: Bond order totals (row sums) for each atom
    """
    if wiberg_matrix is None or len(wiberg_matrix) == 0:
        return np.array([])
    
    matrix = np.array(wiberg_matrix)
    return np.sum(matrix, axis=1)

def get_bound_hydrogens_per_atom(
    atomic_numbers: Union[List[int], np.ndarray], 
    wiberg_matrix: Union[List[List[float]], np.ndarray],
    bond_threshold: float = 0.4
) -> np.ndarray:
    """
    Helper function that returns the number of hydrogen neighbors, given an atom, vectorized.
    
    Args:
        atomic_numbers (list[int]): List of atomic numbers within the molecule.
        wiberg_matrix (list[list[float]]): Wiberg bond order matrix.
        bond_threshold (float): Minimum bond order threshold (default 0.4)

    Returns:
        np.ndarray: Number of hydrogen neighbors for each atom
    """
    if not atomic_numbers or not wiberg_matrix:
        return np.array([])
    
    atomic_nums = np.array(atomic_numbers)
    matrix = np.array(wiberg_matrix)
    
    # Create mask for hydrogen atoms (atomic number 1)
    hydrogen_mask = atomic_nums == 1
    
    # Create mask for bonds above threshold
    bond_mask = matrix > bond_threshold
    
    # For each atom, count hydrogens it's bonded to
    # Multiply bond_mask with hydrogen_mask (broadcasted)
    hydrogen_bonds = bond_mask * hydrogen_mask[np.newaxis, :]
    
    # Sum along columns (count hydrogen neighbors for each atom)
    # Subtract diagonal to exclude self-bonds
    np.fill_diagonal(hydrogen_bonds, 0)
    bound_h_counts = np.sum(hydrogen_bonds, axis=1)
    
    return bound_h_counts.astype(int)

def get_node_degrees(
    wiberg_matrix: Union[List[List[float]], np.ndarray], 
    threshold: float = 0.4
) -> np.ndarray:
    """
    Helper function that computes the node degree (number of bonded atoms) for each atom, vectorized.
    
    Args:
        wiberg_matrix (list[list[float]]): 2D Wiberg bond order matrix
        threshold (float): Minimum bond order threshold (default 0.4)

    Returns:
        np.ndarray: Node degree for each atom
    """
    if not wiberg_matrix:
        return np.array([])
    
    matrix = np.array(wiberg_matrix)
    
    # Create boolean mask for bonds above threshold
    bond_mask = matrix > threshold
    
    # Set diagonal to False to exclude self-bonds
    np.fill_diagonal(bond_mask, False)
    
    # Count True values in each row
    degrees = np.sum(bond_mask, axis=1)
    
    return degrees.astype(int)

def get_lowest_vibrational_frequency(
    frequencies: Union[List[float], np.ndarray]
) -> Optional[float]:
    """
    Returns the lowest vibrational frequency from a list of vibrational frequencies.

    Args:
        frequencies (list): list of vibrational frequencies
        
    Returns:
        min_freq (float): lowest vibrational frequency, or None if not available
    """
    if not frequencies:
        return None
    
    freq_array = np.array(frequencies)
    
    if len(freq_array) == 0:
        return None
    
    return float(np.min(freq_array))

def get_highest_vibrational_frequency(
    frequencies: Union[List[float], np.ndarray]
) -> Optional[float]:
    """
    Returns the highest vibrational frequency from a list of vibrational frequencies.

    Args:
        frequencies (list): list of vibrational frequencies
        
    Returns:
        max_freq (float): highest vibrational frequency, or None if not available
    """
    if not frequencies:
        return None

    freq_array = np.array(frequencies)

    if len(freq_array) == 0:
        return None

    return float(np.max(freq_array))


def get_atom_indices(atom_labels):
    """
    Helper function that extracts the integer indices from a list of atom labels (e.g., ['C1', 'C2', ..., 'H11']).

    Args:
        atom_labels (list[str]): List of atom labels, each consisting of an element symbol and an index (e.g., 'C1', 'H11').

    Returns:
        list[int]: List of atom indices extracted from the labels (e.g., [1, 2, ..., 11]).
    """
    return [int(''.join(filter(str.isdigit, label))) for label in atom_labels]

def get_clpo_by_type(
    clpo_data: List[List], 
    target_types: Union[str, List[str]]
) -> List[List]:
    """
    Vectorized filtering of CLPO data by type(s).
    
    Args:
        clpo_data: List of CLPO entries [id, type, indices, occupancy]
        target_types: Single type or list of types to filter for
        
    Returns:
        List of filtered CLPO entries
    """
    if not clpo_data:
        return []
    
    if isinstance(target_types, str):
        target_types = [target_types]
    
    # Extract types and convert to numpy array for vectorized operations
    types = np.array([entry[1] for entry in clpo_data])
    
    # Create boolean mask for target types
    mask = np.isin(types, target_types)
    
    # Filter using boolean indexing (convert back to list)
    return [clpo_data[i] for i in range(len(clpo_data)) if mask[i]]

def get_LP_CLPOs(clpo_data: List[List]) -> List[List]:
    """
    Extracts all lone pair CLPOs from the provided CLPO data.
    
    Args:
        CLPO_data: list(list[str,list[int],float,list[float]]), containing:
            - CLPO_id: int, unique identifier
            - CLPO_type: str, the type of CLPO (BD, NB, LP)
            - atom_indices: list[int], a list containing the atom indices of the atoms involved in the CLPO
            - occupancy: float, the occupancy of the CLPO
        
    Returns:
        list(list[str,list[int],float,list[float]])
        A list of lists containing all lone pair CLPOs (CLPO_type == 'LP'), if none are found, returns an empty list.
    """
    return get_clpo_by_type(clpo_data, 'LP')

def get_BD_CLPOs(clpo_data: List[List]) -> List[List]:
    """
    Extracts all bond pair CLPOs from the provided CLPO data.

    Args:
        CLPO_data: list(list[str,list[int],float,list[float]]), containing:
            - CLPO_id: int, unique identifier
            - CLPO_type: str, the type of CLPO (BD, NB, LP)
            - atom_indices: list[int], a list containing the atom indices of the atoms involved in the CLPO
            - occupancy: float, the occupancy of the CLPO

    Returns:
        list(list[str,list[int],float,list[float]])
        A list of lists containing all bond pair CLPOs (CLPO_type == 'BD'), if none are found, returns an empty list.
    """
    return get_clpo_by_type(clpo_data, 'BD')

def get_AB_CLPOs(clpo_data: List[List]) -> List[List]:
    """
    Extracts all atom pair CLPOs from the provided CLPO data.

    Args:
        CLPO_data: list(list[str,list[int],float,list[float]]), containing:
            - CLPO_id: int, unique identifier
            - CLPO_type: str, the type of CLPO (BD, NB, LP)
            - atom_indices: list[int], a list containing the atom indices of the atoms involved in the CLPO
            - occupancy: float, the occupancy of the CLPO

    Returns:
        list(list[str,list[int],float,list[float]])
        A list of lists containing all atom pair CLPOs (CLPO_type == 'NB'), if none are found, returns an empty list.
    """
    return get_clpo_by_type(clpo_data, 'NB')

def get_wiberg_bond_orders_batch(
    wiberg_matrix: Union[List[List[float]], np.ndarray],
    bond_indices_list: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Vectorized batch extraction of Wiberg bond orders for multiple bonds.
    
    Args:
        wiberg_matrix (list[list[float]]): A 2-dimensional list corresponding to the Wiberg bond order matrix.
        bond_indices (list[tuple[int, int]]): A list of 2-tuples corresponding to the bond indices of interest.

    Returns:
        np.ndarray: Array of bond orders for each bond pair
    """
    if not wiberg_matrix or not bond_indices_list:
        return np.array([])
    
    matrix = np.array(wiberg_matrix)
    
    # Extract i and j indices
    i_indices = np.array([pair[0] for pair in bond_indices_list])
    j_indices = np.array([pair[1] for pair in bond_indices_list])
    
    # Get bond orders using advanced indexing
    bond_orders_ij = matrix[i_indices, j_indices]
    bond_orders_ji = matrix[j_indices, i_indices]
    
    # Average the symmetric entries
    return (bond_orders_ij + bond_orders_ji) / 2.0

def atomic_property_lookup(
    atomic_numbers: Union[List[int], np.ndarray],
    property_dict: dict
) -> np.ndarray:
    """
    Helper function to map atomic numbers to specified property values, vectorized.
    
    Args:
        atomic_numbers (list): List of atomic numbers
        property_dict (dict): Dictionary mapping atomic numbers to property values
        
    Returns:
        np.ndarray: Array of property values (NaN for missing keys)
    """
    if not atomic_numbers:
        return np.array([])
    
    atomic_nums = np.array(atomic_numbers)
    
    # Create array for results
    result = np.full(len(atomic_nums), np.nan)
    
    # Vectorized lookup using boolean indexing
    for atomic_num, prop_value in property_dict.items():
        mask = atomic_nums == atomic_num
        result[mask] = prop_value
    
    return result

def get_formula(smiles_string):
    """
    Calculates the stoichiometric formula of a molecule given its SMILES string.
    
    Args: 
        smiles_string (str): The SMILES string of the molecule

    Returns:
        str: The stoichiometric formula of the molecule.
    
    Raises:
        ValueError: If the SMILES string cannot be parsed into an RDKit molecule.
    """
    _, atoms = get_atomic_data_dicts()
    molecule = Chem.MolFromSmiles(smiles_string)
    molecule = Chem.AddHs(molecule)
    atom_list = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    unique_values, counts = np.unique(atom_list, return_counts=True)
    values = [6, 1] + [num.item() for num in unique_values if num not in [6, 1]]
    reordered_counts = [
        counts[np.where(unique_values == num)[0][0]].item()
        if num in unique_values else 0
        for num in values
    ]
    formula = ""
    for num, count in zip(values, reordered_counts):
        symbol = atoms[num]
        if count == 1:
            formula += symbol
        else:
            formula += f"{symbol}{count}"
    if not formula:
        raise ValueError("The SMILES string does not represent a valid molecule.")
    return formula

def generate_qm_data_dict(
    mol_id: str,
    smiles: str,
    xyz_file: Union[str, Path],
    shermo_output: Union[str, Path],
    janpa_output: Union[str, Path, None] = None,
    nbo_output: Union[str, Path, None] = None
):
    """
    Generate a dictionary of QM data from available files using memory mapping, pre-loading all files and batch-processing for efficiency. Accepts either or both JANPA and NBO files.
    Args:
        mol_id (str): Molecule identifier.
        smiles (str): SMILES string.
        xyz_file (str): Path to geometry (.xyz) file.
        shermo_output (str): Path to Shermo output file.
        janpa_output (str, optional): Path to JANPA output file. If None, JANPA fields will be missing.
        nbo_output (str, optional): Path to NBO output file. If None, NBO fields will be missing.
    Returns:
        dict: Dictionary containing all available QM data fields. Missing fields are set to None or empty.
    """
    # Convert paths to Path objects
    xyz_file = Path(xyz_file)
    shermo_output = Path(shermo_output)
    janpa_output = Path(janpa_output) if janpa_output else None
    nbo_output = Path(nbo_output) if nbo_output else None
    
    # Identify graph type
    if not janpa_output and not nbo_output:
        graph_type = "DFT"
    elif janpa_output and not nbo_output:
        graph_type = "NPA"
    elif not janpa_output and nbo_output:
        graph_type = "NBO"
    elif janpa_output and nbo_output:
        graph_type = "QM"
    else:
        graph_type = "DFT"
    
    # Parse xyz file
    atom_list, atomic_numbers, xyz_coordinates, num_atoms, atom_labels = read_xyz_file(xyz_file)
    
    formula = get_formula(smiles)
    molecular_mass = get_molecular_weight(smiles)
    
    data = {
        "graph_type": graph_type,
        "id": mol_id,
        "smiles": smiles,
        "formula": formula,
        "molecular_mass": molecular_mass,
        "xyz_coordinates": xyz_coordinates,
        "num_atoms": num_atoms,
        "atomic_numbers": atomic_numbers,
        "atom_labels": atom_labels,
    }
    
    shermo_data = get_all_shermo_data(shermo_output)
    data.update(shermo_data)
    
    if janpa_output and janpa_output.exists():
        janpa_data = get_all_janpa_data(janpa_output)
        data.update(janpa_data)
    
    if nbo_output and nbo_output.exists():
        nbo_data = get_all_nbo_data(nbo_output)
        data.update(nbo_data)
    
    # Clear cache to free memory
    clear_file_cache()
    
    return data

# Convenience function for clearing cache manually
def clear_mmap_cache():
    """Clear the memory-mapped file cache to free memory."""
    clear_file_cache()
    
