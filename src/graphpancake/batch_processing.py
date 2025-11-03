#!/usr/bin/env python3
"""
    1. Copy config_template.yaml to config.yaml
    2. Edit config.yaml with your data paths and settings
    3. Prepare a CSV file with molecular identifiers, SMILES, and labels  
    4. Run: python batch_processing_script.py
"""
import sys
import os
import logging
import gzip
import tarfile
import zipfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List
import pandas as pd
import json
import yaml
from tqdm import tqdm
import gc
import psutil
import time
import multiprocessing as mp
import argparse

from graphpancake.graph import MolecularGraph
from graphpancake.classes import DictData
from graphpancake.functions import generate_qm_data_dict, clear_file_cache, clear_mmap_cache

def process_single_molecule_multiprocessing(args):
    """
    Multiprocessing worker function to process a single molecule.
    
    This function must be defined at module level for multiprocessing.
    It takes a tuple of (mol_dict, config) and returns processed graph data.
    """
    mol_dict, config = args
    mol_id = mol_dict['mol_id']
    
    try:
        smiles_column = get_config_value(config, 'labels_config.smiles_column')
        smiles = mol_dict['labels'].get(smiles_column, '') if smiles_column else ''
        
        qm_dict = generate_qm_data_dict(
            mol_id=mol_id,
            smiles=smiles,
            xyz_file=mol_dict['xyz'],
            shermo_output=mol_dict['shermo'],
            janpa_output=mol_dict['janpa'],
            nbo_output=mol_dict['nbo']
        )
        
        if qm_dict is None:
            return {
                'error': True,
                'mol_id': mol_id,
                'error_message': 'generate_qm_data_dict returned None - likely parsing error'
            }
            
        required_fields = ['atomic_numbers', 'xyz_coordinates']
        missing_fields = [field for field in required_fields if qm_dict.get(field) is None]
        if missing_fields:
            return {
                'error': True,
                'mol_id': mol_id,
                'error_message': f'Missing required fields: {missing_fields}'
            }
        
        for field_name, field_value in qm_dict.items():
            if field_value is None:
                continue
        
        # Add graph type and metadata
        qm_dict['graph_type'] = mol_dict['graph_type']
        
        # Create molecular graph and save directly to database
        dict_data = DictData(qm_dict)
        molecular_graph = MolecularGraph(dict_data)
        molecular_graph.graph_id = mol_id
        molecular_graph.graph_type = mol_dict['graph_type']
        
        # Set the graph ID in graph_info so the save_to_database method can access it via self.id
        molecular_graph.graph_info.id = mol_id
        
        # Store labels data for database saving
        molecular_graph._labels_data = mol_dict['labels']
        
        # Get database path from config using consistent function
        database_path = get_database_path(config)
        
        # Save full graph structure to database
        molecular_graph.save_to_database(database_path)
        
        if hasattr(molecular_graph.nodes, '__len__'):
            node_count = len(molecular_graph.nodes)
        elif hasattr(molecular_graph, 'node_count'):
            node_count = molecular_graph.node_count
        else:
            node_count = 0
            
        if hasattr(molecular_graph.edges, '__len__'):
            edge_count = len(molecular_graph.edges)
        elif hasattr(molecular_graph, 'edge_count'):
            edge_count = molecular_graph.edge_count
        else:
            edge_count = 0
        
        return {
            'id': molecular_graph.graph_id,
            'graph_type': molecular_graph.graph_type,
            'smiles': getattr(molecular_graph._qm_data, 'smiles', ''),
            'nodes': node_count,
            'edges': edge_count,
            'node_features': len(molecular_graph.get_node_features()) if molecular_graph.get_node_features() else 0,
            'edge_features': len(molecular_graph.get_edge_features()) if molecular_graph.get_edge_features() else 0,
            'saved_to_db': True
        }
        
    except Exception as e:
        return {
            'error': True,
            'mol_id': mol_id,
            'error_message': str(e)
        }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_file: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_file).resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please copy config_template.yaml to {config_file} and customize it for your data."
        )
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['data_paths', 'file_patterns', 'settings']
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            raise ValueError(f"Missing required sections in config file: {missing_sections}")
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config.yaml file: {e}")

def get_config_value(config: Dict, path: str, default=None):
    """
    Get nested config value using dot notation (e.g., 'data_paths.base_dir').
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path to config value
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def get_database_path(config: Dict) -> str:
    """
    Get the database path with consistent .db extension handling.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Database path with .db extension
    """
    db_name = get_config_value(config, 'output.database_name', 'molecular_graphs')
    if not db_name.endswith('.db'):
        db_name += '.db'
    return db_name

class MemoryMonitor:
    """Monitor and manage memory usage during batch processing."""
    
    __slots__ = ['start_memory', 'peak_memory', 'gc_threshold', 'cache_clear_threshold', 'cpu_count']
    
    def __init__(self, gc_threshold=85, cache_clear_threshold=80):
        """
        Args:
            gc_threshold: Memory percentage to trigger garbage collection
            cache_clear_threshold: Memory percentage to clear file caches
        """
        self.start_memory = psutil.virtual_memory().percent
        self.peak_memory = self.start_memory
        self.gc_threshold = gc_threshold
        self.cache_clear_threshold = cache_clear_threshold
        self.cpu_count = psutil.cpu_count(logical=True)
    
    def check_memory(self, force_gc=False):
        """Check memory usage and perform cleanup if needed."""
        current_memory = psutil.virtual_memory().percent
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if current_memory > self.cache_clear_threshold or force_gc:
            clear_file_cache()
            clear_mmap_cache()
            
        if current_memory > self.gc_threshold or force_gc:
            gc.collect()
            
        return current_memory
    
    def get_memory_stats(self):
        """Get memory usage statistics."""
        current = psutil.virtual_memory().percent
        return {
            'start': self.start_memory,
            'current': current,
            'peak': self.peak_memory,
            'increase': current - self.start_memory
        }

class MoleculeData:
    """Memory-optimized container for molecule data using __slots__."""
    
    __slots__ = ['mol_id', 'xyz', 'shermo', 'janpa', 'nbo', 'labels', 'graph_type', '_file_cache']
    
    def __init__(self, mol_id: str, labels: dict = None):
        self.mol_id = mol_id
        self.xyz = None
        self.shermo = None
        self.janpa = None
        self.nbo = None
        self.labels = labels or {}
        self.graph_type = None
        self._file_cache = {}  # Cache for pre-loaded file contents
    
    def to_dict(self):
        """Convert to dictionary for compatibility with existing code."""
        return {
            'mol_id': self.mol_id,
            'xyz': self.xyz,
            'shermo': self.shermo,
            'janpa': self.janpa,
            'nbo': self.nbo,
            'labels': self.labels,
            'graph_type': self.graph_type
        }

class BatchProcessor:
    """Handles batch processing of molecular data files with memory optimizations."""
    
    __slots__ = ['config', 'failed_molecules', 'successful_count', 'temp_extract_dir', 
                 '_molecule_cache', '_file_cache', '_memory_monitor']
    
    def __init__(self, config: Dict):
        self.config = config
        self.failed_molecules = []
        self.successful_count = 0
        self.temp_extract_dir = None  # Track temporary extraction directory
        self._molecule_cache = {}  # Cache for reusable molecule data
        self._file_cache = {}  # Cache for discovered files
        self._memory_monitor = MemoryMonitor()
        
    def run(self):
        """Main batch processing pipeline with memory optimization and detailed timing."""
        start_time = time.time()
        self._memory_monitor.check_memory()  # Initialize memory monitoring
        
        logger.info("Starting graphpancake batch processing with optimizations...")
        logger.info(f"Configuration: {self._get_config_summary()}")
        logger.info(f"Initial memory usage: {self._memory_monitor.start_memory:.1f}%")
        
        steps = [
            ("Creating database", self._create_database),
            ("Loading labels/metadata", self._load_labels),
            ("Discovering/extracting data files", self._get_data_files),
            ("Matching files to molecules", None), 
            ("Processing molecules", None), 
            ("Generating report", self._generate_report),
            ("Cleaning up", self._cleanup_temp_files)
        ]
        
        try:
            step_progress = tqdm(steps, desc="Pipeline progress", unit="step")
            
            for step_name, step_func in step_progress:
                step_progress.set_description(f"Pipeline: {step_name}")
                step_start = time.time()
                memory_before = self._memory_monitor.check_memory()
                
                if step_name == "Loading labels/metadata":
                    labels_df = step_func()
                elif step_name == "Discovering/extracting data files":
                    data_files = step_func()
                elif step_name == "Matching files to molecules":
                    matched_molecules = self._match_files_to_molecules(data_files, labels_df)
                    # Clear labels_df to free memory
                    del labels_df
                    gc.collect()
                elif step_name == "Processing molecules":
                    self._process_molecules(matched_molecules)
                    # Clear matched molecules to free memory
                    del matched_molecules
                    gc.collect()
                elif step_func:
                    step_func()
                
                step_time = time.time() - step_start
                memory_after = self._memory_monitor.check_memory()
                memory_delta = memory_after - memory_before
                
                logger.info(f"Completed: {step_name} ({step_time:.2f}s, Mem: {memory_after:.1f}% [{memory_delta:+.1f}%])")
            
            step_progress.close()
            
            total_time = time.time() - start_time
            final_memory_stats = self._memory_monitor.get_memory_stats()
            
            logger.info(f"Batch processing completed successfully in {total_time:.2f}s")
            logger.info(f"Memory usage summary:")
            logger.info(f"  Start: {final_memory_stats['start']:.1f}%")
            logger.info(f"  Peak:  {final_memory_stats['peak']:.1f}%")
            logger.info(f"  Final: {final_memory_stats['current']:.1f}%")
            logger.info(f"  Total increase: {final_memory_stats['increase']:+.1f}%")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            logger.error(f"Memory at failure: {self._memory_monitor.check_memory():.1f}%")
            raise
    
    def _get_data_files(self):
        """Get data files from archive or directories."""
        if get_config_value(self.config, 'data_paths.archive_file'):
            return self._extract_and_discover_archive()
        else:
            return self._discover_files_from_directories()
    
    def _get_config_summary(self) -> str:
        """Get configuration summary for logging."""
        summary = []
        summary.append(f"Archive mode: {bool(get_config_value(self.config, 'data_paths.archive_file'))}")
        summary.append(f"Max workers: {get_config_value(self.config, 'settings.max_workers', 32)}")
        summary.append(f"Batch size: {get_config_value(self.config, 'settings.batch_size', 100)}")
        summary.append(f"Output: {get_config_value(self.config, 'output.database_name', 'molecular_graphs.db')}")
        return ", ".join(summary)
    
    def _create_database(self):
        """Create output database with appropriate schema."""
        output_path = Path(get_database_path(self.config))

        # Always recreate the database to ensure correct schema
        if output_path.exists():
            logger.info(f"Removing existing database: {output_path}")
            output_path.unlink()

        logger.info(f"Creating new database: {output_path}")
        MolecularGraph.create_database(output_path)
        logger.info("Database created successfully")
    
    def _load_labels(self) -> pd.DataFrame:
        """Load molecular labels and metadata from CSV."""
        data_file = get_config_value(self.config, 'data_file')
        labels_file = get_config_value(self.config, 'labels_file') or data_file
        
        if not labels_file:
            raise ValueError("No data_file or labels_file specified in configuration")
        
        labels_path = Path(labels_file)
        
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        logger.info(f"Loading labels from: {labels_path}")
        labels_df = pd.read_csv(labels_path)
        
        id_column = get_config_value(self.config, 'labels_config.id_column', 'mol_id')
        required_cols = [id_column]
        
        smiles_column = get_config_value(self.config, 'labels_config.smiles_column')
        if smiles_column:
            required_cols.append(smiles_column)
        
        missing_cols = [col for col in required_cols if col not in labels_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in labels file: {missing_cols}")
        
        logger.info(f"Loaded {len(labels_df)} labels")
        logger.info(f"Available columns: {list(labels_df.columns)}")
        
        return labels_df
    
    def _extract_and_discover_archive(self) -> Dict[str, List[Path]]:
        """Extract archive and discover files."""
        archive_path = Path(get_config_value(self.config, 'data_paths.archive_file'))
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        
        # Create temporary extraction directory
        extract_dir = Path("temp_extracted_data")
        extract_dir.mkdir(exist_ok=True)
        self.temp_extract_dir = extract_dir  # Store for cleanup later
        
        logger.info(f"Extracting archive: {archive_path}")
        
        # Extract based on file type with progress tracking
        if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting tar.gz files", unit="file"):
                    tar.extract(member, extract_dir, filter='data')
        elif archive_path.suffix in ['.tar', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting tar files", unit="file"):
                    tar.extract(member, extract_dir, filter='data') 
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                for file_name in tqdm(file_list, desc="Extracting zip files", unit="file"):
                    zip_file.extract(file_name, extract_dir)
        elif archive_path.suffix == '.gz':
            # Single gzip file
            output_path = extract_dir / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    # Read in chunks with progress
                    total_size = archive_path.stat().st_size
                    with tqdm(total=total_size, desc="Extracting gzip file", unit="B", unit_scale=True) as pbar:
                        while True:
                            chunk = f_in.read(8192)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            pbar.update(len(chunk))
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        logger.info(f"Archive extracted to: {extract_dir}")
        
        # Discover files in extracted directory
        return self._discover_files_in_directory(extract_dir)
    
    def _discover_files_from_directories(self) -> Dict[str, List[Path]]:
        """Discover files from specified directories."""
        base_dir = get_config_value(self.config, 'data_paths.base_dir')
        
        if not base_dir:
            raise ValueError("No base_dir specified in data_paths configuration")
        
        base_dir = Path(base_dir)
        
        if not base_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {base_dir}")
        
        return self._discover_files_in_directory(base_dir)
    
    def _discover_files_in_directory(self, base_dir: Path) -> Dict[str, List[Path]]:
        """Discover QM output files in directory structure using config file patterns with caching and parallel processing."""
        
        # Check cache first
        cache_key = str(base_dir)
        if cache_key in self._file_cache:
            logger.info("Using cached file discovery results")
            return self._file_cache[cache_key]
        
        # Get file patterns from config
        config_patterns = get_config_value(self.config, 'file_patterns', {})
        
        if not config_patterns:
            raise ValueError("No file_patterns specified in configuration")
        
        # Build search patterns based on config patterns
        file_patterns = {}
        
        # Map of file types to their pattern keys in config
        pattern_map = {
            'xyz': 'xyz_pattern',
            'shermo': 'shermo_pattern', 
            'janpa': 'janpa_pattern',
            'nbo': 'nbo_pattern'
        }
        
        for file_type, pattern_key in pattern_map.items():
            if pattern_key in config_patterns:
                pattern = config_patterns[pattern_key]
                if '{mol_id}' in pattern:
                    # Convert pattern to glob pattern
                    glob_pattern = pattern.replace('{mol_id}', '*')
                    file_patterns[file_type] = [glob_pattern]
                else:
                    # Use pattern as-is if no {mol_id} placeholder
                    file_patterns[file_type] = [pattern]
            else:
                logger.warning(f"No {pattern_key} specified in config, skipping {file_type} files")
                file_patterns[file_type] = []

        discovered_files = {key: [] for key in file_patterns}
        
        # Use parallel processing for file discovery if there are many patterns
        logger.info("Discovering files with parallel processing...")
        
        def discover_file_type(file_type_patterns):
            """Helper function for parallel file discovery."""
            file_type, patterns = file_type_patterns
            found_files = []
            for pattern in patterns:
                if pattern:  # Skip empty patterns
                    pattern_files = list(base_dir.rglob(pattern))
                    found_files.extend(pattern_files)
            return file_type, found_files
        
        # Use parallel processing for file discovery
        max_workers = min(4, len(file_patterns))  # Limit workers for I/O operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(discover_file_type, item): item[0] 
                      for item in file_patterns.items()}
            
            for future in tqdm(as_completed(futures), 
                             desc="Discovering file types", 
                             total=len(futures)):
                file_type, found_files = future.result()
                discovered_files[file_type] = found_files
        
        # Remove duplicates using sets
        for file_type in discovered_files:
            discovered_files[file_type] = list(set(discovered_files[file_type]))
        
        # Cache results
        self._file_cache[cache_key] = discovered_files
        
        # Log discovery results
        for file_type, files in discovered_files.items():
            logger.info(f"Found {len(files)} {file_type} files")
        
        return discovered_files
    
    def _match_files_to_molecules(self, discovered_files: Dict, labels_df: pd.DataFrame) -> List[MoleculeData]:
        """Match discovered files to molecules based on IDs with optimized data structures."""
        id_column = get_config_value(self.config, 'labels_config.id_column', 'mol_id')
        mol_ids = labels_df[id_column].astype(str).tolist()
        
        # Create file lookup dictionaries for faster matching
        file_lookups = {}
        for file_type, files in discovered_files.items():
            file_lookups[file_type] = {}
            for file_path in files:
                # Extract potential molecule IDs from filename
                stem = file_path.stem
                file_lookups[file_type][stem] = file_path
                # Also index by full name without extension
                name_parts = stem.split('_')
                for part in name_parts:
                    if part:
                        file_lookups[file_type][part] = file_path
        
        matched_molecules = []
        file_patterns = get_config_value(self.config, 'file_patterns', {})
        
        logger.info(f"Matching files for {len(mol_ids)} molecules...")
        
        # Batch process labels for faster lookup
        labels_dict = {}
        
        # Get configuration for label processing
        smiles_column = get_config_value(self.config, 'labels_config.smiles_column')
        
        # Get the configured label columns (only extract these, not all CSV columns)
        label_columns = []
        labels_column = get_config_value(self.config, 'labels_config.labels_column')
        if labels_column and labels_column in labels_df.columns:
            label_columns.append(labels_column)
        
        # Always include SMILES if configured
        if smiles_column and smiles_column in labels_df.columns:
            label_columns.append(smiles_column)
            
        logger.info(f"Processing label columns: {label_columns}")
        
        for _, row in labels_df.iterrows():
            mol_id = str(row[id_column])
            # Only extract specified label columns, not all CSV columns
            labels_data = {}
            for col in label_columns:
                if col in row and pd.notna(row[col]):  # Skip NaN values
                    labels_data[col] = row[col]
            labels_dict[mol_id] = labels_data
        
        processed_count = 0
        total_mols = len(mol_ids)
        
        for mol_id in mol_ids:
            mol_data = MoleculeData(mol_id, labels_dict.get(mol_id, {}))
            
            # Lookup dictionaries for file matching
            for file_type in ['xyz', 'shermo', 'janpa', 'nbo']:
                pattern_key = f"{file_type}_pattern"
                file_path = None
                
                if pattern_key in file_patterns:
                    pattern = file_patterns[pattern_key]
                    expected_filename = pattern.format(mol_id=mol_id)
                    
                    # Try exact match first
                    expected_stem = Path(expected_filename).stem
                    if expected_stem in file_lookups[file_type]:
                        file_path = file_lookups[file_type][expected_stem]
                    # Try molecule ID match
                    elif mol_id in file_lookups[file_type]:
                        file_path = file_lookups[file_type][mol_id]
                    else:
                        # Fallback to substring search (slower)
                        for file_stem, path in file_lookups[file_type].items():
                            if mol_id in file_stem:
                                file_path = path
                                break
                
                setattr(mol_data, file_type, file_path)
            
            # Check if minimum file requirements are met
            if self._validate_molecule_files(mol_data):
                mol_data.graph_type = self._determine_graph_type_optimized(mol_data)
                matched_molecules.append(mol_data)
            else:
                logger.warning(f"Skipping molecule {mol_id}: missing required files")
            
            # Log progress every 10%
            processed_count += 1
            if processed_count % max(1, total_mols // 10) == 0:
                logger.info(f"Matched files: {processed_count}/{total_mols} ({100*processed_count/total_mols:.1f}%)")
        
        logger.info(f"Matched files for {len(matched_molecules)} molecules")
        return matched_molecules
    
    def _validate_molecule_files(self, mol_data: MoleculeData) -> bool:
        """Validate that molecule has required files."""
        if mol_data.xyz is None and mol_data.shermo is None:
            return False
        return True
    
    def _determine_graph_type_optimized(self, mol_data: MoleculeData) -> str:
        """Determine the appropriate graph type based on available files."""
        has_janpa = mol_data.janpa is not None
        has_nbo = mol_data.nbo is not None
        
        if has_janpa and has_nbo:
            return 'QM'
        elif has_nbo:
            return 'NBO'
        elif has_janpa:
            return 'NPA'
        else:
            return 'DFT'
    
    def _process_molecules(self, matched_molecules: List[MoleculeData]):
        """Process molecules in chunks with parallel workers and memory optimization."""
        total_molecules = len(matched_molecules)
        
        # Optimize batch size for CPU cores (32 threads = 16 cores)
        cpu_cores = self._memory_monitor.cpu_count or 16  # Fallback to 16 if unknown
        # Use batch size that's a multiple of CPU cores
        default_batch_size = max(cpu_cores * 4, 128)  # At least 4x cores, minimum 128
        batch_size = get_config_value(self.config, 'settings.batch_size', default_batch_size)
        
        logger.info(f"Processing {total_molecules} molecules in chunks of {batch_size} (CPU cores: {cpu_cores})")
        
        # Progress bar
        total_chunks = (total_molecules + batch_size - 1) // batch_size
        chunk_progress = tqdm(range(0, total_molecules, batch_size), 
                             desc="Processing chunks", 
                             unit="chunk",
                             total=total_chunks)
        
        for i in chunk_progress:
            chunk = matched_molecules[i:i + batch_size]
            chunk_num = i // batch_size + 1
            
            memory_stats = self._memory_monitor.get_memory_stats()
            chunk_progress.set_description(f"Chunk {chunk_num}/{total_chunks} (Mem: {memory_stats['current']:.1f}%)")
            
            self._process_chunk_optimized(chunk)
            
            if chunk_num % 2 == 0:
                self._memory_monitor.check_memory(force_gc=True)
                
            clear_cache_frequency = get_config_value(self.config, 'memory.clear_cache_frequency', 50)
            if chunk_num % max(1, clear_cache_frequency // batch_size) == 0:
                clear_file_cache()
                clear_mmap_cache()
                logger.info("Cleared file and memory caches")
        
        chunk_progress.close()
        final_memory = self._memory_monitor.get_memory_stats()
        logger.info(f"Processing complete: {self.successful_count} successful, {len(self.failed_molecules)} failed")
        logger.info(f"Memory usage - Start: {final_memory['start']:.1f}%, Peak: {final_memory['peak']:.1f}%, Final: {final_memory['current']:.1f}%")
    
    def _process_chunk_optimized(self, molecules_chunk: List[MoleculeData]):
        """Process molecules using multiprocessing to bypass Python GIL for true parallelism."""
        cpu_cores = self._memory_monitor.cpu_count or 16
        
        default_workers = min(cpu_cores // 2, 32) 
        max_workers = get_config_value(self.config, 'settings.max_workers', default_workers)
        
        # Adaptive worker count based on memory usage
        current_memory = self._memory_monitor.check_memory()
        if current_memory > 80:
            max_workers = max(1, max_workers // 2)
        elif current_memory > 90:
            max_workers = max(1, max_workers // 4)
        
        logger.info(f"Processing {len(molecules_chunk)} molecules with {max_workers} processes (Memory: {current_memory:.1f}%)")
        
        worker_args = []
        for mol_data in molecules_chunk:
            mol_dict = {
                'mol_id': mol_data.mol_id,
                'xyz': str(mol_data.xyz) if mol_data.xyz else None,
                'shermo': str(mol_data.shermo) if mol_data.shermo else None,
                'janpa': str(mol_data.janpa) if mol_data.janpa else None,
                'nbo': str(mol_data.nbo) if mol_data.nbo else None,
                'labels': mol_data.labels,
                'graph_type': mol_data.graph_type
            }
            worker_args.append((mol_dict, self.config))
        
        if max_workers == 1:
            # Single-process fallback
            processed_results = []
            for args in worker_args:
                result = process_single_molecule_multiprocessing(args)
                if result and not result.get('error'):
                    processed_results.append(result)
            
            # Save to database
            if processed_results:
                self._save_processed_results_to_database(processed_results)
        else:
            logger.info(f"Multiprocessing with {max_workers} processes")
            
            start_time = time.time()
            processed_results = []
            completed = 0
            errors = 0
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = [executor.submit(process_single_molecule_multiprocessing, args) 
                          for args in worker_args]
                
                logger.info(f"Submitted {len(futures)} tasks to {max_workers} processes")
                
                # Collect results with timeout
                timeout_seconds = 120  # 2 minute timeout
                try:
                    for future in as_completed(futures, timeout=timeout_seconds):
                        try:
                            result = future.result(timeout=120)
                            if result:
                                # Debug logging
                                if result.get('error'):
                                    logger.debug(f"Got error result: {result['mol_id']} - {result['error_message']}")
                                else:
                                    logger.debug(f"Got successful result: {result.get('id', 'unknown')}")
                                processed_results.append(result)
                            else:
                                logger.debug("Got None result from worker")
                            completed += 1
                            
                            # Progress logging every 2%
                            if completed % max(1, len(futures) // 50) == 0:
                                elapsed = time.time() - start_time
                                rate = completed / elapsed if elapsed > 0 else 0
                                logger.info(f"Progress: {completed}/{len(futures)} ({100*completed/len(futures):.1f}%) - {rate:.1f} mol/s - {len(processed_results)} successful")
                                
                        except Exception as e:
                            errors += 1
                            logger.warning(f"Processing error #{errors}: {e}")
                            completed += 1
                            
                except Exception as timeout_error:
                    logger.error(f"Timeout after {timeout_seconds}s: {timeout_error}")
                    logger.error(f"Progress before timeout: {completed}/{len(futures)} completed, {len(processed_results)} successful")
            
            processing_time = time.time() - start_time
            logger.info(f"Molecular graph processing {len(processed_results)} successful, {errors} errors in {processing_time:.2f}s")

            # Batch database write
            if processed_results:
                logger.info(f"Now writing {len(processed_results)} molecules to database")
                self._save_processed_results_to_database(processed_results)
            else:
                logger.warning("All molecular graph generation efforts were unsuccessful, nothing to save to database")
    
    def _save_processed_results_to_database(self, processed_results: List[Dict]):
        """Process results from multiprocessing - database saves already completed in worker functions."""
        if not processed_results:
            return
            
        try:
            start_time = time.time()
            
            successful_count = 0
            for result in processed_results:
                if result.get('error'):
                    error_info = {
                        'mol_id': result['mol_id'],
                        'error': result['error_message'],
                        'available_files': {}
                    }
                    self.failed_molecules.append(error_info)
                    logger.debug(f"Skipping error result for {result['mol_id']}: {result['error_message']}")
                    continue
                
                if result.get('saved_to_db'):
                    successful_count += 1
                    logger.debug(f"Confirmed database save for {result.get('id', 'unknown')}")
            
            self.successful_count += successful_count
            
            duration = time.time() - start_time
            if successful_count > 0:
                logger.info(f"Saved {successful_count} molecules to database in {duration:.3f}s")
            else:
                logger.warning("No valid data to save to database")
                    
        except Exception as e:
            logger.error(f"Processing results failed: {e}")
    
    def _process_single_molecule_return_graph(self, mol_data: MoleculeData):
        """Process a single molecule and return the graph object (don't save to database yet)."""
        mol_id = mol_data.mol_id
        
        try:
            smiles_column = get_config_value(self.config, 'labels_config.smiles_column')
            smiles = mol_data.labels.get(smiles_column, '') if smiles_column else ''
            
            graph_type = mol_data.graph_type or self._determine_graph_type_optimized(mol_data)
            
            mol_dict = mol_data.to_dict()
            
            qm_dict = generate_qm_data_dict(
                mol_id=mol_id,
                smiles=smiles,
                xyz_file=mol_dict['xyz'],
                shermo_output=mol_dict['shermo'],
                janpa_output=mol_dict['janpa'],
                nbo_output=mol_dict['nbo']
            )
            
            if qm_dict is None:
                logger.warning(f"generate_qm_data_dict returned None for {mol_id} - likely parsing error")
                return None
            
            for field_name, field_value in qm_dict.items():
                if field_value is None:  
                    logger.warning(f"Field {field_name} is None for {mol_id}")
                    continue
            
            qm_dict['graph_type'] = graph_type
            
            dict_data = DictData(qm_dict)
            molecular_graph = MolecularGraph(dict_data)
            molecular_graph.graph_id = mol_id
            molecular_graph.graph_type = graph_type
            
            molecular_graph.dict_data = dict_data
            
            del qm_dict
            
            return molecular_graph
            
        except Exception as e:
            error_info = {
                'mol_id': mol_id,
                'error': str(e),
                'available_files': {
                    'xyz': mol_data.xyz is not None,
                    'shermo': mol_data.shermo is not None,
                    'janpa': mol_data.janpa is not None,
                    'nbo': mol_data.nbo is not None
                }
            }
            self.failed_molecules.append(error_info)
            
            continue_on_error = get_config_value(self.config, 'error_handling.continue_on_error', True)
            if continue_on_error:
                logger.warning(f"Failed to process molecule {mol_id}: {e}")
            else:
                logger.error(f"Failed to process molecule {mol_id}: {e}")
                raise
            
            return None
    
    def _cleanup_temp_files(self):
        """Clean up temporary extraction directory if it was created."""
        cleanup_temp = get_config_value(self.config, 'cleanup.remove_temp_files', True)
        
        if self.temp_extract_dir and self.temp_extract_dir.exists() and cleanup_temp:
            try:
                logger.info(f"Cleaning up temporary extraction directory: {self.temp_extract_dir}")
                shutil.rmtree(self.temp_extract_dir)
                logger.info("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
        elif self.temp_extract_dir and self.temp_extract_dir.exists() and not cleanup_temp:
            logger.info(f"Temporary extraction directory preserved: {self.temp_extract_dir}")
    
    def _generate_report(self):
        """Generate final processing report."""
        logger.info("-" * 60)
        logger.info("Molecular graph processing report")
        logger.info("-" * 60)
        logger.info(f"Successfully processed: {self.successful_count} molecules")
        logger.info(f"Failed: {len(self.failed_molecules)} molecules")
        
        database_name = get_config_value(self.config, 'output.database_name', 'molecular_graphs.db')
        logger.info(f"Database created: {database_name}")
        
        save_error_log = get_config_value(self.config, 'error_handling.save_error_log', True)
        if self.failed_molecules and save_error_log:
            error_log_path = f"{database_name}_failed_molecules.json"
            with open(error_log_path, 'w') as f:
                json.dump(self.failed_molecules, f, indent=2)
            logger.info(f"Error log saved: {error_log_path}")
        
        logger.info("-" * 60)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='GraphPancake batch processing script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m graphpancake.batch_processing_script
  python -m graphpancake.batch_processing_script --config /path/to/config.yaml
  python -m graphpancake.batch_processing_script --config ../config.yaml
        """
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration from YAML file
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
    except FileNotFoundError:
        logger.error(f"Configuration file '{args.config}' not found!")
        logger.info("Please copy config_template.yaml to config.yaml and customize it for your data.")
        return
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Basic path validation
    archive_file = get_config_value(config, 'data_paths.archive_file')
    if archive_file:
        archive_path = Path(archive_file)
        if not archive_path.exists():
            logger.error(f"Archive file does not exist: {archive_path}")
            logger.info("Please update the archive_file path in config.yaml")
            return
    else:
        base_dir = get_config_value(config, 'data_paths.base_dir')
        if not base_dir:
            logger.error("Neither archive_file nor base_dir specified in config.yaml")
            logger.info("Please specify data paths in config.yaml")
            return
        
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.error(f"Data directory does not exist: {base_path}")
            logger.info("Please update the base_dir path in config.yaml")
            return
    
    # Validate labels file
    data_file = get_config_value(config, 'data_file')
    labels_file = get_config_value(config, 'labels_file') or data_file
    
    if not labels_file:
        logger.error("No data_file or labels_file specified in config.yaml")
        logger.info("Please specify label file paths in config.yaml")
        return
    
    labels_path = Path(labels_file)
    if not labels_path.exists():
        logger.error(f"Labels file does not exist: {labels_path}")
        logger.info("Please update the labels_file path in config.yaml")
        return
    
    # Run batch processing
    processor = BatchProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
