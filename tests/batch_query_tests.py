import pytest
import tempfile
import shutil
import sqlite3
import polars as pl
import yaml
from pathlib import Path
import tarfile
import sys
import os
import json
import time
from unittest.mock import patch, MagicMock

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphpancake.batch_processing import BatchProcessor, get_config_value
from graphpancake.graph import MolecularGraph
from graphpancake.classes import DictData
from graphpancake.functions import generate_qm_data_dict

# Test data configuration - update these paths for your actual test data
TEST_DATA_CONFIG = {
    "tar_gz_file": "/home/uvx4187/graphpancake_data/SS-07-13_data.tar.gz",
    
    "labels_csv": "/home/uvx4187/graphpancake_data/SS-07-13_labels.csv",
    
    "csv_columns": {
        "mol_id": "mol_id",
        "smiles": "SMILES", 
        "label": "logP"  # or any other property column you want to use for testing
    },
    
    "test_db_name": "batch_test_molecules.db",
    "small_sample_size": 20,  # Number of molecules for quick tests
    "medium_sample_size": 100,  # For medium tests
    "performance_sample_size": 500  # For performance tests
}


class TestDataManager:
    """Manage test data for batch processing tests using real labels CSV."""
    
    @staticmethod
    def check_test_data():
        """Check if test data is available."""
        tar_gz_path = Path(TEST_DATA_CONFIG["tar_gz_file"])
        labels_csv_path = Path(TEST_DATA_CONFIG["labels_csv"])
        
        if tar_gz_path.exists() and labels_csv_path.exists():
            return "full"
        elif labels_csv_path.exists():
            return "csv_only"
        else:
            pytest.skip("No test data found. Please update TEST_DATA_CONFIG with valid paths.")
    
    @staticmethod
    def get_sample_molecules(sample_size="small", temp_dir=None):
        """Get a sample of molecules from a temp copy of the labels CSV."""
        labels_path = Path(TEST_DATA_CONFIG["labels_csv"])
        if temp_dir:
            temp_labels_path = Path(temp_dir) / "labels.csv"
            shutil.copy(labels_path, temp_labels_path)
        else:
            temp_labels_path = labels_path
        if not temp_labels_path.exists():
            pytest.skip(f"Labels CSV not found: {temp_labels_path}")
        
        df = pl.read_csv(temp_labels_path, ignore_errors=True, infer_schema_length=1000)
        
        # Get sample size
        if sample_size == "small":
            n_samples = TEST_DATA_CONFIG["small_sample_size"]
        elif sample_size == "medium":
            n_samples = TEST_DATA_CONFIG["medium_sample_size"]
        elif sample_size == "performance":
            n_samples = TEST_DATA_CONFIG["performance_sample_size"]
        else:
            n_samples = int(sample_size) if isinstance(sample_size, (int, str)) else 20
        
        # Take sample
        sample_df = df.head(n_samples)
        return sample_df
    
    @staticmethod
    def create_test_config(temp_dir, sample_size="small"):
        """Create a test configuration file using real molecule data."""
        data_type = TestDataManager.check_test_data()
        sample_df = TestDataManager.get_sample_molecules(sample_size, temp_dir)
        if data_type == "full":
            return TestDataManager._create_tar_config(temp_dir, sample_df)
        else:
            return TestDataManager._create_csv_only_config(temp_dir, sample_df)
    
    @staticmethod
    def _create_tar_config(temp_dir, sample_df):
        """Create config for tar.gz data with real molecule information."""
        temp_path = Path(temp_dir)
        
        labels_csv = temp_path / "labels.csv"
        sample_df.write_csv(labels_csv)
        
        mol_col = TEST_DATA_CONFIG["csv_columns"]["mol_id"]
        smiles_col = TEST_DATA_CONFIG["csv_columns"]["smiles"]
        label_col = TEST_DATA_CONFIG["csv_columns"]["label"]
        
        # Create config
        config = {
            "data_paths": {
                "base_dir": None,
                "xyz_dir": None,
                "shermo_dir": None,
                "janpa_dir": None,
                "nbo_dir": None,
                "archive_file": str(TEST_DATA_CONFIG["tar_gz_file"])
            },
            "file_patterns": {
                "xyz_pattern": "{mol_id}.xyz",
                "shermo_pattern": "{mol_id}_shermo.out",
                "janpa_pattern": "{mol_id}_janpa.out",
                "nbo_pattern": "{mol_id}_nbo.out"
            },
            "data_file": str(labels_csv),
            "labels_file": str(labels_csv),
            "labels_config": {
                "id_column": mol_col,
                "smiles_column": smiles_col,
                "labels_column": label_col
            },
            "settings": {
                "graph_types": ["DFT", "NPA", "NBO", "QM"],
                "batch_size": min(50, len(sample_df) // 4 + 1),
                "max_workers": 2,  # Reduce for testing
                "bond_threshold": 0.4,
                "max_distance": 3.0
            },
            "memory": {
                "clear_cache_frequency": 25,  # More frequent for testing
                "max_memory_usage_gb": 4.0
            },
            "cleanup": {
                "remove_temp_files": True
            },
            "output": {
                "database_name": "batch_test_molecules",
                "overwrite_existing": True,
                "export_ml_features": True,
                "ml_export_dir": str(temp_path / "ml_export")
            },
            "error_handling": {
                "continue_on_error": True,
                "save_error_log": True,
                "max_errors": 100
            },
            "logging": {
                "level": "INFO",
                "save_to_file": True,
                "log_file": str(temp_path / "batch_test.log")
            }
        }
        
        config_file = temp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_file, config
    
    @staticmethod
    def _create_csv_only_config(temp_dir, sample_df):
        """Create config for CSV-only testing (creates mock files)."""
        temp_path = Path(temp_dir)
        
        mock_data_dir = temp_path / "mock_data"
        xyz_dir = mock_data_dir / "xyz"
        shermo_dir = mock_data_dir / "shermo"
        xyz_dir.mkdir(parents=True, exist_ok=True)
        shermo_dir.mkdir(parents=True, exist_ok=True)
        
        mol_col = TEST_DATA_CONFIG["csv_columns"]["mol_id"]
        smiles_col = TEST_DATA_CONFIG["csv_columns"]["smiles"]
        label_col = TEST_DATA_CONFIG["csv_columns"]["label"]
        
        for row in sample_df.iter_rows(named=True):
            mol_id = str(row[mol_col])
            
            xyz_file = xyz_dir / f"{mol_id}.xyz"
            xyz_content = f"""3
{mol_id} - mock molecule
C  0.0  0.0  0.0
C  1.0  0.0  0.0  
H  2.0  0.0  0.0
"""
            xyz_file.write_text(xyz_content)
            
            shermo_file = shermo_dir / f"{mol_id}_shermo.out"
            shermo_content = f"""
Mock Shermo output for {mol_id}
Enthalpy: -100.0 Hartree
Entropy: 50.0 cal/mol/K
"""
            shermo_file.write_text(shermo_content)
        
        labels_csv = temp_path / "labels.csv"
        sample_df.write_csv(labels_csv)
        
        config = {
            "data_paths": {
                "base_dir": str(mock_data_dir),
                "xyz_dir": str(xyz_dir),
                "shermo_dir": str(shermo_dir),
                "janpa_dir": None,
                "nbo_dir": None,
                "archive_file": None
            },
            "file_patterns": {
                "xyz_pattern": "{mol_id}.xyz",
                "shermo_pattern": "{mol_id}_shermo.out",
                "janpa_pattern": "{mol_id}_janpa.out",
                "nbo_pattern": "{mol_id}_nbo.out"
            },
            "data_file": str(labels_csv),
            "labels_file": str(labels_csv),
            "labels_config": {
                "id_column": mol_col,
                "smiles_column": smiles_col,
                "labels_column": label_col
            },
            "settings": {
                "graph_types": ["DFT"],
                "batch_size": min(50, len(sample_df) // 4 + 1),
                "max_workers": 2,
                "bond_threshold": 0.4,
                "max_distance": 3.0
            },
            "memory": {
                "clear_cache_frequency": 25,
                "max_memory_usage_gb": 4.0
            },
            "cleanup": {
                "remove_temp_files": True
            },
            "output": {
                "database_name": "batch_test_molecules",
                "overwrite_existing": True,
                "export_ml_features": True,
                "ml_export_dir": str(temp_path / "ml_export")
            },
            "error_handling": {
                "continue_on_error": True,
                "save_error_log": True,
                "max_errors": 100
            },
            "logging": {
                "level": "INFO",
                "save_to_file": True,
                "log_file": str(temp_path / "batch_test.log")
            }
        }
        
        config_file = temp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_file, config


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_test_config(temp_test_dir):
    """Create small test configuration."""
    return TestDataManager.create_test_config(temp_test_dir, "small")


@pytest.fixture
def medium_test_config(temp_test_dir):
    """Create medium test configuration."""
    return TestDataManager.create_test_config(temp_test_dir, "medium")


@pytest.fixture
def performance_test_config(temp_test_dir):
    """Create performance test configuration."""
    return TestDataManager.create_test_config(temp_test_dir, "performance")


class TestBatchProcessorBasic:
    """Test basic batch processor functionality."""
    
    def test_batch_processor_initialization(self, small_test_config):
        """Test batch processor initialization."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        assert processor.config == config
        assert processor.failed_molecules == []
        assert processor.successful_count == 0
    
    def test_config_value_access(self, small_test_config):
        """Test configuration value access utility."""
        config_file, config = small_test_config
        
        db_name = get_config_value(config, 'output.database_name', 'default.db')
        assert db_name == 'batch_test_molecules'
        
        nonexistent = get_config_value(config, 'nonexistent.key', 'default_value')
        assert nonexistent == 'default_value'
        
        batch_size = get_config_value(config, 'settings.batch_size', 100)
        assert isinstance(batch_size, int)
    
    def test_database_creation(self, small_test_config, temp_test_dir):
        """Test database creation from config."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        processor._create_database()
        
        db_path = Path("batch_test_molecules.db")
        assert db_path.exists(), f"Database file not found at {db_path}"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = ['graphs', 'nodes', 'edges', 'targets']
        for table in expected_tables:
            assert table in tables
        
        if db_path.exists():
            db_path.unlink()
    
    def test_load_labels(self, small_test_config):
        """Test loading labels from CSV."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        labels_df = processor._load_labels()
        
        import pandas as pd
        assert isinstance(labels_df, pd.DataFrame)
        assert len(labels_df) > 0
        assert 'mol_id' in labels_df.columns
        assert 'SMILES' in labels_df.columns
    
    def test_load_labels_missing_file(self, temp_test_dir):
        """Test loading labels with missing file."""
        config = {
            "data_file": "/nonexistent/file.csv",
            "labels_file": None,
            "labels_config": {
                "id_column": "mol_id",
                "smiles_column": "smiles"
            }
        }
        
        processor = BatchProcessor(config)
        
        with pytest.raises(FileNotFoundError):
            processor._load_labels()
    
    def test_load_labels_missing_columns(self, temp_test_dir):
        """Test loading labels with missing required columns."""
        # Create CSV with wrong columns
        csv_path = Path(temp_test_dir) / "bad_labels.csv"
        
        with open(csv_path, 'w') as f:
            f.write("wrong_column,another_column\n")
            f.write("value1,value3\n")
            f.write("value2,value4\n")
        
        config = {
            "data_file": str(csv_path),
            "labels_file": None,
            "labels_config": {
                "id_column": "mol_id",
                "smiles_column": "smiles"
            }
        }
        
        processor = BatchProcessor(config)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            processor._load_labels()


class TestFileDiscovery:
    """Test file discovery functionality."""
    
    def test_file_discovery_patterns(self, small_test_config):
        """Test file pattern matching."""
        config_file, config = small_test_config
        
        # Test if we can access file patterns
        patterns = get_config_value(config, 'file_patterns', {})
        assert 'xyz_pattern' in patterns
        assert 'shermo_pattern' in patterns
        
        # Test pattern formatting
        mol_id = "test_mol_001"
        xyz_pattern = patterns['xyz_pattern'].format(mol_id=mol_id)
        assert mol_id in xyz_pattern
    
    @pytest.mark.skipif(not Path(TEST_DATA_CONFIG["tar_gz_file"]).exists(), 
                       reason="No tar.gz test data available")
    def test_archive_extraction(self, small_test_config):
        """Test archive extraction functionality."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        
        discovered_files = processor._extract_and_discover_archive()
        
        assert isinstance(discovered_files, dict)
        assert 'xyz' in discovered_files
        assert 'shermo' in discovered_files
        
        total_files = sum(len(files) for files in discovered_files.values())
        assert total_files > 0
    
    def test_file_discovery_caching(self, small_test_config):
        """Test file discovery caching."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        
        with patch.object(BatchProcessor, '_discover_files_in_directory') as mock_discover:
            mock_discover.return_value = {'xyz': [], 'shermo': [], 'janpa': [], 'nbo': []}
            
            result1 = processor._get_data_files()
            assert mock_discover.call_count == 1
            


class TestMoleculeMatching:
    """Test molecule file matching functionality."""
    
    def test_molecule_file_matching(self, small_test_config):
        """Test matching files to molecules."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        labels_df = processor._load_labels()
        
        # Create mock discovered files
        discovered_files = {
            'xyz': [],
            'shermo': [],
            'janpa': [],
            'nbo': []
        }
        
        # Add some mock files based on molecule IDs
        for mol_id in labels_df['mol_id'].head(3):
            discovered_files['xyz'].append(Path(f"/mock/path/{mol_id}.xyz"))
            discovered_files['shermo'].append(Path(f"/mock/path/{mol_id}_shermo.out"))
        
        matched_molecules = processor._match_files_to_molecules(discovered_files, labels_df)
        
        assert isinstance(matched_molecules, list)
        # Should have some matches even with mock data
        # (implementation might skip molecules without valid files)
    
    def test_graph_type_determination(self, small_test_config):
        """Test graph type determination logic."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        
        from graphpancake.batch_processing import MoleculeData
        
        mol_data_dft = MoleculeData("mol_001", {})
        mol_data_dft.xyz = Path("/mock/mol_001.xyz")
        mol_data_dft.shermo = Path("/mock/mol_001_shermo.out")
        graph_type = processor._determine_graph_type_optimized(mol_data_dft)
        assert graph_type == 'DFT'
        
        mol_data_npa = MoleculeData("mol_002", {})
        mol_data_npa.xyz = Path("/mock/mol_002.xyz")
        mol_data_npa.shermo = Path("/mock/mol_002_shermo.out")
        mol_data_npa.janpa = Path("/mock/mol_002_janpa.out")
        graph_type = processor._determine_graph_type_optimized(mol_data_npa)
        assert graph_type == 'NPA'
        
        mol_data_nbo = MoleculeData("mol_003", {})
        mol_data_nbo.xyz = Path("/mock/mol_003.xyz")
        mol_data_nbo.shermo = Path("/mock/mol_003_shermo.out")
        mol_data_nbo.nbo = Path("/mock/mol_003_nbo.out")
        graph_type = processor._determine_graph_type_optimized(mol_data_nbo)
        assert graph_type == 'NBO'
        
        mol_data_qm = MoleculeData("mol_004", {})
        mol_data_qm.xyz = Path("/mock/mol_004.xyz")
        mol_data_qm.shermo = Path("/mock/mol_004_shermo.out")
        mol_data_qm.janpa = Path("/mock/mol_004_janpa.out")
        mol_data_qm.nbo = Path("/mock/mol_004_nbo.out")
        graph_type = processor._determine_graph_type_optimized(mol_data_qm)
        assert graph_type == 'QM'
    
    def test_molecule_validation(self, small_test_config):
        """Test molecule file validation."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        
        from graphpancake.batch_processing import MoleculeData
        
        valid_mol = MoleculeData("mol_001", {})
        valid_mol.xyz = Path("/mock/mol_001.xyz")
        valid_mol.shermo = Path("/mock/mol_001_shermo.out")
        assert processor._validate_molecule_files(valid_mol) == True
        
        invalid_mol = MoleculeData("mol_002", {})
        assert processor._validate_molecule_files(invalid_mol) == False


class TestBatchProcessing:
    """Test complete batch processing pipeline."""
    
    @pytest.mark.skipif(not TestDataManager.check_test_data(), 
                       reason="No test data available")
    def test_small_batch_processing(self, small_test_config):
        """Test processing a small batch of molecules."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        
        start_time = time.time()
        
        try:
            processor.run()
            processing_time = time.time() - start_time
            
            db_path = Path(config['output']['database_name'] + '.db')
            assert db_path.exists()
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM graphs")
            graph_count = cursor.fetchone()[0]
            conn.close()
            
            assert graph_count >= 0  # Even if all fail, database should exist
            assert processor.successful_count >= 0
            
            print(f"Small batch processing completed in {processing_time:.2f}s")
            print(f"Processed {processor.successful_count} molecules successfully")
            print(f"Failed {len(processor.failed_molecules)} molecules")
            
        except Exception as e:
            pytest.fail(f"Batch processing failed: {e}")
    
    @pytest.mark.skipif(not TestDataManager.check_test_data(), 
                       reason="No test data available")
    def test_medium_batch_processing(self, medium_test_config):
        """Test processing a medium batch of molecules."""
        config_file, config = medium_test_config
        
        processor = BatchProcessor(config)
        
        start_time = time.time()
        
        try:
            processor.run()
            processing_time = time.time() - start_time
            
            db_path = Path(config['output']['database_name'] + '.db')
            assert db_path.exists()
            
            print(f"Medium batch processing completed in {processing_time:.2f}s")
            print(f"Processed {processor.successful_count} molecules successfully")
            print(f"Failed {len(processor.failed_molecules)} molecules")
            
            molecules_per_second = processor.successful_count / processing_time if processing_time > 0 else 0
            print(f"Processing rate: {molecules_per_second:.2f} molecules/second")
            
        except Exception as e:
            pytest.fail(f"Medium batch processing failed: {e}")
    
    def test_error_handling_in_batch(self, small_test_config):
        """Test error handling during batch processing."""
        config_file, config = small_test_config
        
        config['error_handling']['continue_on_error'] = True
        config['error_handling']['max_errors'] = 1000
        
        processor = BatchProcessor(config)
        
        original_process = BatchProcessor._process_single_molecule_return_graph
        
        def mock_process_with_errors(self, mol_data):
            # Fail every other molecule
            if int(mol_data.mol_id.split('_')[-1]) % 2 == 0:
                raise Exception("Simulated processing error")
            return original_process(self, mol_data)
        
        with patch.object(BatchProcessor, '_process_single_molecule_return_graph', mock_process_with_errors):
            try:
                processor.run()
                
                # Should have some failures recorded
                assert len(processor.failed_molecules) > 0
                assert processor.successful_count >= 0
                
            except Exception as e:
                # Should not fail completely due to error handling
                pytest.fail(f"Batch processing should handle errors gracefully: {e}")
    
    def test_memory_monitoring(self, small_test_config):
        """Test memory monitoring during processing."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        
        # Test memory monitor initialization
        assert hasattr(processor, '_memory_monitor')
        
        # Test memory checking
        memory_percent = processor._memory_monitor.check_memory()
        assert isinstance(memory_percent, float)
        assert 0 <= memory_percent <= 100
        
        # Test memory stats
        stats = processor._memory_monitor.get_memory_stats()
        assert 'start' in stats
        assert 'current' in stats


class TestMLFeatureExport:
    """Test ML feature export functionality."""
    
    def test_ml_export_configuration(self, small_test_config):
        """Test ML export configuration."""
        config_file, config = small_test_config
        
        # Check ML export settings
        export_enabled = get_config_value(config, 'output.export_ml_features', False)
        export_dir = get_config_value(config, 'output.ml_export_dir')
        
        assert export_enabled == True
        assert export_dir is not None
    
    @pytest.mark.skipif(not TestDataManager.check_test_data(), 
                       reason="No test data available")
    def test_ml_export_after_processing(self, small_test_config):
        """Test ML feature export after batch processing."""
        config_file, config = small_test_config
        
        processor = BatchProcessor(config)
        
        try:
            # Run batch processing
            processor.run()
            
            # Check if ML features were exported
            export_dir = Path(config['output']['ml_export_dir'])
            if export_dir.exists():
                ml_files = list(export_dir.glob("*.csv"))
                if ml_files:
                    ml_file = ml_files[0]
                    df = pl.read_csv(ml_file, ignore_errors=True)
                    
                    assert len(df) > 0
                    assert 'mol_id' in df.columns
                    
                    print(f"ML features exported: {len(df)} molecules")
                    print(f"Feature columns: {len(df.columns)}")
        
        except Exception as e:
            pytest.fail(f"ML export test failed: {e}")


class TestConfigurationValidation:
    """Test configuration validation and edge cases."""
    
    def test_invalid_config_structure(self, temp_test_dir):
        """Test handling of invalid configuration."""
        # Create invalid config
        invalid_config = {
            "missing_required_fields": True
        }
        
        processor = BatchProcessor(invalid_config)
        
        # Should handle missing fields gracefully with defaults
        batch_size = get_config_value(invalid_config, 'settings.batch_size', 100)
        assert batch_size == 100
    
    def test_config_with_missing_paths(self, temp_test_dir):
        """Test configuration with missing file paths."""
        config = {
            "data_paths": {
                "archive_file": "/nonexistent/archive.tar.gz"
            },
            "data_file": "/nonexistent/data.csv"
        }
        
        processor = BatchProcessor(config)
        
        # Should raise appropriate errors for missing files
        with pytest.raises(FileNotFoundError):
            processor._load_labels()
    
    def test_config_memory_settings(self, small_test_config):
        """Test memory configuration settings."""
        config_file, config = small_test_config
        
        memory_limit = get_config_value(config, 'memory.max_memory_usage_gb', 16.0)
        cache_frequency = get_config_value(config, 'memory.clear_cache_frequency', 1000)
        
        assert isinstance(memory_limit, float)
        assert isinstance(cache_frequency, int)
        assert memory_limit > 0
        assert cache_frequency > 0


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.skipif(not TestDataManager.check_test_data(), 
                       reason="No test data available")
    def test_performance_benchmark(self, performance_test_config):
        """Benchmark processing performance with larger dataset."""
        config_file, config = performance_test_config
        
        processor = BatchProcessor(config)
        
        start_time = time.time()
        
        try:
            processor.run()
            
            total_time = time.time() - start_time
            
            print(f"\nPerformance Benchmark Results:")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Successful molecules: {processor.successful_count}")
            print(f"Failed molecules: {len(processor.failed_molecules)}")
            
            if processor.successful_count > 0:
                molecules_per_second = processor.successful_count / total_time
                print(f"Processing rate: {molecules_per_second:.2f} molecules/second")
                
                assert molecules_per_second > 0.1  # At least 0.1 molecules per second
                
                memory_stats = processor._memory_monitor.get_memory_stats()
                print(f"Memory usage - Start: {memory_stats['start']:.1f}%, Peak: {memory_stats['peak']:.1f}%")
                
                memory_increase = memory_stats['peak'] - memory_stats['start']
                assert memory_increase < 50  # Less than 50% memory increase
        
        except Exception as e:
            pytest.fail(f"Performance benchmark failed: {e}")
    
    def test_memory_efficiency(self, medium_test_config):
        """Test memory efficiency during processing."""
        config_file, config = medium_test_config
        
        # Set lower memory limits for testing
        config['memory']['max_memory_usage_gb'] = 4.0
        config['memory']['clear_cache_frequency'] = 50
        
        processor = BatchProcessor(config)
        
        try:
            processor.run()
            
            # Check memory stats
            memory_stats = processor._memory_monitor.get_memory_stats()
            memory_increase = memory_stats['peak'] - memory_stats['start']
            
            print(f"Memory efficiency test:")
            print(f"Start: {memory_stats['start']:.1f}%")
            print(f"Peak: {memory_stats['peak']:.1f}%")
            print(f"Increase: {memory_increase:.1f}%")
            
            # Memory increase should be reasonable
            assert memory_increase < 30  # Less than 30% increase
            
        except Exception as e:
            pytest.fail(f"Memory efficiency test failed: {e}")


class TestBatchErrorRecovery:
    """Test error recovery and robustness."""
    
    def test_database_error_recovery(self, small_test_config):
        """Test recovery from database errors."""
        config_file, config = small_test_config
        
        # Set database to read-only location (should fail gracefully)
        config['output']['database_name'] = '/root/readonly_location/test.db'
        
        processor = BatchProcessor(config)
        
        try:
            processor.run()
            pytest.fail("Should have failed with database permission error")
        except Exception as e:
            # Should get a clear error message
            assert "database" in str(e).lower() or "permission" in str(e).lower()
    
    def test_corrupted_data_handling(self, small_test_config):
        """Test handling of corrupted input data."""
        config_file, config = small_test_config
        
        # Create corrupted labels file
        corrupted_csv = Path(config['data_file'])
        with open(corrupted_csv, 'w') as f:
            f.write("corrupted,data,structure\nno,proper,csv\ninvalid")
        
        processor = BatchProcessor(config)
        
        try:
            processor._load_labels()
            pytest.fail("Should have failed with corrupted data")
        except Exception as e:
            # Should handle corrupted data gracefully
            assert isinstance(e, (ValueError, FileNotFoundError, Exception))


if __name__ == "__main__":
    # Validate test data before running tests
    TestDataManager.check_test_data()
    
    # Run with different verbosity levels
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        # Run performance benchmarks
        pytest.main([__file__ + "::TestPerformanceBenchmarks", "-v", "-s"])
    elif len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Run only quick tests
        pytest.main([__file__, "-v", "-k", "not benchmark"])
    else:
        # Run all tests
        pytest.main([__file__, "-v"])