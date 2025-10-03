import pytest
import tempfile
import shutil
import sqlite3
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tarfile
import os
import subprocess
import json

TEST_DATA_CONFIG = {
    "tar_gz_file": "/home/uvx4187/graphpancake_data/SS-07-13_data.tar.gz",
    
    "labels_csv": "/home/uvx4187/graphpancake_data/SS-07-13_labels.csv",
    
    "csv_columns": {
        "mol_id": "mol_id",
        "smiles": "SMILES", 
        "label": "logP" 
    },
    
    "test_db_name": "test_molecules.db",
    "small_sample_size": 10,  # Number of molecules for quick tests
    "medium_sample_size": 50,  # For more comprehensive tests
    "large_sample_size": 200   # For performance tests
}

class TestDataValidator:
    """Validate and manage test data using real labels CSV."""
    
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
    def get_sample_molecules(sample_size="small"):
        """Get a sample of molecules from the labels CSV."""
        labels_path = Path(TEST_DATA_CONFIG["labels_csv"])
        
        if not labels_path.exists():
            pytest.skip(f"Labels CSV not found: {labels_path}")
        
        df = pd.read_csv(labels_path)
        
        if sample_size == "small":
            n_samples = TEST_DATA_CONFIG["small_sample_size"]
        elif sample_size == "medium":
            n_samples = TEST_DATA_CONFIG["medium_sample_size"]
        elif sample_size == "large":
            n_samples = TEST_DATA_CONFIG["large_sample_size"]
        else:
            n_samples = int(sample_size) if isinstance(sample_size, (int, str)) else 10
        
        sample_df = df.head(n_samples).copy()
        
        return sample_df
    
    @staticmethod
    def extract_sample_files(temp_dir, sample_size="small"):
        """Extract sample files for the selected molecules."""
        data_type = TestDataValidator.check_test_data()
        sample_df = TestDataValidator.get_sample_molecules(sample_size)
        
        if data_type == "full":
            return TestDataValidator._extract_from_tar_with_labels(temp_dir, sample_df)
        else:
            return TestDataValidator._create_mock_files_with_labels(temp_dir, sample_df)
    
    @staticmethod
    def _extract_from_tar_with_labels(temp_dir, sample_df):
        """Extract sample files from tar.gz based on molecule IDs in labels."""
        tar_path = Path(TEST_DATA_CONFIG["tar_gz_file"])
        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        mol_col = TEST_DATA_CONFIG["csv_columns"]["mol_id"]
        mol_ids = sample_df[mol_col].tolist()
        
        extracted_files = {"xyz": [], "shermo": [], "janpa": [], "nbo": []}
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = tar.getmembers()
            
            for member in members:
                filename = Path(member.name).name
                
                for mol_id in mol_ids:
                    if mol_id in filename:
                        try:
                            tar.extract(member, extract_dir)
                            extracted_path = extract_dir / member.name
                            
                            if filename.endswith('.xyz'):
                                extracted_files["xyz"].append(extracted_path)
                            elif 'shermo' in filename.lower():
                                extracted_files["shermo"].append(extracted_path)
                            elif 'janpa' in filename.lower():
                                extracted_files["janpa"].append(extracted_path)
                            elif 'nbo' in filename.lower():
                                extracted_files["nbo"].append(extracted_path)
                        except Exception as e:
                            print(f"Warning: Could not extract {member.name}: {e}")
                        break
        
        sample_csv = extract_dir / "sample_labels.csv"
        sample_df.to_csv(sample_csv, index=False)
        
        return {
            "base_dir": extract_dir,
            "molecules_csv": sample_csv,
            "sample_df": sample_df,
            "extracted_files": extracted_files
        }
    
    @staticmethod
    def _create_mock_files_with_labels(temp_dir, sample_df):
        """Create mock files when tar.gz is not available."""
        mock_dir = Path(temp_dir) / "mock_data"
        mock_dir.mkdir(exist_ok=True)
        
        mol_col = TEST_DATA_CONFIG["csv_columns"]["mol_id"]
        mol_ids = sample_df[mol_col].tolist()
        
        xyz_dir = mock_dir / "xyz_files"
        shermo_dir = mock_dir / "shermo_files"
        xyz_dir.mkdir(exist_ok=True)
        shermo_dir.mkdir(exist_ok=True)
        
        mock_files = {"xyz": [], "shermo": [], "janpa": [], "nbo": []}
        
        for mol_id in mol_ids:
            # Create mock XYZ file
            xyz_file = xyz_dir / f"{mol_id}.xyz"
            xyz_content = f"""3
{mol_id} - mock molecule
C  0.0  0.0  0.0
C  1.0  0.0  0.0  
H  2.0  0.0  0.0
"""
            xyz_file.write_text(xyz_content)
            mock_files["xyz"].append(xyz_file)
            
            # Create mock Shermo file
            shermo_file = shermo_dir / f"{mol_id}_shermo.out"
            shermo_content = f"""
Mock Shermo output for {mol_id}
Enthalpy: -100.0 Hartree
Entropy: 50.0 cal/mol/K
"""
            shermo_file.write_text(shermo_content)
            mock_files["shermo"].append(shermo_file)
        
        # Create sample labels CSV in temp directory
        sample_csv = mock_dir / "sample_labels.csv"
        sample_df.to_csv(sample_csv, index=False)
        
        return {
            "base_dir": mock_dir,
            "molecules_csv": sample_csv,
            "sample_df": sample_df,
            "extracted_files": mock_files
        }


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_test_data(temp_test_dir):
    """Extract sample test data for small tests."""
    return TestDataValidator.extract_sample_files(temp_test_dir, "small")


@pytest.fixture
def medium_test_data(temp_test_dir):
    """Extract sample test data for medium tests."""
    return TestDataValidator.extract_sample_files(temp_test_dir, "medium")


@pytest.fixture
def large_test_data(temp_test_dir):
    """Extract sample test data for large tests."""
    return TestDataValidator.extract_sample_files(temp_test_dir, "large")


@pytest.fixture
def test_database(temp_test_dir):
    """Create a test database path."""
    return Path(temp_test_dir) / TEST_DATA_CONFIG["test_db_name"]


class TestCreateDatabase:
    """Test database creation commands."""
    
    def test_create_database_basic(self, test_database):
        """Test basic database creation."""
        cmd = [
            "graphpancake", "create-db", str(test_database),
            "--graph-types", "DFT", "NPA", "NBO", "QM"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert test_database.exists()
        
        # Verify database schema
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = ['graphs', 'nodes', 'edges', 'targets']
        for table in expected_tables:
            assert table in tables
    
    def test_create_database_force_overwrite(self, test_database):
        """Test database creation with force overwrite."""
        # Create initial database
        cmd1 = ["graphpancake", "create-db", str(test_database)]
        subprocess.run(cmd1, capture_output=True)
        
        # Try to overwrite without force (should fail)
        cmd2 = ["graphpancake", "create-db", str(test_database)]
        result = subprocess.run(cmd2, capture_output=True, text=True)
        assert result.returncode == 1
        assert "already exists" in result.stderr
        
        # Overwrite with force (should succeed)
        cmd3 = ["graphpancake", "create-db", str(test_database), "--force"]
        result = subprocess.run(cmd3, capture_output=True, text=True)
        assert result.returncode == 0
    
    def test_create_database_invalid_path(self):
        """Test database creation with invalid path."""
        invalid_path = "/invalid/path/database.db"
        cmd = ["graphpancake", "create-db", invalid_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 1


class TestLoadData:
    """Test data loading commands."""
    
    def test_load_single_molecule(self, test_database, sample_test_data):
        """Test loading a single molecule."""
        # Create database first
        subprocess.run([
            "graphpancake", "create-db", str(test_database),
            "--graph-types", "DFT", "NPA", "NBO", "QM"
        ], capture_output=True)
        
        # Get sample molecule data
        sample_df = sample_test_data["sample_df"]
        extracted_files = sample_test_data["extracted_files"]
        
        if len(extracted_files["xyz"]) == 0 or len(extracted_files["shermo"]) == 0:
            pytest.skip("No XYZ or Shermo files found in test data")
        
        # Use first molecule
        mol_col = TEST_DATA_CONFIG["csv_columns"]["mol_id"]
        smiles_col = TEST_DATA_CONFIG["csv_columns"]["smiles"]
        
        first_mol = sample_df.iloc[0]
        mol_id = first_mol[mol_col]
        mol_smiles = first_mol[smiles_col]
        
        xyz_file = None
        shermo_file = None
        
        for file_path in extracted_files["xyz"]:
            if mol_id in str(file_path):
                xyz_file = file_path
                break
        
        for file_path in extracted_files["shermo"]:
            if mol_id in str(file_path):
                shermo_file = file_path
                break
        
        if not xyz_file or not shermo_file:
            xyz_file = extracted_files["xyz"][0]
            shermo_file = extracted_files["shermo"][0]
        
        cmd = [
            "graphpancake", "load-data",
            "--database", str(test_database),
            "--mol-id", str(mol_id),
            "--xyz-file", str(xyz_file),
            "--shermo-output", str(shermo_file),
            "--smiles", str(mol_smiles),
            "--graph-type", "DFT"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM graphs WHERE graph_id = ?", (str(mol_id),))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count > 0
    
    def test_load_data_missing_files(self, test_database):
        """Test loading data with missing files."""
        subprocess.run([
            "graphpancake", "create-db", str(test_database)
        ], capture_output=True)
        
        cmd = [
            "graphpancake", "load-data",
            "--database", str(test_database),
            "--mol-id", "test_mol",
            "--xyz-file", "/nonexistent/file.xyz",
            "--shermo-output", "/nonexistent/shermo.out",
            "--smiles", "CCO"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 1
        assert "not found" in result.stderr
    
    def test_load_data_missing_database(self):
        """Test loading data with missing database."""
        cmd = [
            "graphpancake", "load-data",
            "--database", "/nonexistent/database.db",
            "--mol-id", "test_mol",
            "--xyz-file", "/some/file.xyz",
            "--shermo-output", "/some/shermo.out",
            "--smiles", "CCO"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 1
        assert "Database not found" in result.stderr


class TestQueryDatabase:
    """Test database query commands."""
    
    @pytest.fixture
    def populated_database(self, test_database, sample_test_data):
        """Create a database with sample data."""
        subprocess.run([
            "graphpancake", "create-db", str(test_database), "--force"
        ], capture_output=True)
        
        sample_df = sample_test_data["sample_df"]
        extracted_files = sample_test_data["extracted_files"]
        
        mol_col = TEST_DATA_CONFIG["csv_columns"]["mol_id"]
        smiles_col = TEST_DATA_CONFIG["csv_columns"]["smiles"]
        
        loaded_count = 0
        max_to_load = min(3, len(sample_df))
        
        for idx, row in sample_df.head(max_to_load).iterrows():
            mol_id = str(row[mol_col])
            mol_smiles = str(row[smiles_col])
            
            xyz_file = None
            shermo_file = None
            
            for file_path in extracted_files["xyz"]:
                if mol_id in str(file_path):
                    xyz_file = file_path
                    break
            
            for file_path in extracted_files["shermo"]:
                if mol_id in str(file_path):
                    shermo_file = file_path
                    break
            
            if not xyz_file and extracted_files["xyz"]:
                xyz_file = extracted_files["xyz"][loaded_count % len(extracted_files["xyz"])]
            if not shermo_file and extracted_files["shermo"]:
                shermo_file = extracted_files["shermo"][loaded_count % len(extracted_files["shermo"])]
            
            if xyz_file and shermo_file:
                result = subprocess.run([
                    "graphpancake", "load-data",
                    "--database", str(test_database),
                    "--mol-id", mol_id,
                    "--xyz-file", str(xyz_file),
                    "--shermo-output", str(shermo_file),
                    "--smiles", mol_smiles,
                    "--graph-type", "DFT"
                ], capture_output=True)
                
                if result.returncode == 0:
                    loaded_count += 1
        
        return test_database
    
    def test_query_all_molecules(self, populated_database):
        """Test querying all molecules."""
        cmd = ["graphpancake", "query", "--database", str(populated_database)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "Found" in result.stderr
    
    def test_query_with_limit(self, populated_database):
        """Test querying with limit."""
        cmd = [
            "graphpancake", "query", 
            "--database", str(populated_database),
            "--limit", "2"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    
    def test_query_json_format(self, populated_database, temp_test_dir):
        """Test querying with JSON output."""
        output_file = Path(temp_test_dir) / "results.json"
        
        cmd = [
            "graphpancake", "query",
            "--database", str(populated_database),
            "--format", "json",
            "--output", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert output_file.exists()
        
        # Verify JSON format
        with open(output_file) as f:
            data = json.load(f)
        assert isinstance(data, list)
    
    def test_query_csv_format(self, populated_database, temp_test_dir):
        """Test querying with CSV output."""
        output_file = Path(temp_test_dir) / "results.csv"
        
        cmd = [
            "graphpancake", "query",
            "--database", str(populated_database),
            "--format", "csv",
            "--output", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert output_file.exists()
        
        # Verify CSV format
        df = pd.read_csv(output_file)
        assert len(df) > 0
    
    def test_query_nonexistent_database(self):
        """Test querying nonexistent database."""
        cmd = [
            "graphpancake", "query",
            "--database", "/nonexistent/database.db"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 1
        assert "Database not found" in result.stderr


class TestExportMLFeatures:
    """Test ML feature export commands."""
    
    def test_export_ml_csv(self, populated_database, temp_test_dir):
        """Test exporting ML features in CSV format."""
        output_dir = Path(temp_test_dir) / "ml_export"
        
        cmd = [
            "graphpancake", "export-ml",
            "--database", str(populated_database),
            "--output", str(output_dir),
            "--format", "csv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check output files - file may not exist if no molecules were successfully loaded
        csv_file = output_dir / "ml_features.csv"
        if csv_file.exists():
            # Verify CSV content if file exists
            df = pd.read_csv(csv_file)
            assert len(df) > 0
            assert 'mol_id' in df.columns
        else:
            # If no file exists, should have warning about no molecules
            assert "No molecules found" in result.stderr or "No features were extracted" in result.stderr
    
    def test_export_ml_json(self, populated_database, temp_test_dir):
        """Test exporting ML features in JSON format."""
        output_dir = Path(temp_test_dir) / "ml_export"
        
        cmd = [
            "graphpancake", "export-ml",
            "--database", str(populated_database),
            "--output", str(output_dir),
            "--format", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check output files - file may not exist if no molecules were successfully loaded
        json_file = output_dir / "ml_features.json"
        if json_file.exists():
            # Verify JSON content if file exists
            with open(json_file) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
        else:
            assert "No molecules found" in result.stderr or "No features were extracted" in result.stderr
    
    def test_export_ml_with_graph_type_filter(self, populated_database, temp_test_dir):
        """Test exporting ML features with graph type filter."""
        output_dir = Path(temp_test_dir) / "ml_export"
        
        cmd = [
            "graphpancake", "export-ml",
            "--database", str(populated_database),
            "--output", str(output_dir),
            "--graph-type", "DFT",
            "--format", "csv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    
    def test_export_ml_empty_database(self, test_database, temp_test_dir):
        """Test exporting from empty database."""
        # Create empty database
        subprocess.run([
            "graphpancake", "create-db", str(test_database), "--force"
        ], capture_output=True)
        
        output_dir = Path(temp_test_dir) / "ml_export"
        
        cmd = [
            "graphpancake", "export-ml",
            "--database", str(test_database),
            "--output", str(output_dir),
            "--format", "csv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "No molecules found" in result.stderr


class TestAddLabels:
    """Test label addition commands."""
    
    def test_add_manual_label(self, populated_database):
        """Test adding a manual label."""
        conn = sqlite3.connect(populated_database)
        cursor = conn.cursor()
        cursor.execute("SELECT graph_id FROM graphs LIMIT 1")
        mol_id = cursor.fetchone()[0]
        conn.close()
        
        cmd = [
            "graphpancake", "add-labels",
            "--database", str(populated_database),
            "--graph-id", mol_id,
            "--label-name", "test_property",
            "--value", "1.5",
            "--label-type", "regression"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    
    def test_add_labels_from_csv(self, populated_database, temp_test_dir):
        """Test adding labels from CSV file."""
        conn = sqlite3.connect(populated_database)
        df_mols = pd.read_sql_query("SELECT graph_id as mol_id FROM graphs", conn)
        conn.close()
        
        if len(df_mols) == 0:
            pytest.skip("No molecules in database to add labels to")
        
        sample_df = TestDataValidator.get_sample_molecules("small")
        label_col = TEST_DATA_CONFIG["csv_columns"]["label"]
        mol_col = TEST_DATA_CONFIG["csv_columns"]["mol_id"]
        
        labels_data = []
        for _, db_row in df_mols.iterrows():
            mol_id = db_row['mol_id']
            matching_rows = sample_df[sample_df[mol_col].astype(str) == str(mol_id)]
            if len(matching_rows) > 0:
                label_value = matching_rows.iloc[0][label_col]
                labels_data.append({'mol_id': mol_id, 'property_value': label_value})
            else:
                labels_data.append({'mol_id': mol_id, 'property_value': 1.0})
        
        labels_df = pd.DataFrame(labels_data)
        labels_file = Path(temp_test_dir) / "labels.csv"
        labels_df.to_csv(labels_file, index=False)
        
        cmd = [
            "graphpancake", "add-labels",
            "--database", str(populated_database),
            "--csv-file", str(labels_file),
            "--id-column", "mol_id",
            "--label-column", "property_value",
            "--label-name", "test_property",
            "--label-type", "regression"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    
    def test_add_labels_invalid_molecule(self, populated_database):
        """Test adding labels to nonexistent molecule."""
        cmd = [
            "graphpancake", "add-labels",
            "--database", str(populated_database),
            "--graph-id", "nonexistent_molecule",
            "--label-name", "test_property",
            "--value", "1.5"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 1
        assert "not found" in result.stderr


class TestDatabaseStats:
    """Test database statistics commands."""
    
    def test_stats_basic(self, populated_database):
        """Test basic database statistics."""
        cmd = ["graphpancake", "stats", str(populated_database)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "Database Statistics" in result.stdout
        assert "Total molecules" in result.stdout
    
    def test_stats_detailed(self, populated_database):
        """Test detailed database statistics."""
        cmd = ["graphpancake", "stats", str(populated_database), "--detailed"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "Detailed Analysis" in result.stdout
    
    def test_stats_nonexistent_database(self):
        """Test statistics on nonexistent database."""
        cmd = ["graphpancake", "stats", "/nonexistent/database.db"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 1
        assert "Database not found" in result.stderr


class TestFilterData:
    """Test data filtering commands."""
    
    def test_filter_by_criteria_summary(self, populated_database):
        """Test filtering with summary output."""
        cmd = [
            "graphpancake", "filter", "by-criteria",
            "--database", str(populated_database),
            "--where", "num_atoms > 0",
            "--show", "summary"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    
    def test_filter_by_criteria_details(self, populated_database):
        """Test filtering with detailed output."""
        cmd = [
            "graphpancake", "filter", "by-criteria",
            "--database", str(populated_database),
            "--where", "num_atoms > 0",
            "--show", "details"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    
    def test_filter_by_criteria_ids_only(self, populated_database):
        """Test filtering with IDs only output."""
        cmd = [
            "graphpancake", "filter", "by-criteria",
            "--database", str(populated_database),
            "--where", "num_atoms > 0",
            "--show", "ids"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    
    def test_filter_save_to_file(self, populated_database, temp_test_dir):
        """Test filtering with file output."""
        output_file = Path(temp_test_dir) / "filtered_results.csv"
        
        cmd = [
            "graphpancake", "filter", "by-criteria",
            "--database", str(populated_database),
            "--where", "num_atoms > 0",
            "--show", "details",
            "--format", "csv",
            "--output", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert output_file.exists()
    
    def test_filter_no_matches(self, populated_database):
        """Test filtering with no matches."""
        cmd = [
            "graphpancake", "filter", "by-criteria",
            "--database", str(populated_database),
            "--where", "num_atoms > 999999",
            "--show", "summary"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "No graphs match" in result.stderr


class TestDeleteData:
    """Test data deletion commands."""
    
    def test_delete_nonexistent_graph(self, populated_database):
        """Test deleting nonexistent graph."""
        cmd = [
            "graphpancake", "delete", "graph",
            "--database", str(populated_database),
            "--graph-id", "nonexistent_graph",
            "--confirm"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 1
        assert "not found" in result.stderr
    
    def test_delete_by_criteria_no_matches(self, populated_database):
        """Test deleting by criteria with no matches."""
        cmd = [
            "graphpancake", "delete", "by-criteria",
            "--database", str(populated_database),
            "--where", "num_atoms > 999999",
            "--confirm"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "No graphs match" in result.stderr


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""
    
    def test_no_command(self):
        """Test CLI with no command."""
        result = subprocess.run(["graphpancake"], capture_output=True, text=True)
        assert result.returncode == 1
    
    def test_invalid_command(self):
        """Test CLI with invalid command."""
        result = subprocess.run(["graphpancake", "invalid-command"], capture_output=True, text=True)
        assert result.returncode != 0
    
    def test_help_commands(self):
        """Test help for all commands."""
        commands = [
            ["graphpancake", "--help"],
            ["graphpancake", "create-db", "--help"],
            ["graphpancake", "load-data", "--help"],
            ["graphpancake", "query", "--help"],
            ["graphpancake", "export-ml", "--help"],
            ["graphpancake", "add-labels", "--help"],
            ["graphpancake", "stats", "--help"],
            ["graphpancake", "delete", "--help"],
            ["graphpancake", "filter", "--help"]
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0
            assert "usage:" in result.stdout.lower() or "help" in result.stdout.lower()
    
    def test_version_command(self):
        """Test version command."""
        result = subprocess.run(["graphpancake", "--version"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "graphpancake" in result.stdout


@pytest.fixture
def populated_database(test_database, sample_test_data):
    """Create a database with sample data for multiple test classes."""
    subprocess.run([
        "graphpancake", "create-db", str(test_database), "--force"
    ], capture_output=True)
    
    sample_df = sample_test_data["sample_df"]
    extracted_files = sample_test_data["extracted_files"]
    
    mol_col = TEST_DATA_CONFIG["csv_columns"]["mol_id"]
    smiles_col = TEST_DATA_CONFIG["csv_columns"]["smiles"]
    
    loaded_count = 0
    max_to_load = min(3, len(sample_df))
    
    for idx, row in sample_df.head(max_to_load).iterrows():
        mol_id = str(row[mol_col])
        mol_smiles = str(row[smiles_col])
        
        xyz_file = None
        shermo_file = None
        
        for file_path in extracted_files["xyz"]:
            if mol_id in str(file_path):
                xyz_file = file_path
                break
        
        for file_path in extracted_files["shermo"]:
            if mol_id in str(file_path):
                shermo_file = file_path
                break
        
        if not xyz_file and extracted_files["xyz"]:
            xyz_file = extracted_files["xyz"][loaded_count % len(extracted_files["xyz"])]
        if not shermo_file and extracted_files["shermo"]:
            shermo_file = extracted_files["shermo"][loaded_count % len(extracted_files["shermo"])]
        
        if xyz_file and shermo_file:
            result = subprocess.run([
                "graphpancake", "load-data",
                "--database", str(test_database),
                "--mol-id", mol_id,
                "--xyz-file", str(xyz_file),
                "--shermo-output", str(shermo_file),
                "--smiles", mol_smiles,
                "--graph-type", "DFT"
            ], capture_output=True)
            
            if result.returncode == 0:
                loaded_count += 1
    
    return test_database


if __name__ == "__main__":
    TestDataValidator.check_test_data()
    pytest.main([__file__])