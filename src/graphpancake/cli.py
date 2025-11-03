#!/usr/bin/env python3
"""
graphpancake: electronic structure theory-based molecular graphs, because computers haven't taken organic chemistry class

Usage:
    graphpancake <command> [options]

Commands:
    create-db    Create a new molecular graph database
    load-data    Load QM data for a single molecule
    query        Query and filter the database
    export-ml    Export ML-ready features
    add-labels   Add target labels from CSV file
    stats        Show database statistics
    delete       Delete graphs from database

Examples:

    # Create database
    graphpancake create-db molecules.db
    
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
        
    # Show database statistics
    graphpancake stats molecules.db --detailed
    
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
        
    # Export all quantitative features to CSV
    graphpancake export-ml \
        --database molecules.db \
        --output ml_features/ \
        --graph-type QM \
        --format csv

    # Export all quantitative features to JSON
    graphpancake export-ml \
        --database molecules.db \
        --output ml_features/ \
        --graph-type QM \
        --format json

Alternative Usage (if entry point not available):
    python -m graphpancake.cli <command> [options]
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import pandas as pd
import sqlite3

try:
    from .graph import MolecularGraph
    from .classes import DictData
    from .functions import generate_qm_data_dict
    from ._version import __version__
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from graphpancake.graph import MolecularGraph
    from graphpancake.classes import DictData
    from graphpancake.functions import generate_qm_data_dict
    from graphpancake._version import __version__


def print_logo():
    logo = """
    ---------------------------------------------------------------------
    |                        graphpancake v. 1.0                        |
    |                                                                   |
    |        electronic structure theory-based molecular graphs,        |
    |      because computers haven't taken organic chemistry class      |
    |                                                                   |
    |                          Sneha Sil, 2025                          |
    ---------------------------------------------------------------------

    """.format(version=__version__)
    
    print(logo)


class graphpancakeCLI:
    """Main CLI handler for graphpancake operations."""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self, level=logging.INFO, log_file=None):
        """Setup logging configuration."""
        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)
    
    def create_database(self, args):
        """Create a new molecular graph database."""
        db_path = Path(args.database)
        
        if db_path.exists() and not args.force:
            self.logger.error(f"Database {db_path} already exists. Use --force to overwrite.")
            return 1
        
        try:
            graph_types = args.graph_types or ['DFT', 'NPA', 'NBO', 'QM']
            MolecularGraph.create_database(str(db_path), graph_types=graph_types)
            
            self.logger.info(f"\tCreated database: {db_path}")
            self.logger.info(f"\tSupported graph types: {', '.join(graph_types)}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to create database: {e}")
            return 1
    
    def load_data(self, args):
        """Load QM data for a single molecule."""
        if not Path(args.database).exists():
            self.logger.error(f"Database not found: {args.database}")
            return 1
        
        if not Path(args.xyz_file).exists():
            self.logger.error(f".xyz file not found: {args.xyz_file}")
            return 1
        
        if not Path(args.shermo_output).exists():
            self.logger.error(f"Shermo output file not found: {args.shermo_output}")
            return 1
        
        try:
            self.logger.info(f"\t Processing molecule: {args.mol_id}, {args.smiles}")
            qm_data = generate_qm_data_dict(
                mol_id=args.mol_id,
                smiles=args.smiles or "unknown",
                xyz_file=args.xyz_file,
                shermo_output=args.shermo_output,
                janpa_output=args.janpa_output,
                nbo_output=args.nbo_output
            )
            
            dict_data = DictData(qm_data)
            
            mol_graph = MolecularGraph(dict_data)
            
            mol_graph.graph_info.id = args.mol_id
            mol_graph._qm_data.graph_type = args.graph_type
            
            mol_graph.save_to_database(args.database)
            
            self.logger.info(f"\tSuccessfully loaded {args.mol_id} into database")
            return 0
            
        except Exception as e:
            self.logger.error(f"\tFailed to load molecule {args.mol_id}: {e}")
            return 1
    
    def query_database(self, args):
        """Query and display database contents."""
        if not Path(args.database).exists():
            self.logger.error(f"Database not found: {args.database}")
            return 1
        
        try:
            conn = sqlite3.connect(args.database)
            
            query = "SELECT * FROM graphs"
            params = []
            
            if args.graph_type:
                query += " WHERE graph_type = ?"
                params.append(args.graph_type)
            
            if args.limit:
                query += " LIMIT ?"
                params.append(args.limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if len(df) == 0:
                self.logger.info("\tNo molecules found matching criteria")
                return 0
            
            if args.output:
                output_path = Path(args.output)
                if args.format == 'csv':
                    df.to_csv(output_path, index=False)
                    self.logger.info(f"\tResults saved to {output_path}")
                elif args.format == 'json':
                    df.to_json(output_path, orient='records', indent=2)
                    self.logger.info(f"\tResults saved to {output_path}")
            else:
                if args.format == 'json':
                    print(df.to_json(orient='records', indent=2))
                elif args.format == 'csv':
                    print(df.to_csv(index=False))
                else:
                    print(df.to_string(index=False))
            
            self.logger.info(f"\tFound {len(df)} molecules")
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to query database: {e}")
            return 1
    
    def export_ml_features(self, args):
        """Export ML-ready features from database."""
        if not Path(args.database).exists():
            self.logger.error(f"Database not found: {args.database}")
            return 1
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            conn = sqlite3.connect(args.database)
            query = "SELECT graph_id as mol_id, graph_type FROM graphs"
            params = []
            if args.graph_type:
                query += " WHERE graph_type = ?"
                params.append(args.graph_type)
            
            molecules = pd.read_sql_query(query, conn, params=params)
            
            if len(molecules) == 0:
                self.logger.warning("No molecules found for ML export")
                return 0
            
            graph_features = []
            node_features = []
            edge_features = []
            target_features = []
            
            for _, row in molecules.iterrows():
                mol_id, graph_type = row['mol_id'], row['graph_type']
                try:
                    mol_graph = MolecularGraph()
                    mol_graph.load_from_database(args.database, mol_id, graph_type)
                    graph_feat = {
                        'mol_id': mol_id,
                        'graph_type': graph_type,
                        'num_atoms': len(mol_graph.nodes) if hasattr(mol_graph, 'nodes') else 0,
                        'num_edges': len(mol_graph.edges) if hasattr(mol_graph, 'edges') else 0,
                    }
                    if hasattr(mol_graph, 'metadata') and mol_graph.metadata:
                        for key, value in mol_graph.metadata.items():
                            if isinstance(value, (int, float, str)):
                                graph_feat[f'meta_{key}'] = value
                    graph_features.append(graph_feat)
                    if hasattr(mol_graph, 'nodes'):
                        for node_id, node_data in mol_graph.nodes.items():
                            node_feat = {
                                'mol_id': mol_id,
                                'node_id': node_id,
                                'graph_type': graph_type
                            }
                            for key, value in node_data.items():
                                if isinstance(value, (int, float)):
                                    node_feat[f'node_{key}'] = value
                            if node_feat:
                                node_features.append(node_feat)
                    if hasattr(mol_graph, 'edges'):
                        for edge_id, edge_data in mol_graph.edges.items():
                            edge_feat = {
                                'mol_id': mol_id,
                                'edge_id': edge_id,
                                'graph_type': graph_type
                            }
                            for key, value in edge_data.items():
                                if isinstance(value, (int, float)):
                                    edge_feat[f'edge_{key}'] = value
                            if edge_feat:
                                edge_features.append(edge_feat)
                    if hasattr(mol_graph, 'targets'):
                        for target_name, target_value in mol_graph.targets.items():
                            if isinstance(target_value, (int, float)):
                                target_features.append({
                                    'mol_id': mol_id,
                                    'target_name': target_name,
                                    'target_value': target_value
                                })
                except Exception as e:
                    self.logger.warning(f"Failed to process {mol_id}: {e}")
                    continue
            combined_features = []
            for graph_feat in graph_features:
                combined_feat = dict(graph_feat)
                targets = [tf for tf in target_features if tf['mol_id'] == graph_feat['mol_id']]
                for target in targets:
                    combined_feat[f"target_{target['target_name']}"] = target['target_value']
                nodes = [nf for nf in node_features if nf['mol_id'] == graph_feat['mol_id']]
                for node in nodes:
                    for k, v in node.items():
                        if k not in ['mol_id', 'node_id', 'graph_type']:
                            combined_feat[f"node_{k}"] = v
                edges = [ef for ef in edge_features if ef['mol_id'] == graph_feat['mol_id']]
                for edge in edges:
                    for k, v in edge.items():
                        if k not in ['mol_id', 'edge_id', 'graph_type']:
                            combined_feat[f"edge_{k}"] = v
                combined_features.append(combined_feat)
            if combined_features:
                if args.format == 'csv':
                    pd.DataFrame(combined_features).to_csv(output_dir / 'ml_features.csv', index=False)
                    self.logger.info(f"\tSaved {len(combined_features)} combined features to ml_features.csv")
                else:
                    pd.DataFrame(combined_features).to_json(output_dir / 'ml_features.json', orient='records', indent=2)
                    self.logger.info(f"\tSaved {len(combined_features)} combined features to ml_features.json")
                return 0
            else:
                self.logger.warning("No features were extracted to export")
                return 0

        except Exception as e:
            self.logger.warning(f"Failed to create combined features file: {e}")
            return 1
    
    def add_labels(self, args):
        """Add target labels from CSV file or manual input to database."""
        if not Path(args.database).exists():
            self.logger.error(f"Database not found: {args.database}")
            return 1
        
        try:
            conn = sqlite3.connect(args.database)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS labels (
                    graph_id TEXT,
                    label_name TEXT,
                    label_value REAL,
                    label_type TEXT,
                    PRIMARY KEY (graph_id, label_name),
                    FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
                )
            ''')
            
            if args.graph_id:
                if args.value is None:
                    self.logger.error("--value is required when using manual input mode (--graph-id)")
                    return 1
                
                cursor.execute("SELECT COUNT(*) FROM graphs WHERE graph_id = ?", (args.graph_id,))
                if cursor.fetchone()[0] == 0:
                    self.logger.error(f"Graph '{args.graph_id}' not found in database")
                    return 1
                
                cursor.execute('''
                    INSERT OR REPLACE INTO labels (graph_id, label_name, label_value, label_type)
                    VALUES (?, ?, ?, ?)
                ''', (args.graph_id, args.label_name, args.value, args.label_type))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"\tSuccessfully added label for graph '{args.graph_id}'")
                self.logger.info(f"\tLabel name: '{args.label_name}' = {args.value}")
                self.logger.info(f"\tLabel type: '{args.label_type}'")
                
                return 0
            
            elif args.csv_file:
                if not Path(args.csv_file).exists():
                    self.logger.error(f"CSV file not found: {args.csv_file}")
                    return 1
                
                df = pd.read_csv(args.csv_file)
                
                if args.id_column not in df.columns:
                    self.logger.error(f"ID column '{args.id_column}' not found in CSV. Available columns: {list(df.columns)}")
                    return 1
                
                if args.label_column not in df.columns:
                    self.logger.error(f"Label column '{args.label_column}' not found in CSV. Available columns: {list(df.columns)}")
                    return 1
                
                added_count = 0
                error_count = 0
                
                for _, row in df.iterrows():
                    graph_id = row[args.id_column]
                    target_value = row[args.label_column]
                    
                    try:
                        target_value = float(target_value)
                        
                        cursor.execute("SELECT COUNT(*) FROM graphs WHERE graph_id = ?", (graph_id,))
                        if cursor.fetchone()[0] == 0:
                            self.logger.warning(f"Graph '{graph_id}' not found in database, skipping")
                            error_count += 1
                            continue
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO labels (graph_id, label_name, label_value, label_type)
                            VALUES (?, ?, ?, ?)
                        ''', (graph_id, args.label_name, target_value, args.label_type))
                        
                        added_count += 1
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Invalid target value for {graph_id}: {target_value} ({e})")
                        error_count += 1
                        continue
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"\tSuccessfully added {added_count} labels to database")
                if error_count > 0:
                    self.logger.warning(f"\tSkipped {error_count} entries due to errors")
                
                self.logger.info(f"\tLabel name: '{args.label_name}'")
                self.logger.info(f"\tLabel type: '{args.label_type}'")
                
                return 0
            
            else:
                self.logger.error("Either --csv-file or --graph-id must be specified")
                return 1
            
        except Exception as e:
            self.logger.error(f"Failed to add labels: {e}")
            return 1
    
    def show_stats(self, args):
        """Show database statistics."""
        if not Path(args.database).exists():
            self.logger.error(f"Database not found: {args.database}")
            return 1
        
        try:
            conn = sqlite3.connect(args.database)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM graphs")
            total_molecules = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM nodes")
            total_nodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM edges")
            total_edges = cursor.fetchone()[0]
            
            cursor.execute("SELECT graph_type, COUNT(*) FROM graphs GROUP BY graph_type")
            by_type = cursor.fetchall()
            
            print(f"\n Database Statistics")
            print(f"{'-'*50}")
            print(f"Database: {args.database}")
            print(f"Total molecules: {total_molecules}")
            print(f"Total nodes: {total_nodes}")
            print(f"Total edges: {total_edges}")
            
            if total_molecules > 0:
                avg_nodes = total_nodes / total_molecules
                avg_edges = total_edges / total_molecules
                print(f"Average nodes per molecule: {avg_nodes:.1f}")
                print(f"Average edges per molecule: {avg_edges:.1f}")
            
            if by_type:
                print(f"\nBy graph type:")
                for graph_type, count in by_type:
                    print(f"  {graph_type}: {count}")
            
            if args.detailed:
                print(f"\n Detailed Analysis:")
                
                cursor.execute("SELECT SUM(num_bonds) FROM graphs")
                expected_bonds = cursor.fetchone()[0] or 0
                if expected_bonds != total_edges:
                    print(f"   Bond count mismatch:")
                    print(f"   Expected bonds (from graphs): {expected_bonds}")
                    print(f"   Actual edges in database: {total_edges}")
                
                cursor.execute("SELECT num_atoms, COUNT(*) FROM graphs GROUP BY num_atoms ORDER BY num_atoms")
                size_dist = cursor.fetchall()
                if size_dist:
                    print(f"\n Molecule size distribution:")
                    for size, count in size_dist:
                        print(f"   {size} atoms: {count} molecules")
                
                cursor.execute("SELECT COUNT(*) FROM targets")
                total_targets = cursor.fetchone()[0]
                print(f"\n Additional data:")
                print(f"   Target features: {total_targets}")
                
                cursor.execute("SELECT graph_id, graph_type, num_atoms, num_bonds, created_timestamp FROM graphs ORDER BY created_timestamp DESC LIMIT 10")
                recent = cursor.fetchall()
                
                if recent:
                    print(f"\n Recent molecules:")
                    for graph_id, graph_type, num_atoms, num_bonds, created_at in recent:
                        print(f"   {graph_id} ({graph_type}): {num_atoms} atoms, {num_bonds} bonds - {created_at}")
            else:
                cursor.execute("SELECT SUM(num_bonds) FROM graphs")
                expected_bonds = cursor.fetchone()[0] or 0
                if expected_bonds != total_edges and total_molecules > 0:
                    print(f"\nNote: Expected {expected_bonds} bonds but found {total_edges} edges in database")
                    print("   Use --detailed flag for more analysis")
            
            conn.close()
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to show statistics: {e}")
            return 1

    def delete_data(self, args):
        """Handle delete operations."""
        if not args.delete_command:
            self.logger.error("No delete operation specified. Use --help for options.")
            return 1
        
        try:
            if args.delete_command == 'graph':
                return self._delete_single_graph(args)
            elif args.delete_command == 'by-criteria':
                return self._delete_by_criteria(args)
            elif args.delete_command == 'all':
                return self._clear_database(args)
            else:
                self.logger.error(f"Unknown delete operation: {args.delete_command}")
                return 1
                
        except Exception as e:
            self.logger.error(f"Delete operation failed: {e}")
            return 1
    
    def _delete_single_graph(self, args):
        """Delete a single graph by ID."""
        from .graph import MolecularGraph
        
        # Confirmation prompt
        if not args.confirm:
            response = input(f"Are you sure you want to delete graph '{args.graph_id}'? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                self.logger.info("Deletion cancelled")
                return 0
        
        # Delete the graph
        success = MolecularGraph.delete_graph(args.graph_id, args.database)
        
        if success:
            self.logger.info(f"Graph '{args.graph_id}' deleted successfully")
            return 0
        else:
            self.logger.error(f"Graph '{args.graph_id}' not found")
            return 1
    
    def _delete_by_criteria(self, args):
        """Delete graphs matching criteria."""
        from .graph import MolecularGraph
        
        # Parse parameters
        params = None
        if args.params:
            # Try to convert params to appropriate types
            params = []
            for param in args.params:
                try:
                    params.append(int(param))
                except ValueError:
                    try:
                        params.append(float(param))
                    except ValueError:
                        # Keep as string
                        params.append(param)
            params = tuple(params)
        
        if not args.confirm:
            try:
                conn = sqlite3.connect(args.database)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                count_query = f"SELECT COUNT(*) as count FROM graphs WHERE {args.where}"
                
                if params:
                    cursor.execute(count_query, params)
                else:
                    cursor.execute(count_query)
                    
                count_result = cursor.fetchone()
                total_count = count_result['count']
                
                if total_count == 0:
                    self.logger.info("No graphs match the specified criteria")
                    conn.close()
                    return 0
                
                # Get sample of graphs to show
                preview_query = f"SELECT graph_id FROM graphs WHERE {args.where} LIMIT 5"
                
                if params:
                    cursor.execute(preview_query, params)
                else:
                    cursor.execute(preview_query)
                    
                preview_results = cursor.fetchall()
                conn.close()
                
                print(f"Found {total_count} graphs matching criteria:")
                for row in preview_results:
                    print(f"  - {row['graph_id']}")
                if total_count > 5:
                    print(f"  ... and {total_count - 5} more")
                
                response = input(f"Delete these graphs? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    self.logger.info("Deletion cancelled")
                    return 0
                    
            except Exception as e:
                self.logger.warning(f"Could not preview graphs to delete: {e}")
                response = input(f"Continue with deletion anyway? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    self.logger.info("Deletion cancelled")
                    return 0
        
        # Delete the graphs
        deleted_count = MolecularGraph.delete_graphs_by_criteria(
            args.database, 
            args.where, 
            params
        )
        
        self.logger.info(f"Successfully deleted {deleted_count} graphs")
        return 0
    
    def _clear_database(self, args):
        """Clear entire database."""
        from .graph import MolecularGraph
        
        # Confirmation prompt
        if not args.confirm:
            response = input("Are you sure you want to delete ALL data from the database? This cannot be undone! [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                self.logger.info("Database clear cancelled")
                return 0
        
        # Clear the database
        deleted_count = MolecularGraph.clear_database(args.database, confirm=True)
        
        self.logger.info(f"Database cleared successfully. Removed {deleted_count} total records.")
        return 0

    def filter_data(self, args):
        """Handle filter operations."""
        if not args.filter_command:
            self.logger.error("No filter operation specified. Use --help for options.")
            return 1
        
        try:
            if args.filter_command == 'by-criteria':
                return self._filter_by_criteria(args)
            else:
                self.logger.error(f"Unknown filter operation: {args.filter_command}")
                return 1
                
        except Exception as e:
            self.logger.error(f"Filter operation failed: {e}")
            return 1
    
    def _filter_by_criteria(self, args):
        """Filter graphs matching criteria."""
        
        params = None
        if args.params:
            params = []
            for param in args.params:
                try:
                    params.append(int(param))
                except ValueError:
                    try:
                        params.append(float(param))
                    except ValueError:
                        # Keep as string
                        params.append(param)
            params = tuple(params)
        
        try:
            conn = sqlite3.connect(args.database)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            base_query = """
            SELECT g.graph_id, g.graph_type, g.num_atoms, g.charge,
                   g.smiles, g.formula, g.molecular_mass, g.num_electrons, g.num_bonds,
                   COUNT(DISTINCT n.node_id) as node_count,
                   COUNT(DISTINCT e.edge_id) as edge_count
            FROM graphs g
            LEFT JOIN nodes n ON g.graph_id = n.graph_id
            LEFT JOIN edges e ON g.graph_id = e.graph_id
            WHERE {where_clause}
            GROUP BY g.graph_id
            """.format(where_clause=args.where)
            
            # Execute query
            if params:
                cursor.execute(base_query, params)
            else:
                cursor.execute(base_query)
                
            results = cursor.fetchall()
            conn.close()
            
            # Display results based on show option
            if not results:
                self.logger.info("No graphs match the specified criteria")
                return 0
            
            df = pd.DataFrame([dict(row) for row in results])
            
            if args.show == 'summary':
                self.logger.info(f"Found {len(results)} graphs matching criteria")
                
            elif args.show == 'ids':
                if args.output:
                    output_path = Path(args.output)
                    if args.format == 'csv':
                        df[['graph_id']].to_csv(output_path, index=False)
                        self.logger.info(f"Graph IDs saved to {output_path}")
                    elif args.format == 'json':
                        df[['graph_id']].to_json(output_path, orient='records', indent=2)
                        self.logger.info(f"Graph IDs saved to {output_path}")
                else:
                    if args.format == 'csv':
                        print(df[['graph_id']].to_csv(index=False))
                    elif args.format == 'json':
                        print(df[['graph_id']].to_json(orient='records', indent=2))
                    else:
                        print("Graph IDs:")
                        for _, row in df.iterrows():
                            print(f"  {row['graph_id']}")
                    
            elif args.show == 'details':
                if args.output:
                    output_path = Path(args.output)
                    if args.format == 'csv':
                        df.to_csv(output_path, index=False)
                        self.logger.info(f"Detailed results saved to {output_path}")
                    elif args.format == 'json':
                        df.to_json(output_path, orient='records', indent=2)
                        self.logger.info(f"Detailed results saved to {output_path}")
                else:
                    if args.format == 'csv':
                        print(df.to_csv(index=False))
                    elif args.format == 'json':
                        print(df.to_json(orient='records', indent=2))
                    else:
                        print(f"Found {len(results)} graphs matching criteria:")
                        print()
                        
                        headers = ["Graph ID", "Type", "Formula", "Atoms", "Electrons", "Bonds", "Charge", "Mass", "Nodes", "Edges", "SMILES"]
                        col_widths = [10, 6, 8, 6, 9, 6, 7, 8, 6, 6, 25]
                        
                        header_line = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
                        separator_line = "-+-".join("-" * width for width in col_widths)
                        
                        print(header_line)
                        print(separator_line)
                        
                        for _, row in df.iterrows():
                            values = [
                                str(row['graph_id'])[:10],  # Truncate if too long
                                str(row['graph_type'])[:6],
                                str(row['formula'] or 'N/A')[:8],
                                str(row['num_atoms']),
                                str(row['num_electrons']),
                                str(row['num_bonds']),
                                f"{row['charge']:.1f}",
                                f"{row['molecular_mass']:.1f}" if row['molecular_mass'] else "N/A",
                                str(row['node_count']),
                                str(row['edge_count']),
                                str(row['smiles'] or 'N/A')[:25]  # Truncate SMILES if too long
                            ]
                            
                            data_line = " | ".join(f"{value:<{width}}" for value, width in zip(values, col_widths))
                            print(data_line)
            
            return 0
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"Error executing filter: {e}")
            return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='graphpancake',
        description='graphpancake v. 1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create database command
    create_parser = subparsers.add_parser('create-db', help='Create a new database')
    create_parser.add_argument('database', type=str, help='Database file path')
    create_parser.add_argument('--graph-types', nargs='+', choices=['DFT', 'NPA', 'NBO', 'QM'], help='Supported graph types')
    create_parser.add_argument('--force', action='store_true', help='Overwrite existing database')
    
    # Load data commands
    load_parser = subparsers.add_parser('load-data', help='Load QM data for a single molecule')
    load_parser.add_argument('--database', '-d', required=True, type=str, help='Database file path')
    load_parser.add_argument('--mol-id', '-id', required=True, type=str, help='Name or identifier')
    load_parser.add_argument('--xyz-file', '-xyz', required=True, type=str, help='.xyz coordinate file')
    load_parser.add_argument('--shermo-output', '-shermo', required=True, type=str, help='Shermo output file')
    load_parser.add_argument('--smiles', '-smiles', required=True, type=str, help='SMILES string')
    load_parser.add_argument('--janpa-output', '-janpa', type=str, help='NPA output file')
    load_parser.add_argument('--nbo-output', '-nbo', type=str, help='NBO output file')
    load_parser.add_argument('--graph-type', '-graph', default='QM', choices=['DFT', 'NPA', 'NBO', 'QM'],
                           help='Graph type')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query and filter the database')
    query_parser.add_argument('--database', '-d', required=True, type=str, help='Database file path')
    query_parser.add_argument('--graph-type', type=str, choices=['DFT', 'NPA', 'NBO', 'QM'],
                            help='Filter by graph type')
    query_parser.add_argument('--limit', type=int, help='Maximum number of results')
    query_parser.add_argument('--output', '-o', type=str, help='Output file path')
    query_parser.add_argument('--format', choices=['table', 'json', 'csv'], default='table',
                            help='Output format')
    
    # Export ML command
    export_parser = subparsers.add_parser('export-ml', help='Export ML-ready features')
    export_parser.add_argument('--database', '-d', required=True, type=str, help='Database file path')
    export_parser.add_argument('--output', '-o', required=True, type=str, help='Output directory')
    export_parser.add_argument('--graph-type', type=str, choices=['DFT', 'NPA', 'NBO', 'QM'],
                             help='Filter by graph type')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                             help='Output format')
    
    # Add labels command
    labels_parser = subparsers.add_parser('add-labels', help='Add target labels from CSV file or manual input')
    labels_parser.add_argument('--database', '-d', required=True, type=str, help='Database file path')
    
    # Make it a mutually exclusive group - either CSV file OR manual input
    label_input = labels_parser.add_mutually_exclusive_group(required=True)
    label_input.add_argument('--csv-file', type=str, help='CSV file containing labels')
    label_input.add_argument('--graph-id', type=str, help='Graph ID for manual label input')
    
    # CSV-specific options
    labels_parser.add_argument('--id-column', default='graph_id', type=str, help='Column name for graph IDs (CSV mode only)')
    labels_parser.add_argument('--label-column', default='label', type=str, help='Column name for label values (CSV mode only)')
    
    # Manual input option
    labels_parser.add_argument('--value', type=float, help='Target value for manual input mode')

    # Common options for both modes
    labels_parser.add_argument('--label-name', default='label', type=str, help='Name for this label type (CSV or manual)')
    labels_parser.add_argument('--label-type', default='regression', choices=['regression', 'classification'],
                             help='Label type: regression or classification (CSV or manual)')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument('database', type=str, help='Database file path')
    stats_parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    
    # Delete commands
    delete_parser = subparsers.add_parser('delete', help='Delete graphs from database')
    delete_subparsers = delete_parser.add_subparsers(dest='delete_command', help='Delete operations')
    
    # Delete by graph ID
    delete_graph_parser = delete_subparsers.add_parser('graph', help='Delete a specific graph by ID')
    delete_graph_parser.add_argument('--database', '-d', required=True, type=str, help='Database file path')
    delete_graph_parser.add_argument('--graph-id', '-id', required=True, type=str, help='Graph ID to delete')
    delete_graph_parser.add_argument('--confirm', action='store_true', help='Confirm deletion without prompt')
    
    # Delete by criteria
    delete_criteria_parser = delete_subparsers.add_parser('by-criteria', help='Delete graphs matching criteria')
    delete_criteria_parser.add_argument('--database', '-d', required=True, type=str, help='Database file path')
    delete_criteria_parser.add_argument('--where', required=True, type=str, help='SQL WHERE clause (without WHERE keyword)')
    delete_criteria_parser.add_argument('--params', nargs='*', help='Parameters for WHERE clause')
    delete_criteria_parser.add_argument('--confirm', action='store_true', help='Confirm deletion without prompt')
    
    # Clear database
    delete_all_parser = delete_subparsers.add_parser('all', help='Clear entire database')
    delete_all_parser.add_argument('--database', '-d', required=True, type=str, help='Database file path')
    delete_all_parser.add_argument('--confirm', action='store_true', help='Confirm deletion without prompt')
    
    # Filter commands
    filter_parser = subparsers.add_parser('filter', help='Filter and query graphs from database')
    filter_subparsers = filter_parser.add_subparsers(dest='filter_command', help='Filter operations')
    
    # Filter by criteria
    filter_criteria_parser = filter_subparsers.add_parser('by-criteria', help='Filter graphs matching criteria')
    filter_criteria_parser.add_argument('--database', '-d', required=True, type=str, help='Database file path')
    filter_criteria_parser.add_argument('--where', required=True, type=str, help='SQL WHERE clause (without WHERE keyword)')
    filter_criteria_parser.add_argument('--params', nargs='*', help='Parameters for WHERE clause')
    filter_criteria_parser.add_argument('--show', choices=['summary', 'details', 'ids'], default='summary', 
                                       help='What to show: summary (count), details (full info), or ids (graph IDs only)')
    filter_criteria_parser.add_argument('--format', choices=['table', 'csv', 'json'], default='table',
                                       help='Output format')
    filter_criteria_parser.add_argument('--output', type=str, help='Save results to file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize CLI handler
    cli = graphpancakeCLI()
    
    # Print logo
    print_logo()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    cli.setup_logging(log_level, args.log_file)
    
    # Execute command
    try:
        if args.command == 'create-db':
            return cli.create_database(args)
        elif args.command == 'load-data':
            return cli.load_data(args)
        elif args.command == 'query':
            return cli.query_database(args)
        elif args.command == 'export-ml':
            return cli.export_ml_features(args)
        elif args.command == 'add-labels':
            return cli.add_labels(args)
        elif args.command == 'stats':
            return cli.show_stats(args)
        elif args.command == 'delete':
            return cli.delete_data(args)
        elif args.command == 'filter':
            return cli.filter_data(args)
        else:
            cli.logger.error(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        cli.logger.info("\tOperation cancelled by user")
        return 1
    except Exception as e:
        cli.logger.error(f"\tUnexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())