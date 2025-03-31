"""
Vector Storage Module

This module provides vector storage capabilities using ChromaDB,
with both in-memory and persistent storage options based on data size.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger("RepoRadar.storage")

class VectorStorage:
    """
    Stores and retrieves vector embeddings for GitHub repositories
    using ChromaDB for efficient similarity search.
    """
    
    def __init__(self, device_config: Dict[str, str], 
                 data_dir: str = "./data/vector_db",
                 memory_threshold: int = 10000):
        """
        Initialize the vector storage with appropriate settings based on device config.
        
        Args:
            device_config: Hardware configuration dictionary
            data_dir: Directory to store persistent vector data
            memory_threshold: Number of items above which to use persistent storage
        """
        self.device_config = device_config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_threshold = memory_threshold
        self.client = None
        self.collection = None
        
        # Initialize the ChromaDB client and collection
        self._init_storage()
    
    def _init_storage(self):
        """Initialize the ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Check stored count to determine storage type
            stored_count = self._get_stored_count()
            
            if stored_count > self.memory_threshold:
                logger.info(f"Using persistent storage with {stored_count} items")
                self.client = chromadb.PersistentClient(path=str(self.data_dir))
            else:
                logger.info(f"Using in-memory storage with {stored_count} items")
                self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            
            # Create or get the collection
            self.collection = self.client.get_or_create_collection(
                name="github_repos",
                metadata={"description": "GitHub trending repositories"}
            )
            
            logger.info(f"Initialized ChromaDB storage with {self.collection.count()} items")
            
        except ImportError:
            logger.error("Failed to import chromadb. Please install with: pip install chromadb>=0.4.0")
            raise
    
    def _get_stored_count(self) -> int:
        """Get the count of stored items from metadata file"""
        try:
            with open(self.data_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
                return metadata.get("count", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0
    
    def _update_stored_count(self, count: int):
        """Update the count of stored items in metadata file"""
        try:
            metadata = {"count": count, "model": self.device_config["model_size"]}
            with open(self.data_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
    
    def has_data(self) -> bool:
        """Check if the storage has any data"""
        if not self.collection:
            return False
        return self.collection.count() > 0
    
    def store_vectors(self, repos_with_vectors: List[Dict[str, Any]], 
                      original_data: Optional[List[Dict[str, Any]]] = None):
        """
        Store repository vectors in the ChromaDB collection.
        
        Args:
            repos_with_vectors: List of repositories with embedding vectors
            original_data: Optional original data if different from repos_with_vectors
        """
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return
        
        data_to_store = repos_with_vectors if original_data is None else original_data
        
        # Prepare data for batch insertion
        ids = []
        embeddings = []
        metadatas = []
        
        for i, repo in enumerate(repos_with_vectors):
            if "embedding" not in repo:
                logger.warning(f"Repository {repo.get('name', f'at index {i}')} has no embedding, skipping")
                continue
            
            # Use repo owner/name as ID or fallback to index
            repo_id = f"{repo.get('owner', 'unknown')}_{repo.get('name', f'repo{i}')}"
            
            # Get the embedding vector
            embedding = repo["embedding"]
            
            # Prepare metadata (everything except the embedding)
            metadata = {k: v for k, v in repo.items() if k != "embedding" and not isinstance(v, (list, dict))}
            
            # Add to batch lists
            ids.append(repo_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
        
        if not ids:
            logger.warning("No valid repositories with embeddings to store")
            return
        
        # Store in ChromaDB
        try:
            logger.info(f"Storing {len(ids)} repository vectors in ChromaDB")
            
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
            
            # Update metadata
            self._update_stored_count(self.collection.count())
            
        except Exception as e:
            logger.error(f"Error storing vectors in ChromaDB: {e}")
            raise
    
    def search_similar(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for repositories similar to the query vector.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of similar repositories with similarity scores
        """
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            
            if results and "ids" in results:
                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                
                for i, repo_id in enumerate(ids):
                    distance = distances[i] if distances else None
                    metadata = metadatas[i] if metadatas else {}
                    
                    # Convert distance to similarity score (1 - distance for cosine distance)
                    similarity = 1 - distance if distance is not None else None
                    
                    result = {
                        "id": repo_id,
                        "similarity_score": similarity,
                        **metadata
                    }
                    
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {e}")
            return []
    
    def get_all_repos(self) -> List[Dict[str, Any]]:
        """
        Get all stored repositories.
        
        Returns:
            List of all repositories in storage
        """
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return []
        
        try:
            # Get all items in the collection
            results = self.collection.get()
            
            formatted_results = []
            
            if results and "ids" in results:
                ids = results["ids"]
                metadatas = results.get("metadatas", [])
                
                for i, repo_id in enumerate(ids):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    result = {
                        "id": repo_id,
                        **metadata
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting all repos from ChromaDB: {e}")
            return []
    
    def add_tag(self, repo_id: str, tag: str):
        """
        Add a tag to a repository.
        
        Args:
            repo_id: Repository ID
            tag: Tag to add
        """
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return
        
        try:
            # Get current metadata
            results = self.collection.get(ids=[repo_id])
            
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                
                # Add tag to tags list
                tags = metadata.get("tags", "").split(",")
                tags = [t.strip() for t in tags if t.strip()]
                
                if tag not in tags:
                    tags.append(tag)
                
                # Update metadata
                metadata["tags"] = ",".join(tags)
                
                # Update in ChromaDB
                self.collection.update(
                    ids=[repo_id],
                    metadatas=[metadata]
                )
                
                logger.info(f"Added tag '{tag}' to repository {repo_id}")
            
        except Exception as e:
            logger.error(f"Error adding tag to repository: {e}")
    
    def backup(self, backup_dir: str = "./backups"):
        """
        Backup the vector database.
        
        Args:
            backup_dir: Directory to store backups
        """
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if not self.collection:
                logger.error("ChromaDB collection not initialized")
                return
            
            # Get all data
            all_data = self.collection.get()
            
            # Save as JSON
            backup_file = backup_path / f"github_repos_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, "w") as f:
                json.dump(all_data, f)
            
            logger.info(f"Backed up {self.collection.count()} items to {backup_file}")
            
        except Exception as e:
            logger.error(f"Error backing up data: {e}")
            
    def restore(self, backup_file: str):
        """
        Restore the vector database from a backup.
        
        Args:
            backup_file: Path to the backup file
        """
        try:
            # Load backup data
            with open(backup_file, "r") as f:
                backup_data = json.load(f)
            
            if not self.collection:
                logger.error("ChromaDB collection not initialized")
                return
            
            # Clear existing data
            self.collection.delete(where={})
            
            # Restore data
            if "ids" in backup_data and backup_data["ids"]:
                ids = backup_data["ids"]
                embeddings = backup_data.get("embeddings", [])
                metadatas = backup_data.get("metadatas", [])
                
                # Add in batches
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_embeddings = embeddings[i:i+batch_size] if embeddings else None
                    batch_metadatas = metadatas[i:i+batch_size] if metadatas else None
                    
                    add_kwargs = {"ids": batch_ids}
                    if batch_embeddings:
                        add_kwargs["embeddings"] = batch_embeddings
                    if batch_metadatas:
                        add_kwargs["metadatas"] = batch_metadatas
                    
                    self.collection.add(**add_kwargs)
                    
                    logger.info(f"Restored batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
                
                # Update metadata
                self._update_stored_count(self.collection.count())
                
                logger.info(f"Restored {self.collection.count()} items from backup")
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")

# Add import for datetime if backup method is used
from datetime import datetime 