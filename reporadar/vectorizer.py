"""
Text Vectorization Module

This module handles the conversion of repository text data into vector embeddings
using sentence transformer models with hardware acceleration support.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Union, Optional
from pathlib import Path

logger = logging.getLogger("RepoRadar.vectorizer")

class TextVectorizer:
    """
    Converts repository text content into vector embeddings using various models
    with hardware acceleration support.
    """
    
    # Default model configuration if not provided in config
    DEFAULT_MODEL_CONFIGS = {
        "small": {
            "name": "all-MiniLM-L6-v2",
            "dim": 384,
            "onnx_path": "./models/all-MiniLM-L6-v2-onnx/"
        },
        "medium": {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "dim": 384,
            "onnx_path": None  # Will use HuggingFace model
        },
        "large": {
            "name": "all-mpnet-base-v2",
            "dim": 768,
            "onnx_path": None  # Will use HuggingFace model
        }
    }
    
    def __init__(self, device_config: Dict[str, str]):
        """
        Initialize the vectorizer with the appropriate model based on hardware.
        
        Args:
            device_config: Hardware configuration dictionary with keys:
                           device, precision, model_size
                           Optional: model_configs (from config.yaml)
        """
        self.device = device_config["device"]
        self.precision = device_config["precision"]
        self.model_size = device_config["model_size"]
        
        # Use model configs from config.yaml if available, otherwise use defaults
        model_configs = device_config.get("model_configs", self.DEFAULT_MODEL_CONFIGS)
        self.model_config = model_configs.get(self.model_size, self.DEFAULT_MODEL_CONFIGS[self.model_size])
        
        self.model = None
        
        logger.info(f"Initializing vectorizer with {self.model_size} model ({self.model_config['name']}) on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate model based on hardware configuration"""
        try:
            # Import here to avoid unnecessary dependency loading if not used
            from sentence_transformers import SentenceTransformer
            
            model_name = self.model_config["name"]
            
            # Check if we should use ONNX quantized model (for CPU)
            if self.device == "cpu" and self.model_config.get("onnx_path"):
                onnx_path = Path(self.model_config["onnx_path"])
                
                if onnx_path.exists():
                    logger.info(f"Loading ONNX model from {onnx_path}")
                    self.model = SentenceTransformer(str(onnx_path))
                else:
                    # Download and convert to ONNX if path doesn't exist
                    logger.info(f"ONNX model not found, downloading {model_name} and converting")
                    
                    # Create directory
                    onnx_path.mkdir(parents=True, exist_ok=True)
                    
                    # First load regular model
                    self.model = SentenceTransformer(model_name)
                    
                    # Then convert to ONNX (simplified - would need actual implementation)
                    logger.info("Converting model to ONNX format")
                    try:
                        from optimum.onnxruntime import ORTModelForFeatureExtraction
                        from transformers import AutoTokenizer
                        
                        # In a real implementation, this would convert the model to ONNX
                        # For simplicity, we're just mentioning the dependency
                        logger.info("ONNX conversion would happen here in a full implementation")
                        
                    except ImportError:
                        logger.warning("ONNX optimization packages not installed, using regular model")
            else:
                # Use regular HuggingFace model with GPU if available
                logger.info(f"Loading {model_name} model")
                self.model = SentenceTransformer(model_name, device=self.device)
                
                # Set half precision if requested and on GPU
                if self.precision == "fp16" and self.device == "cuda":
                    logger.info("Using FP16 precision")
                    self.model.half()
        
        except ImportError:
            logger.error("Failed to import sentence-transformers. Please install with: pip install sentence-transformers")
            raise
    
    def _prepare_repo_text(self, repo: Dict[str, Any]) -> str:
        """
        Prepare repository text for vectorization by combining relevant fields.
        
        Args:
            repo: Repository data dictionary
            
        Returns:
            Concatenated text representation of the repository
        """
        text_parts = []
        
        if "name" in repo:
            text_parts.append(f"Repository: {repo['name']}")
        
        if "owner" in repo:
            text_parts.append(f"Owner: {repo['owner']}")
        
        if "description" in repo and repo["description"]:
            text_parts.append(f"Description: {repo['description']}")
        
        if "language" in repo and repo["language"]:
            text_parts.append(f"Language: {repo['language']}")
        
        if "topics" in repo and repo["topics"]:
            text_parts.append(f"Topics: {', '.join(repo['topics'][:10])}")
            
        # Add tags if present (user-added metadata)
        if "tags" in repo and repo["tags"]:
            text_parts.append(f"Tags: {repo['tags']}")
            
        return "\n".join(text_parts)
    
    def vectorize(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a repository dictionary into a vector representation.
        
        Args:
            repo: Repository data dictionary
            
        Returns:
            Dictionary with original data and added embedding
        """
        if not self.model:
            logger.error("Model not loaded, cannot vectorize")
            return repo
        
        text = self._prepare_repo_text(repo)
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Add embedding to repo data
            result = repo.copy()
            result["embedding"] = embedding.tolist()
            result["embedding_model"] = self.model_config["name"]
            result["embedding_dim"] = self.model_config["dim"]
            
            return result
        
        except Exception as e:
            logger.error(f"Error vectorizing repository {repo.get('name', 'unknown')}: {e}")
            return repo
    
    def vectorize_batch(self, repos: List[Dict[str, Any]], batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Convert a batch of repository dictionaries into vector representations.
        
        Args:
            repos: List of repository data dictionaries
            batch_size: Number of items to process at once
            
        Returns:
            List of dictionaries with original data and added embeddings
        """
        if not self.model:
            logger.error("Model not loaded, cannot vectorize batch")
            return repos
        
        logger.info(f"Vectorizing {len(repos)} repositories in batches of {batch_size}")
        
        result = []
        repo_texts = []
        repo_indices = []
        
        # Prepare all texts
        for i, repo in enumerate(repos):
            repo_texts.append(self._prepare_repo_text(repo))
            repo_indices.append(i)
        
        # Process in batches
        for i in range(0, len(repo_texts), batch_size):
            batch_texts = repo_texts[i:i+batch_size]
            batch_indices = repo_indices[i:i+batch_size]
            
            try:
                # Generate embeddings for batch
                embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                
                # Add embeddings to repo data
                for j, embedding in enumerate(embeddings):
                    repo_idx = batch_indices[j]
                    repo_with_embedding = repos[repo_idx].copy()
                    repo_with_embedding["embedding"] = embedding.tolist()
                    repo_with_embedding["embedding_model"] = self.model_config["name"]
                    repo_with_embedding["embedding_dim"] = self.model_config["dim"]
                    result.append(repo_with_embedding)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(repo_texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Error vectorizing batch: {e}")
                # Add original repos without embeddings
                for idx in batch_indices:
                    result.append(repos[idx])
        
        return result
    
    def search_similar(self, query: str, repos_with_embeddings: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for repositories similar to a query text.
        
        Args:
            query: Search query text
            repos_with_embeddings: List of repositories with embedding data
            top_k: Number of results to return
            
        Returns:
            List of top-k similar repositories with similarity scores
        """
        if not self.model:
            logger.error("Model not loaded, cannot search")
            return []
        
        if not repos_with_embeddings:
            logger.warning("No repositories with embeddings provided for search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Calculate similarities
            results_with_scores = []
            
            for repo in repos_with_embeddings:
                if "embedding" in repo:
                    # Convert embedding back to numpy array if needed
                    repo_embedding = np.array(repo["embedding"])
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, repo_embedding)
                    
                    results_with_scores.append((repo, similarity))
            
            # Sort by similarity (descending)
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results with scores
            results = []
            for repo, score in results_with_scores[:top_k]:
                result = repo.copy()
                result["similarity_score"] = float(score)
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching similar repositories: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity value
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 