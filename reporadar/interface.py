"""
Web Interface Module

This module provides a Gradio-based web interface for interacting with
the GitHub trends analysis system.
"""

import logging
import os
import json
import webbrowser
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("RepoRadar.interface")

def launch_interface(storage, vectorizer, host: str = "127.0.0.1", port: int = 7860, 
                     open_browser: bool = True):
    """
    Launch the web interface for interacting with the system.
    
    Args:
        storage: VectorStorage instance
        vectorizer: TextVectorizer instance
        host: Host address to bind the web server
        port: Port to bind the web server
        open_browser: Whether to automatically open the web browser
    """
    try:
        import gradio as gr
        
        # Define interface functions
        def search_repos(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
            """Search for repositories based on query text"""
            logger.info(f"Searching for: {query} (top {top_k})")
            
            try:
                # Generate embedding for query
                query_embedding = vectorizer.model.encode(query, convert_to_numpy=True)
                
                # Search using the storage module
                results = storage.search_similar(query_embedding.tolist(), top_k=top_k)
                
                return results
            
            except Exception as e:
                logger.error(f"Error searching repositories: {e}")
                return []
        
        def format_search_results(results: List[Dict[str, Any]]) -> str:
            """Format search results for display"""
            if not results:
                return "No results found."
            
            output = []
            for i, repo in enumerate(results):
                similarity = repo.get("similarity_score", 0)
                similarity_pct = f"{similarity * 100:.1f}%" if similarity is not None else "N/A"
                
                owner = repo.get("owner", "Unknown")
                name = repo.get("name", "Unknown")
                
                # Format repository details
                repo_details = [
                    f"## {i+1}. {owner}/{name} - Match: {similarity_pct}",
                    f"**Description**: {repo.get('description', 'No description')}",
                    f"**Language**: {repo.get('language', 'Not specified')}",
                    f"**Stars**: {repo.get('stars', 'N/A')}",
                    f"**Topics**: {', '.join(repo.get('topics', [])) or 'None'}"
                ]
                
                # Add user tags if present
                tags = repo.get("tags", "")
                if tags:
                    repo_details.append(f"**Tags**: {tags}")
                    
                # Add GitHub link
                repo_details.append(f"**Link**: https://github.com/{owner}/{name}")
                
                output.append("\n".join(repo_details))
            
            return "\n\n".join(output)
        
        def add_tag_to_repo(repo_id: str, tag: str) -> str:
            """Add a tag to a repository"""
            if not repo_id or not tag:
                return "Please provide both repository ID and tag."
            
            try:
                storage.add_tag(repo_id, tag)
                return f"Added tag '{tag}' to repository {repo_id}"
            
            except Exception as e:
                logger.error(f"Error adding tag: {e}")
                return f"Error adding tag: {str(e)}"
        
        def get_system_info() -> str:
            """Get system information"""
            info = [
                "# RepoRadar System Information",
                f"**Hardware Mode**: {vectorizer.device} ({vectorizer.precision})",
                f"**Model**: {vectorizer.model_config['name']}",
                f"**Database Size**: {storage.collection.count() if storage.collection else 0} repositories",
                f"**Storage Type**: {'Persistent' if hasattr(storage, 'client') and str(storage.client.__class__).find('Persistent') > 0 else 'In-Memory'}"
            ]
            
            return "\n\n".join(info)
        
        # Create the interface
        with gr.Blocks(title="RepoRadar", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🔭 RepoRadar - GitHub Trends Analysis System")
            
            with gr.Tabs():
                with gr.TabItem("Search"):
                    with gr.Row():
                        with gr.Column():
                            query_input = gr.Textbox(label="Search Query", placeholder="Enter search query...")
                            top_k_input = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Results")
                            search_button = gr.Button("Search")
                        
                        results_output = gr.Markdown(label="Search Results")
                    
                    search_button.click(
                        fn=lambda q, k: format_search_results(search_repos(q, k)),
                        inputs=[query_input, top_k_input],
                        outputs=results_output
                    )
                
                with gr.TabItem("Tag Repositories"):
                    with gr.Row():
                        with gr.Column():
                            repo_id_input = gr.Textbox(label="Repository ID", placeholder="owner_name")
                            tag_input = gr.Textbox(label="Tag", placeholder="Enter tag...")
                            add_tag_button = gr.Button("Add Tag")
                        
                        tag_result_output = gr.Markdown(label="Result")
                    
                    add_tag_button.click(
                        fn=add_tag_to_repo,
                        inputs=[repo_id_input, tag_input],
                        outputs=tag_result_output
                    )
                
                with gr.TabItem("System Info"):
                    system_info_output = gr.Markdown()
                    refresh_info_button = gr.Button("Refresh Info")
                    
                    refresh_info_button.click(
                        fn=get_system_info,
                        inputs=[],
                        outputs=system_info_output
                    )
                    
                    # Set initial system info
                    system_info_output.value = get_system_info()
        
        # Launch the interface
        app.launch(server_name=host, server_port=port, share=False, inbrowser=open_browser)
        
    except ImportError:
        logger.error("Failed to import gradio. Please install with: pip install gradio>=3.0")
        print("Please install Gradio with: pip install gradio>=3.0")
        raise 