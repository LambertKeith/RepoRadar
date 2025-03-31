"""
GitHub Trends Data Collector

This module is responsible for fetching trending repository data from GitHub
and caching it locally for efficient processing.
"""

import requests
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Optional, Any
import os
from reporadar.glm_api import GLMFlashAPI

logger = logging.getLogger("RepoRadar.collector")

class GithubTrendsCollector:
    """
    Collects trending repositories data from GitHub and manages local cache.
    """
    
    GITHUB_TRENDS_URL = "https://github.com/trending"
    DEFAULT_LANGUAGES = ["", "python", "javascript", "typescript", "go", "rust", "java", "c++", "c#"]
    DEFAULT_TIME_PERIODS = ["daily", "weekly", "monthly"]
    
    def __init__(self, cache_dir: str = "./data/cache", update_interval: int = 6,
                 languages: Optional[List[str]] = None, time_periods: Optional[List[str]] = None,
                 max_repos_per_language: int = 25, min_stars: int = 10):
        """
        Initialize the collector with cache directory and update interval.
        
        Args:
            cache_dir: Directory to store cached repository data
            update_interval: Hours between updates
            languages: List of languages to collect (or None for defaults)
            time_periods: List of time periods to collect (or None for defaults)
            max_repos_per_language: Maximum number of repos to collect per language
            min_stars: Minimum stars required to consider a repository
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.update_interval = update_interval
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.time_periods = time_periods or self.DEFAULT_TIME_PERIODS
        self.max_repos_per_language = max_repos_per_language
        self.min_stars = min_stars
        self.last_run_file = self.cache_dir / "last_run.txt"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 RepoRadar/0.1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml"
        })
        
        # 初始化 GLMFlashAPI
        self.api_key = os.getenv("GLM_FLASH_KEY") or "your-api-key-here"  # 从环境变量或配置中获取
        self.glm_api = GLMFlashAPI(self.api_key)
        
        logger.info(f"Initialized collector with {len(self.languages)} languages and {len(self.time_periods)} time periods")
    
    def should_run(self) -> bool:
        """检查当前时间是否在凌晨3:00到4:00之间，并且一天只能操作一次"""
        now = datetime.now()
        if now.hour == 2:  # 只在3点时检查
            if self.last_run_file.exists():
                with open(self.last_run_file, "r") as f:
                    last_run = datetime.fromisoformat(f.read().strip())
                if (now - last_run).days < 1:  # 检查是否已经运行过
                    return False
            return True
        return False

    def record_run_time(self):
        """记录当前运行时间"""
        with open(self.last_run_file, "w") as f:
            f.write(datetime.now().isoformat())

    def should_update(self) -> bool:
        """Check if the data should be updated based on last update time"""
        last_update_file = self.cache_dir / "last_update.txt"
        
        if not last_update_file.exists():
            return True
            
        with open(last_update_file, "r") as f:
            try:
                last_update = datetime.fromisoformat(f.read().strip())
                return datetime.now() > last_update + timedelta(hours=self.update_interval)
            except (ValueError, IOError):
                return True
    
    def _record_update_time(self):
        """Record the current time as the last update time"""
        with open(self.cache_dir / "last_update.txt", "w") as f:
            f.write(datetime.now().isoformat())
    
    def _fetch_trending_page(self, language: str = "", time_period: str = "daily") -> Optional[str]:
        """
        Fetch HTML content from GitHub trending page for a specific language and time period.
        
        Args:
            language: Programming language to filter by (empty for all languages)
            time_period: Time period for trending repos (daily, weekly, monthly)
            
        Returns:
            HTML content of the page or None if the request failed
        """
        url = self.GITHUB_TRENDS_URL
        if language:
            url += f"/{language}"
        
        params = {"since": time_period}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch trending page for {language} ({time_period}): {e}")
            return None
    
    def _parse_trending_html(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Parse the GitHub trending page HTML to extract repository information.
        
        Args:
            html_content: HTML content of the GitHub trending page
            
        Returns:
            List of dictionaries containing repository information
        """
        logger.info("Parsing HTML content for trending repositories")
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            repos = []
            
            # Find repository articles
            for article in soup.select('article.Box-row'):
                repo = {}
                
                # Get repo owner and name
                repo_header = article.select_one('h2.h3.lh-condensed')
                if repo_header:
                    a_tag = repo_header.select_one('a')
                    if a_tag:
                        full_name = a_tag.get('href', '').strip('/')
                        if '/' in full_name:
                            owner, name = full_name.split('/', 1)
                            repo['owner'] = owner
                            repo['name'] = name
                
                # Get description
                desc_p = article.select_one('p')
                if desc_p:
                    repo['description'] = desc_p.text.strip()
                
                # Get language
                lang_span = article.select_one('span[itemprop="programmingLanguage"]')
                if lang_span:
                    repo['language'] = lang_span.text.strip()
                
                # Get stars
                stars_link = article.select_one('a[href*="stargazers"]')
                if stars_link:
                    stars = int(stars_link.text.strip().replace(',', ''))
                    repo['stars'] = stars
                    
                    # 过滤掉低星标的项目
                    if stars < self.min_stars:
                        continue
                
                # Get forks
                forks_link = article.select_one('a[href*="network/members"]')
                if forks_link:
                    repo['forks'] = int(forks_link.text.strip().replace(',', ''))
                
                # 只添加有效的项目
                if 'owner' in repo and 'name' in repo and repo.get('description'):
                    repos.append(repo)
                    
                    # 限制最大项目数量
                    if len(repos) >= self.max_repos_per_language:
                        break
            
            return repos
            
        except ImportError:
            logger.warning("BeautifulSoup not installed, using mock data for demo")
            return []
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return []
    
    def _get_repo_details(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Fetch additional details for a repository using the GitHub API.
        
        Args:
            owner: Repository owner/username
            repo: Repository name
            
        Returns:
            Dictionary with repository details
        """
        url = f"https://api.github.com/repos/{owner}/{repo}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch details for {owner}/{repo}: {e}")
            return {}
    
    def search_with_glm(self, query: str) -> str:
        """使用 glm-4-flash API 进行内容检索"""
        return self.glm_api.query(query)

    def collect_trends(self) -> List[Dict[str, Any]]:
        """
        Collect trending repositories from GitHub across languages and time periods.
        
        Returns:
            List of dictionaries with repository information
        """
        if self.should_run():
            self.record_run_time()
            all_repos = []
            repo_ids = set()  # To track unique repositories
            
            # Use configured languages and time periods
            combinations = [(lang, period) for lang in self.languages for period in self.time_periods]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_params = {
                    executor.submit(self._fetch_trending_page, lang, period): (lang, period)
                    for lang, period in combinations
                }
                
                for future in concurrent.futures.as_completed(future_to_params):
                    lang, period = future_to_params[future]
                    try:
                        html_content = future.result()
                        if html_content:
                            repos = self._parse_trending_html(html_content)
                            
                            for repo in repos:
                                repo_id = f"{repo.get('owner')}/{repo.get('name')}"
                                if repo_id not in repo_ids:
                                    repo_ids.add(repo_id)
                                    
                                    # Add language and period info
                                    repo["trending_language"] = lang if lang else "all"
                                    repo["trending_period"] = period
                                    
                                    # Fetch additional details if needed
                                    if "description" not in repo or not repo["description"]:
                                        details = self._get_repo_details(repo.get("owner"), repo.get("name"))
                                        repo.update({
                                            "description": details.get("description", ""),
                                            "stars": details.get("stargazers_count", 0),
                                            "forks": details.get("forks_count", 0),
                                            "open_issues": details.get("open_issues_count", 0),
                                            "topics": details.get("topics", [])
                                        })
                                    
                                    all_repos.append(repo)
                                    
                                    # Save individual repo data to cache
                                    self._save_repo_to_cache(repo)
                                    
                    except Exception as e:
                        logger.error(f"Error processing {lang} ({period}): {e}")
            
            # 在收集到的项目中，使用 glm-4-flash API 进行增强检索
            for repo in all_repos:
                prompt = f"请根据以下信息提供项目的详细介绍：{repo['description']}"
                enhanced_info = self.search_with_glm(prompt)
                repo['enhanced_info'] = enhanced_info  # 将增强的信息添加到项目中
            
            # Save the complete dataset
            self._save_all_repos_to_cache(all_repos)
            self._record_update_time()
            
            logger.info(f"Collected {len(all_repos)} unique repositories")
            return all_repos
        else:
            logger.info("当前不在允许的爬取时间内，爬虫功能被禁用。")
            return self.load_cached_data()  # 返回缓存的数据
    
    def _save_repo_to_cache(self, repo: Dict[str, Any]):
        """Save individual repository data to cache"""
        if "owner" in repo and "name" in repo:
            repo_path = self.cache_dir / f"{repo['owner']}_{repo['name']}.json"
            with open(repo_path, "w") as f:
                json.dump(repo, f, indent=2)
    
    def _save_all_repos_to_cache(self, repos: List[Dict[str, Any]]):
        """Save all repositories data to a single cache file"""
        with open(self.cache_dir / "all_repos.json", "w") as f:
            json.dump({
                "updated_at": datetime.now().isoformat(),
                "count": len(repos),
                "repositories": repos
            }, f, indent=2)
    
    def load_cached_data(self) -> List[Dict[str, Any]]:
        """Load the cached repository data"""
        try:
            with open(self.cache_dir / "all_repos.json", "r") as f:
                data = json.load(f)
                return data.get("repositories", [])
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("No valid cache found, returning empty repository list")
            return [] 