import time
import logging
import random
import requests
import arxiv
import xml.etree.ElementTree as ET
from typing import List, Optional
from functools import lru_cache
from scholarly import scholarly
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.paper import Paper
from utils.text_processor import TextProcessor
from bs4 import BeautifulSoup

# Constants
CACHE_SIZE = 128
MAX_PAPERS = 5
SCHOLAR_MAX_PAPERS = 3
ARXIV_MAX_PAPERS = 5
MAX_WORKERS = 3  # One thread per data source

class ResearchFetcher:
    def __init__(self):
        self.session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 0.34
        self._max_retries = 3
        self._setup_scholarly()
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    def __del__(self):
        """Cleanup executor on deletion"""
        self.executor.shutdown(wait=False)
    
    def _setup_scholarly(self):
        """Configure scholarly with basic settings"""
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        # Set up a random user agent for scholarly
        scholarly._get_page = lambda url: requests.get(url, headers={'User-Agent': random.choice(self.user_agents)})
    
    def _rotate_user_agent(self):
        """Rotate user agent for Google Scholar requests"""
        return random.choice(self.user_agents)

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed PubMed's rate limit"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last)
        self._last_request_time = time.time()

    def _make_request_with_retry(self, url: str, params: dict, timeout: int = 10) -> Optional[requests.Response]:
        """Make a request with retries and rate limiting"""
        for attempt in range(self._max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                    wait_time = (attempt + 1) * self._min_request_interval * 2
                    logging.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                if attempt == self._max_retries - 1:
                    logging.error(f"Error after {self._max_retries} retries: {str(e)}")
                    return None
        return None

    @lru_cache(maxsize=CACHE_SIZE)
    def fetch_arxiv_papers(self, query: str) -> List[Paper]:
        """Fetch papers from arXiv"""
        try:
            # Ensure query includes autism if not already present
            if 'autism' not in query.lower():
                search_query = f"autism {query}"
            else:
                search_query = query

            # Search arXiv
            search = arxiv.Search(
                query=search_query,
                max_results=ARXIV_MAX_PAPERS,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in search.results():
                # Create Paper object
                paper = Paper(
                    title=result.title,
                    authors=', '.join([author.name for author in result.authors]),
                    abstract=result.summary,
                    url=result.pdf_url,
                    publication_date=result.published.strftime("%Y-%m-%d"),
                    relevance_score=1.0 if 'autism' in result.title.lower() else 0.8,
                    source="arXiv"
                )
                papers.append(paper)

            return papers

        except Exception as e:
            logging.error(f"Error fetching arXiv papers: {str(e)}")
            return []

    @lru_cache(maxsize=CACHE_SIZE)
    def fetch_pubmed_papers(self, query: str) -> List[Paper]:
        """Fetch papers from PubMed"""
        try:
            # Ensure query includes autism if not already present
            if 'autism' not in query.lower():
                search_query = f"autism {query}"
            else:
                search_query = query

            # Encode the query for URL
            encoded_query = requests.utils.quote(search_query)
            
            # Search PubMed
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_query}&retmax=5"
            search_response = requests.get(search_url)
            search_tree = ET.fromstring(search_response.content)
            
            # Get IDs of papers
            id_list = search_tree.findall('.//Id')
            if not id_list:
                return []
            
            # Get details for each paper
            papers = []
            for id_elem in id_list:
                paper_id = id_elem.text
                details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={paper_id}&retmode=xml"
                details_response = requests.get(details_url)
                details_tree = ET.fromstring(details_response.content)
                
                # Extract article data
                article = details_tree.find('.//Article')
                if article is None:
                    continue
                
                # Get title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else "No title available"
                
                # Get abstract
                abstract_elem = article.find('.//Abstract/AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                
                # Get authors
                author_list = article.findall('.//Author')
                authors = []
                for author in author_list:
                    last_name = author.find('LastName')
                    fore_name = author.find('ForeName')
                    if last_name is not None and fore_name is not None:
                        authors.append(f"{fore_name.text} {last_name.text}")
                
                # Get publication date
                pub_date = article.find('.//PubDate')
                if pub_date is not None:
                    year = pub_date.find('Year')
                    month = pub_date.find('Month')
                    day = pub_date.find('Day')
                    pub_date_str = f"{year.text if year is not None else ''}-{month.text if month is not None else '01'}-{day.text if day is not None else '01'}"
                else:
                    pub_date_str = "Unknown"
                
                # Create Paper object
                paper = Paper(
                    title=title,
                    authors=', '.join(authors) if authors else "Unknown Authors",
                    abstract=abstract,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                    publication_date=pub_date_str,
                    relevance_score=1.0 if 'autism' in title.lower() else 0.8,
                    source="PubMed"
                )
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logging.error(f"Error fetching PubMed papers: {str(e)}")
            return []

    @lru_cache(maxsize=CACHE_SIZE)
    def fetch_scholar_papers(self, query: str) -> List[Paper]:
        """
        Fetch papers from Google Scholar
        """
        try:
            headers = {'User-Agent': random.choice(self.user_agents)}
            encoded_query = requests.utils.quote(query)
            url = f'https://scholar.google.com/scholar?q={encoded_query}&hl=en&as_sdt=0,5'
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logging.error(f"Google Scholar returned status code {response.status_code}")
                return []

            # Use BeautifulSoup to parse the response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            papers = []
            for result in soup.select('.gs_ri')[:5]:  # Limit to first 5 results
                title_elem = result.select_one('.gs_rt')
                authors_elem = result.select_one('.gs_a')
                snippet_elem = result.select_one('.gs_rs')
                
                if not title_elem:
                    continue
                    
                title = title_elem.get_text(strip=True)
                authors = authors_elem.get_text(strip=True) if authors_elem else "Unknown Authors"
                abstract = snippet_elem.get_text(strip=True) if snippet_elem else ""
                url = title_elem.find('a')['href'] if title_elem.find('a') else ""
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    publication_date="",  # Date not easily available
                    relevance_score=0.8,  # Default score
                    source="Google Scholar"
                )
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logging.error(f"Error fetching Google Scholar papers: {str(e)}")
            return []

    def fetch_all_papers(self, query: str) -> List[Paper]:
        """Fetch papers from all sources concurrently and combine results"""
        all_papers = []
        futures = []

        # Submit tasks to thread pool
        try:
            futures.append(self.executor.submit(self.fetch_arxiv_papers, query))
            futures.append(self.executor.submit(self.fetch_pubmed_papers, query))
            futures.append(self.executor.submit(self.fetch_scholar_papers, query))

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    papers = future.result()
                    all_papers.extend(papers)
                except Exception as e:
                    logging.error(f"Error collecting papers from source: {str(e)}")
        except Exception as e:
            logging.error(f"Error in concurrent paper fetching: {str(e)}")

        # Sort and deduplicate papers
        seen_titles = set()
        unique_papers = []
        
        for paper in sorted(all_papers, key=lambda x: x.relevance_score, reverse=True):
            title_key = paper.title.lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        return unique_papers[:MAX_PAPERS]
