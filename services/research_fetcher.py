import logging
import random
import requests
import arxiv
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import xml.etree.ElementTree as ET
from scholarly import scholarly
from Bio import Entrez, Medline
from models.paper import Paper
from utils.text_processor import TextProcessor
from bs4 import BeautifulSoup
import time

# Constants
CACHE_SIZE = 128
ARXIV_MAX_PAPERS = 10
PUBMED_MAX_PAPERS = 10
SCHOLAR_MAX_PAPERS = 5  # Reduced to minimize rate limiting
MAX_WORKERS = 3  # One thread per data source

class ResearchFetcher:
    def __init__(self, email: str = "your@email.com"):
        """Initialize the research fetcher"""
        self.text_processor = TextProcessor()
        Entrez.email = email
        self.session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 2.0  # Increased delay between requests
        self._setup_scholarly()
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    def __del__(self):
        self.executor.shutdown(wait=False)
    
    def _setup_scholarly(self):
        """Configure scholarly with basic settings"""
        try:
            # Set longer timeout and more retries
            scholarly.set_timeout(15)
            scholarly.set_retries(3)
        except Exception as e:
            logging.error(f"Error setting up scholarly: {str(e)}")

    def _rotate_user_agent(self):
        """Rotate user agent for Google Scholar requests"""
        return random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        ])

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed PubMed's rate limit"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last)
        self._last_request_time = time.time()

    def _make_request_with_retry(self, url: str, params: dict, timeout: int = 10) -> Optional[requests.Response]:
        """Make a request with retries and rate limiting"""
        for attempt in range(3):
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
                if attempt == 2:
                    logging.error(f"Error after 3 retries: {str(e)}")
                    return None
        return None

    def is_autism_related(self, title: str, abstract: str) -> bool:
        """Check if the paper is related to autism"""
        autism_terms = {
            'autism', 'autistic', 'asd', 'autism spectrum disorder',
            'asperger', 'neurodevelopmental', 'neurodivergent'
        }
        
        # Convert to lowercase for case-insensitive matching
        text = (title + ' ' + abstract).lower()
        
        # Check if any autism-related term is present
        return any(term in text for term in autism_terms)
    
    def filter_papers(self, papers: List[Paper]) -> List[Paper]:
        """Filter papers to ensure they're autism-related"""
        return [
            paper for paper in papers
            if self.is_autism_related(paper.title, paper.abstract)
        ]
    
    def calculate_relevance_score(self, title: str, abstract: str) -> float:
        """Calculate relevance score based on autism-related terms"""
        text = (title + ' ' + abstract).lower()
        autism_terms = {
            'autism': 1.0,
            'autistic': 0.9,
            'asd': 0.9,
            'autism spectrum disorder': 1.0,
            'asperger': 0.8,
            'neurodevelopmental': 0.7,
            'neurodivergent': 0.7
        }
        
        # Calculate score based on presence of terms
        score = 0.0
        for term, weight in autism_terms.items():
            if term in text:
                score = max(score, weight)
        
        return score if score > 0 else 0.5  # Default score if no terms found

    @lru_cache(maxsize=CACHE_SIZE)
    def fetch_arxiv_papers(self, query: str) -> List[Paper]:
        """Fetch papers from arXiv"""
        try:
            # Always include autism in the search
            search_query = f"autism {query}" if "autism" not in query.lower() else query
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query,
                max_results=10,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in client.results(search):
                title = result.title
                abstract = result.summary
                paper = Paper(
                    title=title,
                    abstract=abstract,
                    url=result.pdf_url,
                    publication_date=result.published.strftime("%Y-%m-%d"),
                    authors=', '.join(author.name for author in result.authors),
                    source="arXiv",
                    relevance_score=self.calculate_relevance_score(title, abstract)
                )
                papers.append(paper)
            
            # Filter for autism relevance
            papers = self.filter_papers(papers)
            return papers
            
        except Exception as e:
            logging.error(f"Error fetching arXiv papers: {str(e)}")
            return []
    
    @lru_cache(maxsize=CACHE_SIZE)
    def fetch_pubmed_papers(self, query: str) -> List[Paper]:
        """Fetch papers from PubMed"""
        try:
            # Always include autism in the search
            search_query = f"autism {query}" if "autism" not in query.lower() else query
            
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=10,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            if not record["IdList"]:
                return []
            
            # Fetch paper details
            handle = Entrez.efetch(
                db="pubmed",
                id=record["IdList"],
                rettype="medline",
                retmode="text"
            )
            records = Medline.parse(handle)
            
            papers = []
            for record in records:
                if "AB" in record:  # Only include papers with abstracts
                    title = record.get("TI", "No title available")
                    abstract = record.get("AB", "No abstract available")
                    paper = Paper(
                        title=title,
                        abstract=abstract,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}/",
                        publication_date=record.get("DP", "Date not available"),
                        authors=record.get("AU", ["Unknown"])[0] if record.get("AU") else "Unknown",
                        source="PubMed",
                        relevance_score=self.calculate_relevance_score(title, abstract)
                    )
                    papers.append(paper)
            
            # Filter for autism relevance
            papers = self.filter_papers(papers)
            return papers
            
        except Exception as e:
            logging.error(f"Error fetching PubMed papers: {str(e)}")
            return []
    
    def fetch_scholar_papers(self, query: str) -> List[Paper]:
        """Fetch papers from Google Scholar"""
        try:
            # Always include autism in the search
            search_query = f"autism {query}" if "autism" not in query.lower() else query
            
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._min_request_interval:
                time.sleep(self._min_request_interval - time_since_last)
            self._last_request_time = time.time()
            
            # Search for papers
            papers = []
            try:
                search_query = scholarly.search_pubs(search_query)
                
                for i in range(SCHOLAR_MAX_PAPERS):
                    try:
                        # Rate limiting between results
                        if i > 0:
                            time.sleep(self._min_request_interval)
                        
                        result = next(search_query)
                        
                        # Validate result structure
                        if not isinstance(result, dict):
                            logging.error(f"Invalid result type: {type(result)}")
                            continue
                            
                        if 'bib' not in result:
                            logging.error("Missing 'bib' in result")
                            continue
                            
                        bib = result.get('bib', {})
                        if not isinstance(bib, dict):
                            logging.error(f"Invalid bib type: {type(bib)}")
                            continue
                        
                        # Extract paper details with fallbacks
                        title = str(bib.get('title', 'No title available'))
                        abstract = str(bib.get('abstract', 'No abstract available'))
                        pub_year = bib.get('pub_year', 'Year not available')
                        authors = bib.get('author', ['Unknown'])
                        
                        # Ensure authors is a list
                        if not isinstance(authors, list):
                            authors = [str(authors)]
                        
                        # Create paper object with validated data
                        paper = Paper(
                            title=title[:500],  # Limit length to prevent issues
                            abstract=abstract[:2000],  # Limit length
                            url=str(result.get('pub_url', '')),
                            publication_date=str(pub_year),
                            authors=authors[0] if authors else 'Unknown',
                            source="Google Scholar",
                            relevance_score=self.calculate_relevance_score(title, abstract)
                        )
                        papers.append(paper)
                        
                    except StopIteration:
                        break
                    except Exception as e:
                        logging.error(f"Error processing Scholar result: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.error(f"Error in scholarly search: {str(e)}")
            
            # Filter for autism relevance
            papers = self.filter_papers(papers)
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
        
        return unique_papers[:10]
