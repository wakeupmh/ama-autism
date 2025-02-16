from dataclasses import dataclass
from typing import Optional

@dataclass
class Paper:
    title: str
    abstract: str
    url: str
    publication_date: str
    relevance_score: float
    source: str
    authors: Optional[str] = None
