import re
from typing import List
from models.paper import Paper

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content"""
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:()\-\'"]', ' ', text)
        return text.strip()
    
    def format_paper(self, title: str, abstract: str) -> str:
        """Format paper title and abstract for context"""
        title = self.clean_text(title)
        abstract = self.clean_text(abstract)
        return f"Title: {title}\nAbstract: {abstract}"
    
    def create_context(self, papers: List[Paper]) -> str:
        """Create a context string from a list of papers"""
        context_parts = []
        
        for i, paper in enumerate(papers, 1):
            # Format the paper information with clear structure
            paper_context = f"""
Research Paper {i}:
Title: {self.clean_text(paper.title)}
Key Points:
- Authors: {paper.authors if paper.authors else 'Not specified'}
- Publication Date: {paper.publication_date}
- Source: {paper.source}

Main Findings:
{self.format_abstract(paper.abstract)}
"""
            context_parts.append(paper_context)
        
        # Join all paper contexts with clear separation
        full_context = "\n" + "="*50 + "\n".join(context_parts)
        
        return full_context
    
    def format_abstract(self, abstract: str) -> str:
        """Format abstract into bullet points for better readability"""
        # Clean the abstract
        clean_abstract = self.clean_text(abstract)
        
        # Split into sentences
        sentences = [s.strip() for s in clean_abstract.split('.') if s.strip()]
        
        # Format as bullet points, combining short sentences
        bullet_points = []
        current_point = []
        
        for sentence in sentences:
            current_point.append(sentence)
            if len(' '.join(current_point)) > 100 or sentence == sentences[-1]:
                bullet_points.append('- ' + '. '.join(current_point) + '.')
                current_point = []
        
        return '\n'.join(bullet_points)
