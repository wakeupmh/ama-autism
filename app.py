import streamlit as st
import logging
from services.research_fetcher import ResearchFetcher
from services.model_handler import ModelHandler
from utils.text_processor import TextProcessor
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class AutismResearchApp:
    def __init__(self):
        """Initialize the application components"""
        self.research_fetcher = ResearchFetcher()
        self.model_handler = ModelHandler()
        self.text_processor = TextProcessor()
    
    def _setup_streamlit(self):
        """Setup Streamlit UI components"""
        st.title("🧩 AMA Autism")
        st.subheader("Your one-stop shop for autism research!")
        st.markdown("""
        Ask questions about autism research, and I'll analyze recent papers to provide evidence-based answers.
        """)
    
    def _fetch_research(self, query: str):
        """Fetch research papers for the given query"""
        papers = self.research_fetcher.fetch_all_papers(query)
        if not papers:
            st.warning("No relevant research papers found. Please try a different search term.")
            return None
        return papers
    
    def _display_sources(self, papers: List):
        """Display the source papers used to generate the answer"""
        st.markdown("### Sources")
        for i, paper in enumerate(papers, 1):
            st.markdown(f"**{i}. [{paper.title}]({paper.url})**")
            
            # Create three columns for metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                if paper.authors:
                    st.markdown(f"👥 Authors: {paper.authors}")
            with col2:
                st.markdown(f"📅 Published: {paper.publication_date}")
            with col3:
                st.markdown(f"📜 Source: {paper.source}")
            
            # Show abstract in expander
            with st.expander("📝 View Abstract"):
                st.markdown(paper.abstract)
            
            if i < len(papers):  # Add separator between papers except for the last one
                st.divider()
    
    def run(self):
        """Run the main application loop"""
        self._setup_streamlit()
        
        # Initialize session state for papers
        if 'papers' not in st.session_state:
            st.session_state.papers = []
        
        # Get user query
        query = st.text_input("What would you like to know about autism?")
        
        if query:
            # Show status while processing
            with st.status("Processing your question...") as status:
                # Fetch papers
                status.write("🔍 Searching for relevant research papers...")
                try:
                    papers = self.research_fetcher.fetch_all_papers(query)
                except Exception as e:
                    st.error(f"Error fetching research papers: {str(e)}")
                    return
                
                if not papers:
                    st.warning("No relevant papers found. Please try a different query.")
                    return
                
                # Generate and validate answer
                status.write("📚 Analyzing research papers...")
                context = self.text_processor.create_context(papers)
                
                status.write("✍️ Generating answer...")
                answer = self.model_handler.generate_answer(query, context)
                
                status.write("✅ Validating answer...")
                is_valid, validation_message = self.model_handler.validate_answer(answer, context)
                
                status.write("✨ All done! Displaying results...")
            
            # Display results
            if is_valid:
                st.success("✅ Research analysis complete! The answer has been validated for accuracy.")
            else:
                st.warning("⚠️ The answer may contain information not fully supported by the research.")
            
            st.markdown("### Answer")
            st.markdown(answer)
            
            st.markdown("### Validation")
            st.info(f"🔍 {validation_message}")
            
            st.divider()
            self._display_sources(papers)

def main():
    app = AutismResearchApp()
    app.run()

if __name__ == "__main__":
    main()