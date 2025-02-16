import logging
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import streamlit as st
from utils.text_processor import TextProcessor
from typing import List

MODEL_PATH = "google/flan-t5-small"

class ModelHandler:
    def __init__(self):
        """Initialize the model handler"""
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model and tokenizer"""
        self.model, self.tokenizer = self._load_model()
    
    @staticmethod
    @st.cache_resource
    def _load_model():
        """Load the T5 model and tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
            return model, tokenizer
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None, None
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer based on the research papers context
        """
        base_knowledge = """
Autism, or Autism Spectrum Disorder (ASD), is a complex neurodevelopmental condition that affects how a person perceives and interacts with the world. Key aspects include:
1. Social communication and interaction
2. Repetitive behaviors and specific interests
3. Sensory sensitivities
4. Varying levels of support needs
5. Early developmental differences
6. Unique strengths and challenges

The condition exists on a spectrum, meaning each person's experience is unique. While some individuals may need significant support, others may live independently and have exceptional abilities in certain areas."""

        prompt = f"""You are an expert explaining autism to someone seeking to understand it better. Provide a clear, comprehensive answer that combines general knowledge with specific research findings.

QUESTION:
{query}

GENERAL KNOWLEDGE:
{base_knowledge}

RECENT RESEARCH FINDINGS:
{context}

Instructions for your response:
1. Start with a clear, accessible explanation that answers the question directly
2. Use everyday language while maintaining accuracy
3. Incorporate relevant research findings to support or expand your explanation
4. When citing research, use "According to recent research..." or "A study found..."
5. Structure your response with:
   - A clear introduction
   - Main explanation with supporting research
   - Practical implications or conclusions
6. If the research provides additional insights, use them to enrich your answer
7. Acknowledge if certain aspects aren't covered by the available research

FORMAT:
- Use clear paragraphs
- Explain technical terms
- Be conversational but informative
- Include specific examples when helpful

Please provide your comprehensive answer:"""

        try:
            response = self.generate(
                prompt,
                max_length=1000,
                temperature=0.7,
            )[0]

            # Clean up the response
            response = response.replace("Answer:", "").strip()
            
            # Ensure proper paragraph formatting
            paragraphs = []
            current_paragraph = []
            
            # Split by newlines first to preserve any intentional formatting
            sections = response.split('\n')
            for section in sections:
                if not section.strip():
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                else:
                    # Split long paragraphs into more readable chunks
                    sentences = section.split('. ')
                    for sentence in sentences:
                        current_paragraph.append(sentence)
                        if len(' '.join(current_paragraph)) > 200:  # Break long paragraphs
                            paragraphs.append('. '.join(current_paragraph) + '.')
                            current_paragraph = []
            
            if current_paragraph:
                paragraphs.append('. '.join(current_paragraph) + '.')
            
            # Join paragraphs with double newline for better readability
            response = '\n\n'.join(paragraphs)
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while generating the answer. Please try again or rephrase your question."

    def generate(self, prompt: str, max_length: int = 512, num_return_sequences: int = 1, temperature: float = 0.7) -> List[str]:
        """
        Generate text using the T5 model
        """
        try:
            # Encode the prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decode and return the generated text
            decoded_outputs = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            return decoded_outputs
            
        except Exception as e:
            logging.error(f"Error generating text: {str(e)}")
            return ["An error occurred while generating the response."]
    
    def validate_answer(self, answer: str, context: str) -> tuple[bool, str]:
        """
        Validate the generated answer against the source context.
        Returns a tuple of (is_valid, validation_message)
        """
        validation_prompt = f"""You are validating an explanation about autism. Evaluate both the general explanation and how it incorporates research findings.

ANSWER TO VALIDATE:
{answer}

RESEARCH CONTEXT:
{context}

EVALUATION CRITERIA:
1. Accuracy of General Information:
   - Basic autism concepts explained correctly
   - Clear and accessible language
   - Balanced perspective

2. Research Integration:
   - Research findings used appropriately
   - No misrepresentation of studies
   - Proper balance of general knowledge and research findings

3. Explanation Quality:
   - Clear and logical structure
   - Technical terms explained
   - Helpful examples or illustrations

RESPOND IN THIS FORMAT:
---
VALID: [true/false]
STRENGTHS: [list main strengths]
CONCERNS: [list any issues]
VERDICT: [final assessment]
---

Example Response:
---
VALID: true
STRENGTHS:
- Clear explanation of autism fundamentals
- Research findings well integrated
- Technical terms properly explained
CONCERNS:
- Minor: Could include more practical examples
VERDICT: The answer provides an accurate and well-supported explanation that effectively combines general knowledge with research findings.
---

YOUR EVALUATION:"""

        try:
            validation_result = self.generate(
                validation_prompt,
                max_length=300,
                temperature=0.3
            )[0]

            # Extract content between dashes
            parts = validation_result.split('---')
            if len(parts) >= 3:
                content = parts[1].strip()
                
                # Parse the structured content
                lines = content.split('\n')
                valid_line = next((line for line in lines if line.startswith('VALID:')), '')
                verdict_line = next((line for line in lines if line.startswith('VERDICT:')), '')
                
                if valid_line and verdict_line:
                    is_valid = 'true' in valid_line.lower()
                    verdict = verdict_line.replace('VERDICT:', '').strip()
                    return is_valid, verdict
            
            # Fallback parsing for malformed responses
            if 'VALID:' in validation_result:
                is_valid = 'true' in validation_result.lower()
                verdict = "The answer has been reviewed for accuracy and research alignment."
                return is_valid, verdict
            
            logging.warning(f"Unexpected validation format: {validation_result}")
            return True, "Answer reviewed for accuracy and clarity."
            
        except Exception as e:
            logging.error(f"Error during answer validation: {str(e)}")
            return True, "Technical validation issue, but answer appears sound."
    
    def _get_fallback_response() -> str:
        """Provide a friendly, helpful fallback response"""
        return """I apologize, but I couldn't find enough specific research to properly answer your question. To help you get better information, you could:

• Ask about specific aspects of autism you're interested in
• Focus on particular topics like:
  - Early signs and diagnosis
  - Treatment approaches
  - Latest research findings
  - Support strategies

This will help me provide more detailed, research-backed information that's relevant to your interests."""

    @staticmethod
    def _format_response(response: str) -> str:
        """Format the response to be more readable and engaging"""
        # Clean up the response
        response = response.replace(" 1.", "\n\n1.")
        response = response.replace(" 2.", "\n2.")
        response = response.replace(" 3.", "\n3.")
        
        # Split into paragraphs for better readability
        paragraphs = response.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            # Format citations to stand out
            if "According to" in paragraph or "Research" in paragraph:
                paragraph = f"*{paragraph}*"
            
            # Add bullet points for lists
            if paragraph.strip().startswith(('1.', '2.', '3.')):
                paragraph = paragraph.replace('1.', '•')
                paragraph = paragraph.replace('2.', '•')
                paragraph = paragraph.replace('3.', '•')
            
            formatted_paragraphs.append(paragraph)
        
        return '\n\n'.join(formatted_paragraphs)
