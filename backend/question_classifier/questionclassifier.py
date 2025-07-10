import streamlit as st
import os
import re
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PDF and DOCX processing
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Configure Gemini API from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Please create a .env file with your API key.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

class QuestionAnalyzer:
    def __init__(self):
        try:
            self.gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
        except Exception as e:
            st.error(f"Error initializing Gemini model: {str(e)}")
            st.stop()

    def extract_questions_from_text(self, text: str) -> List[str]:
        try:
            # Enhanced prompt for better question extraction
            prompt = f"""
            Extract all exam-style questions from the following text. Follow these rules:
            1. Questions start with a number (like 1., 2., 3., etc.) on the left side
            2. Each question may have subparts like a), b), c), etc.
            3. Include all subparts as part of the same question
            4. A question ends when a new number appears on the left side
            5. Return each complete question (including all its subparts) on a separate line
            6. Do not include any text that is not part of a question
            7. Keep the original numbering and formatting
            
            Text to analyze:
            {text}
            
            Return only the questions, one complete question per line:
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            if not response or not response.text:
                st.warning("No response received from Gemini API")
                return []
                
            lines = response.text.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                if line and self._is_valid_question(line):
                    questions.append(line)
            
            return questions
            
        except Exception as e:
            st.error(f"Gemini extraction error: {str(e)}")
            return []

    def _is_valid_question(self, text: str) -> bool:
        """Check if the text looks like a valid question"""
        # Check if it starts with a number followed by a dot or parenthesis
        if re.match(r'^\d+[\.\)]\s*', text):
            return True
        # Check if it contains question words or ends with question mark
        question_indicators = ['what', 'why', 'how', 'when', 'where', 'which', 'who', 'explain', 'describe', 'define', 'analyze', 'compare', 'discuss']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators) or text.endswith('?')

    def extract_questions_from_pdf(self, file) -> List[str]:
        if not PDF_AVAILABLE:
            st.error("PDF libraries not available.")
            return []

        try:
            file.seek(0)
            text = ""
            
            try:
                # Try PyMuPDF first
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                for page in pdf_document:
                    text += page.get_text() + "\n"
                pdf_document.close()
            except:
                # Fallback to pdfplumber
                file.seek(0)
                import pdfplumber
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            return self.extract_questions_from_text(text)
            
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return []

    def extract_questions_from_docx(self, file) -> List[str]:
        if not DOCX_AVAILABLE:
            st.error("DOCX support not available.")
            return []

        try:
            file.seek(0)
            doc = Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
            return self.extract_questions_from_text(text)
        except Exception as e:
            st.error(f"DOCX extraction error: {str(e)}")
            return []

    def generate_answer(self, question: str) -> str:
        try:
            # Enhanced prompt for better answer generation
            prompt = f"""
            Please provide a detailed and comprehensive answer to this exam question. 
            Make sure to:
            1. Address all parts of the question if it has subparts
            2. Provide clear explanations
            3. Include relevant examples where appropriate
            4. Structure your answer logically
            
            Question: {question}
            
            Answer:
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            if not response or not response.text:
                return "Error: No response received from Gemini API"
                
            return response.text.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    st.set_page_config(page_title="📄 Question Extractor", layout="wide")
    st.title("📄 Question Paper Analyzer")
    
    st.markdown("""
    **Instructions:**
    - Upload question papers in PDF, DOCX, or TXT format
    - The tool will extract questions that start with numbers (1., 2., 3., etc.)
    - Questions with subparts (a), b), c)) will be kept together
    - Click "Generate Answer" to get AI-generated answers
    """)

    uploaded_files = st.file_uploader(
        "Upload question papers (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("Please upload one or more question paper files to get started.")
        return

    analyzer = QuestionAnalyzer()
    all_questions = []

    # Process each uploaded file
    with st.spinner("Processing uploaded files..."):
        for file in uploaded_files:
            st.write(f"Processing: {file.name}")
            ext = file.name.split(".")[-1].lower()
            
            if ext == "pdf":
                questions = analyzer.extract_questions_from_pdf(file)
            elif ext == "docx":
                questions = analyzer.extract_questions_from_docx(file)
            elif ext == "txt":
                content = str(file.read(), "utf-8")
                questions = analyzer.extract_questions_from_text(content)
            else:
                st.warning(f"Unsupported file type: {ext}")
                continue
                
            all_questions.extend(questions)
            st.success(f"Extracted {len(questions)} questions from {file.name}")

    if not all_questions:
        st.warning("No questions were extracted from the uploaded files. Please check if your files contain numbered questions.")
        return

    # Deduplicate questions if multiple files uploaded
    if len(uploaded_files) > 1:
        seen = set()
        deduped_questions = []
        for q in all_questions:
            # More sophisticated deduplication
            normalized = re.sub(r'\W+', '', q.lower())
            if normalized not in seen:
                deduped_questions.append(q)
                seen.add(normalized)
        all_questions = deduped_questions
        st.info(f"Removed duplicates. Total unique questions: {len(all_questions)}")

    st.subheader(f"Extracted Questions ({len(all_questions)} found)")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        for i, q in enumerate(all_questions):
            with st.expander(f"Question {i+1}", expanded=False):
                st.markdown(f"**{q}**")
                
                # Generate answer button
                if st.button(f"Generate Answer", key=f"btn_{i}"):
                    with st.spinner("Generating answer..."):
                        answer = analyzer.generate_answer(q)
                        st.markdown("**Answer:**")
                        st.markdown(answer)
                        
                        # Add copy button functionality
                        st.code(answer, language=None)
    
    with col2:
        st.markdown("### Summary")
        st.metric("Total Questions", len(all_questions))
        st.metric("Files Processed", len(uploaded_files))
        
        # Export functionality
        if st.button("Export All Questions"):
            questions_text = "\n\n".join([f"Q{i+1}: {q}" for i, q in enumerate(all_questions)])
            st.download_button(
                label="Download Questions as Text",
                data=questions_text,
                file_name="extracted_questions.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()