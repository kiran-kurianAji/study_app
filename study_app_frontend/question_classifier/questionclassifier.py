import streamlit as st
import os
import re
from collections import defaultdict
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

class QuestionAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.85
        self.openai_client = None
        self._setup_openai()
    
    def _setup_openai(self):
        """Setup OpenAI client with API key"""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            st.warning("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file to enable answer generation.")
    
    def extract_questions_from_text(self, text: str) -> List[str]:
        """Extract questions from text content"""
        lines = text.split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            
            # Only consider lines that contain a question mark
            if line and '?' in line:
                # Remove leading numbers and special characters
                cleaned_line = re.sub(r'^\d+[\.\)\-\s]*', '', line).strip()
                if cleaned_line and '?' in cleaned_line:
                    questions.append(line)
        
        return questions
    
    def extract_questions_from_pdf(self, file) -> List[str]:
        """Extract questions from PDF file"""
        if not PDF_AVAILABLE:
            st.error("PDF processing libraries not available. Please install PyMuPDF or pdfplumber.")
            return []
        
        try:
            # Try PyMuPDF first
            if 'fitz' in globals():
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                text = ""
                for page in pdf_document:
                    text += page.get_text()
                pdf_document.close()
            else:
                # Fallback to pdfplumber
                import pdfplumber
                text = ""
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            return self.extract_questions_from_text(text)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []
    
    def extract_questions_from_docx(self, file) -> List[str]:
        """Extract questions from DOCX file"""
        if not DOCX_AVAILABLE:
            st.error("DOCX processing library not available. Please install python-docx.")
            return []
        
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return self.extract_questions_from_text(text)
        except Exception as e:
            st.error(f"Error processing DOCX: {str(e)}")
            return []
    
    def normalize_question(self, question: str) -> str:
        """Normalize question text for comparison"""
        # Remove leading numbers and special characters
        normalized = re.sub(r'^\d+[\.\)\-\s]*', '', question)
        # Remove marks notation
        normalized = re.sub(r'\(\d+\s*marks?\)', '', normalized, flags=re.IGNORECASE)
        # Clean whitespace and convert to lowercase
        normalized = normalized.strip().lower()
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def extract_marks(self, question: str) -> int:
        """Extract marks from question using regex"""
        match = re.search(r'\((\d+)\s*marks?\)', question, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0
    
    def group_similar_questions(self, questions: List[str]) -> Dict:
        """Group semantically similar questions"""
        if not questions:
            return {}
        
        # Normalize questions for similarity comparison
        normalized_questions = [self.normalize_question(q) for q in questions]
        
        # Generate embeddings
        embeddings = self.model.encode(normalized_questions)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group similar questions
        groups = defaultdict(list)
        processed = set()
        
        for i, question in enumerate(questions):
            if i in processed:
                continue
            
            # Find similar questions
            similar_indices = []
            for j in range(len(questions)):
                if j != i and similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_indices.append(j)
            
            # Create group with representative question (first occurrence)
            group_key = question
            groups[group_key] = {
                'questions': [question],
                'frequency': 1,
                'marks': self.extract_marks(question),
                'normalized': self.normalize_question(question)
            }
            
            # Add similar questions to group
            for idx in similar_indices:
                if idx not in processed:
                    groups[group_key]['questions'].append(questions[idx])
                    groups[group_key]['frequency'] += 1
                    processed.add(idx)
            
            processed.add(i)
        
        return groups
    
    def generate_answer(self, question: str) -> str:
        """Generate answer using OpenAI API"""
        if not self.openai_client:
            return "OpenAI API not configured. Please add your API key to .env file."
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful academic assistant. Provide clear, comprehensive answers to exam questions."},
                    {"role": "user", "content": f"Please provide a detailed answer to this question: {question}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    st.set_page_config(
        page_title="Question Paper Analyzer",
        layout="wide"
    )
    
    st.title("Question Paper Analyzer")
    st.markdown("Upload multiple question papers to identify the most important and repeated questions")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = QuestionAnalyzer()
    
    # File upload section
    st.header("Upload Question Papers")
    
    # Determine accepted file types
    accepted_types = ['.txt']
    if PDF_AVAILABLE:
        accepted_types.append('.pdf')
    if DOCX_AVAILABLE:
        accepted_types.append('.docx')
    
    uploaded_files = st.file_uploader(
        "Choose question paper files",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx'] if PDF_AVAILABLE and DOCX_AVAILABLE else ['txt'],
        help=f"Supported formats: {', '.join(accepted_types)}"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
        
        # Display uploaded files
        with st.expander("View Uploaded Files"):
            for file in uploaded_files:
                st.write(f"{file.name} ({file.size} bytes)")
    
    # Analysis section
    if uploaded_files and st.button("Analyze Questions", type="primary"):
        with st.spinner("Analyzing question papers..."):
            all_questions = []
            
            # Process each file
            for file in uploaded_files:
                file_extension = file.name.split('.')[-1].lower()
                
                if file_extension == 'txt':
                    content = str(file.read(), "utf-8")
                    questions = st.session_state.analyzer.extract_questions_from_text(content)
                elif file_extension == 'pdf' and PDF_AVAILABLE:
                    file.seek(0)  # Reset file pointer
                    questions = st.session_state.analyzer.extract_questions_from_pdf(file)
                elif file_extension == 'docx' and DOCX_AVAILABLE:
                    file.seek(0)  # Reset file pointer
                    questions = st.session_state.analyzer.extract_questions_from_docx(file)
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    continue
                
                all_questions.extend(questions)
            
            if not all_questions:
                st.error("No questions found in the uploaded files.")
                return
            
            # Group similar questions
            grouped_questions = st.session_state.analyzer.group_similar_questions(all_questions)
            
            # Sort by frequency and marks
            sorted_groups = sorted(
                grouped_questions.items(),
                key=lambda x: (x[1]['frequency'], x[1]['marks']),
                reverse=True
            )
            
            # Store results in session state
            st.session_state.analysis_results = sorted_groups
            st.session_state.total_questions = len(all_questions)
            st.session_state.unique_groups = len(grouped_questions)
    
    # Display results
    if 'analysis_results' in st.session_state:
        display_results()

def display_results():
    """Display analysis results"""
    st.header("Analysis Results")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", st.session_state.total_questions)
    with col2:
        st.metric("Unique Question Groups", st.session_state.unique_groups)
    with col3:
        repeated_count = sum(1 for _, group in st.session_state.analysis_results if group['frequency'] > 1)
        st.metric("Repeated Questions", repeated_count)
    
    st.markdown("---")
    
    # Display grouped questions
    st.header("Question Analysis")
    
    for i, (representative_question, group_data) in enumerate(st.session_state.analysis_results):
        with st.container():
            # Question header
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                st.subheader(f"Question {i+1}")
                st.write(f"**Question:** {representative_question}")
            
            with col2:
                st.metric("Frequency", group_data['frequency'])
            
            with col3:
                marks_display = group_data['marks'] if group_data['marks'] > 0 else "N/A"
                st.metric("Marks", marks_display)
            
            # Show all variations if more than one
            if len(group_data['questions']) > 1:
                with st.expander(f"View all {len(group_data['questions'])} variations"):
                    for j, question in enumerate(group_data['questions']):
                        st.write(f"{j+1}. {question}")
            
            # Answer generation button
            answer_key = f"answer_{i}"
            if st.button("Generate Answer", key=f"btn_{i}", help="Generate answer using OpenAI API"):
                with st.spinner("Generating answer..."):
                    answer = st.session_state.analyzer.generate_answer(representative_question)
                    st.session_state[answer_key] = answer
            
            # Display answer if generated
            if answer_key in st.session_state:
                st.markdown("**Answer:**")
                st.write(st.session_state[answer_key])
            
            st.markdown("---")

if __name__ == "__main__":
    main()