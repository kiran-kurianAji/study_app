import streamlit as st
import os
import time
import tempfile
import re
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# Fix for Windows mimetypes issue
import mimetypes
mimetypes.init()

# Import after mimetypes fix
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# API Keys - Fetch from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if API key is available
if not GOOGLE_API_KEY:
    st.error("❌ Google API Key not found! Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Page configuration
st.set_page_config(page_title="Video Summarizer", page_icon="🎥", layout="wide")
st.title("🎬 Video Summarizer: Upload or YouTube Link")
st.markdown("Choose one of the two options below:")

# Tabs for options
tab1, tab2 = st.tabs(["📁 Upload Video", "📺 YouTube Link"])

### ===== TAB 1: Uploaded Video Option ===== ###
with tab1:
    st.header("Upload a Video and Ask Questions")

    video_file = st.file_uploader("Upload video", type=['mp4', 'mov', 'avi'])
    user_query = st.text_area("What do you want to know from this video?")

    if st.button("🔍 Analyze Uploaded Video"):
        if not video_file:
            st.warning("Please upload a video first.")
        elif not user_query:
            st.warning("Enter a question or query for the video.")
        else:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name

            st.video(video_path)

            try:
                with st.spinner("Uploading and analyzing video..."):
                    uploaded = genai.upload_file(video_path)
                    
                    # Wait for processing
                    while uploaded.state.name == "PROCESSING":
                        time.sleep(1)
                        uploaded = genai.get_file(uploaded.name)
                    
                    if uploaded.state.name == "FAILED":
                        st.error("❌ Video processing failed. Please try again.")
                    else:
                        prompt = f"Analyze the uploaded video for this query: {user_query}\nProvide a clear, concise, and insightful answer."
                        
                        model = genai.GenerativeModel("models/gemini-1.5-flash")
                        response = model.generate_content([prompt, uploaded])
                        
                        st.subheader("📌 Analysis Result")
                        st.markdown(response.text)
                        
                        # Clean up uploaded file from Gemini
                        genai.delete_file(uploaded.name)

            except Exception as e:
                st.error(f"❌ Error analyzing video: {e}")

            # Clean up temporary file
            try:
                Path(video_path).unlink(missing_ok=True)
            except:
                pass

### ===== TAB 2: YouTube Link Option ===== ###
with tab2:
    st.header("Summarize a YouTube Video Transcript")

    def extract_video_id(url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r"v=([a-zA-Z0-9_-]{11})",
            r"youtu\.be/([a-zA-Z0-9_-]{11})",
            r"embed/([a-zA-Z0-9_-]{11})"
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def extract_transcript_details(video_id):
        """Extract transcript from YouTube video"""
        try:
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([i["text"] for i in transcript_text])
            return transcript
        except Exception as e:
            raise e

    def generate_summary(transcript_text, prompt):
        """Generate summary using Gemini"""
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt + transcript_text)
        return response.text

    # YouTube link input
    youtube_link = st.text_input("Paste YouTube Video URL")
    
    # Custom prompt option
    use_custom_prompt = st.checkbox("Use custom prompt")
    if use_custom_prompt:
        custom_prompt = st.text_area("Enter your custom prompt:", 
                                   value="Summarize this video transcript in bullet points, focusing on key takeaways:")
        prompt_base = custom_prompt + "\n\n"
    else:
        prompt_base = (
            "You are a YouTube video summarizer. Summarize the video transcript below in bullet points "
            "within 250 words. Focus on key takeaways:\n\n"
        )

    # Extract video ID and show thumbnail
    video_id = extract_video_id(youtube_link) if youtube_link else None
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

    if st.button("📄 Get Summary from YouTube"):
        if not video_id:
            st.warning("Could not extract a valid video ID. Check the URL format.")
        else:
            try:
                with st.spinner("Fetching transcript and generating summary..."):
                    transcript = extract_transcript_details(video_id)
                    
                    if len(transcript) > 50000:  # Limit very long transcripts
                        st.warning("⚠️ Transcript is very long. Using first 50,000 characters.")
                        transcript = transcript[:50000]
                    
                    summary = generate_summary(transcript, prompt_base)
                    
                st.subheader("📌 YouTube Video Summary")
                st.write(summary)
                    
            except Exception as e:
                st.error(f"Error: {e}")

# Sidebar with instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    **Upload Video Tab:**
    1. Upload a video file (MP4, MOV, AVI)
    2. Ask specific questions about the video
    3. Click "Analyze Uploaded Video"
    
    **YouTube Link Tab:**
    1. Paste a YouTube video URL
    2. Optionally customize the summary prompt
    3. Click "Get Summary from YouTube"
    
    **Features:**
    - Video analysis with AI
    - Custom prompts for summaries
    - Support for various video formats
    
    **Setup:**
    - Create a .env file with your API key
    - Add: GOOGLE_API_KEY=your_api_key_here
    """)
    
    st.header("🔧 Troubleshooting")
    st.markdown("""
    **If you encounter issues:**
    - Make sure all packages are installed
    - Check your .env file exists and has the correct API key
    - Run with: `streamlit run videosummarizer.py`
    """)