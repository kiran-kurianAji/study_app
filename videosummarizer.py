import streamlit as st
import os
import time
import tempfile
import re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import openai

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Page configuration
st.set_page_config(page_title="Video Summarizer", page_icon="🎥", layout="wide")
st.title("🎬 Video Summarizer: Upload or YouTube Link")
st.markdown("Choose one of the two options below:")

# Tabs for options
tab1, tab2 = st.tabs(["📁 Upload Video", "📺 YouTube Link"])

### ===== TAB 1: Uploaded Video Option ===== ###
with tab1:
    st.header("Upload a Video and Ask Questions")

    @st.cache_resource
    def initialize_agent():
        return Agent(
            name="Video AI Summarizer",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    multimodal_Agent = initialize_agent()

    video_file = st.file_uploader("Upload video", type=['mp4', 'mov', 'avi'])

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path)

        user_query = st.text_area("What do you want to know from this video?")

        if st.button("🔍 Analyze Uploaded Video"):
            if not user_query:
                st.warning("Enter a question or query for the video.")
            else:
                try:
                    with st.spinner("Uploading and analyzing video..."):
                        processed_video = upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)

                        analysis_prompt = (
                            f"Analyze the uploaded video for this query: {user_query}\n"
                            "Provide a clear, concise, and insightful answer."
                        )

                        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                    st.subheader("📌 Analysis Result")
                    st.markdown(response.content)

                except Exception as e:
                    st.error(f"Error during video analysis: {e}")
                finally:
                    Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Upload a video to begin analysis.")

### ===== TAB 2: YouTube Link Option ===== ###
with tab2:
    st.header("Summarize a YouTube Video Transcript")

    prompt = (
        "You are a YouTube video summarizer. Summarize the video transcript below in bullet points "
        "within 250 words. Focus on key takeaways:\n"
    )

    def extract_video_id(url):
        patterns = [
            r"v=([a-zA-Z0-9_-]{11})",             # watch?v=...
            r"youtu\.be/([a-zA-Z0-9_-]{11})",     # youtu.be/...
            r"embed/([a-zA-Z0-9_-]{11})"          # embed/...
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def extract_transcript_details(video_id):
        try:
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([i["text"] for i in transcript_text])
            return transcript
        except Exception as e:
            raise e

    def generate_summary_with_fallback(transcript_text, prompt):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt + transcript_text)
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning("summarizing....")
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful video summarizer."},
                            {"role": "user", "content": prompt + transcript_text}
                        ],
                        temperature=0.5,
                    )
                    return completion.choices[0].message.content
                except Exception as openai_error:
                    raise Exception(f"Both Gemini and OpenAI failed: {openai_error}")
            else:
                raise e

    youtube_link = st.text_input("Paste YouTube Video URL")

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
                    summary = generate_summary_with_fallback(transcript, prompt)
                st.subheader("📌 YouTube Video Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error fetching transcript: {e}")