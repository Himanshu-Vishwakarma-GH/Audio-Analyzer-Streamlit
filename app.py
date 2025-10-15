import streamlit as st
import os
import sys
import time
import json
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions
from googleapiclient import errors as googleapiclient_errors
import pandas as pd

# ----------------- Configuration -----------------
load_dotenv()

st.set_page_config(page_title="Gemini Audio Analyzer", layout="wide")
st.title("🎧 Audio Analyzer — Streamlit")
st.markdown(
    """
    Drag & drop karo apna audio file (MP3 / WAV / FLAC / M4A).

    Ye app Gemini (Google Generative AI) use karke:
    - full transcription
    - sentiment (overall + segments)
    - noise analysis
    - scene / environment prediction
    - layers of actions / events (what's happening)
    - focus / topics / what to pay attention to
    - detailed, timestamped segment analysis

    **Note:** App Gemini API key chaahiye — set `GOOGLE_API_KEY` in your environment or in a `.env` file.
    """
)

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Gemini model", options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], index=0)
    st.markdown("---")
    st.markdown("**Install / Run**")
    st.code("pip install streamlit google-generativeai python-dotenv pandas")
    st.code("streamlit run streamlit_gemini_audio_analyzer.py")
    st.markdown("---")
    st.info("Keep your GOOGLE_API_KEY secret. This app uploads audio to Gemini — usage may incur costs.")
    st.markdown(
        """
        The Gemini file upload now requires a RAG store name. Set the environment variable `GOOGLE_RAG_STORE_NAME`
        (or `RAG_STORE_NAME`) to the name of the RAG store you want to upload files to.
        """
    )

# Helper functions

def configure_gemini_api():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Set it in environment or .env and rerun the app.")
        st.stop()
    genai.configure(api_key=api_key)


def upload_file_with_retries(path, max_retries=3, initial_delay=5, rag_store_name=None):
    """Uploads a file with retry mechanism. rag_store_name is required for Gemini uploads that target a RAG store."""
    if not rag_store_name:
        # Fail early with a clear message if the required parameter isn't provided
        raise ValueError('Missing required RAG store name. Set the environment variable GOOGLE_RAG_STORE_NAME or provide rag_store_name to upload_file_with_retries.')

    last_exc = None
    for attempt in range(max_retries):
        try:
            # The server-side error message references "ragStoreName" — pass it in the upload call.
            # Some SDK versions accept snake_case; passing the camelCase name to the underlying API often works.
            file_ref = genai.upload_file(path=path, ragStoreName=rag_store_name)
            return file_ref
        except (exceptions.ServiceUnavailable, googleapiclient_errors.ResumableUploadError, googleapiclient_errors.HttpError) as e:
            # Only retry on 503 Service Unavailable errors or transient upload errors
            if isinstance(e, googleapiclient_errors.HttpError) and getattr(e, "resp", None) and getattr(e.resp, "status", None) != 503:
                raise e  # Re-raise if it's not a 503 error

            last_exc = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                st.warning(f"Upload failed due to service availability. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e
    raise last_exc


def wait_for_processing(file_ref, poll_interval=5, timeout=300):
    """Waits for file processing to complete with a timeout."""
    start = time.time()
    while True:
        file_ref = genai.get_file(file_ref.name)
        state = getattr(file_ref.state, 'name', None)
        if state != "PROCESSING":
            return file_ref
        if time.time() - start > timeout:
            raise TimeoutError("File processing timeout")
        time.sleep(poll_interval)


def extract_json_from_text(text):
    """Extracts JSON from a string, handling potential errors."""
    # Best-effort: try to load JSON directly, else try to find first/last braces
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return None


# Uploader
uploaded = st.file_uploader("Drop your audio file here", type=["mp3", "wav", "flac", "m4a"], accept_multiple_files=False)

if uploaded is not None:
    st.audio(uploaded)
    # Save to temp file because genai.upload_file expects a path
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()

    st.markdown(f"**File:** {uploaded.name} — {os.path.getsize(tmp.name) // 1024} KB")

    # Initialize session state for analysis and chat
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat" not in st.session_state:
        st.session_state.chat = None

    if st.button("Analyze with Gemini") or st.session_state.analysis:
        configure_gemini_api()

        # Resolve RAG store name from environment
        rag_store_name = os.getenv("GOOGLE_RAG_STORE_NAME") or os.getenv("RAG_STORE_NAME")
        if not rag_store_name:
            st.error("Missing RAG store name. Set the environment variable GOOGLE_RAG_STORE_NAME (or RAG_STORE_NAME) to the name of your RAG store and redeploy.")
            st.stop()

        # Perform analysis only if it hasn't been done yet
        if st.session_state.analysis is None:
            with st.spinner("Uploading and processing audio — this can take some time..."):
                try:
                    file_ref = upload_file_with_retries(tmp.name, rag_store_name=rag_store_name)
                    st.session_state.file_ref = file_ref # Save file_ref for chat
                    file_ref = wait_for_processing(file_ref, poll_interval=5, timeout=600)
                except Exception as e:
                    st.error(f"File processing error: {e}")
                    st.stop()

            st.success("Upload complete. Requesting analysis from Gemini...")

            # Prompt asking for structured JSON output
            prompt = f"""
            You are an expert intelligence analyst. Analyze the provided audio file by focusing on vocal characteristics like tone, pitch, and prosody to infer emotional state and intent. Return a sin[...]

            1.  "transcription": A full and accurate transcript of all spoken words.
            2.  "sentiment_analysis": An object describing the overall sentiment based on the words spoken. It must contain "score" (float from -1.0 to 1.0), "label" ('Positive', 'Negative', 'Neutral'[...]
            3.  "emotion_analysis": An object describing the overall emotion detected from the speaker's tone and prosody. It must contain a "label" (e.g., 'Angry', 'Happy', 'Sad', 'Anxious', 'Neutral[...]
            4.  "segments": An array of objects, where each object represents a segment of speech and contains:
                - "start_time": float, in seconds.
                - "end_time": float, in seconds.
                - "text": The transcribed text of the segment.
                - "sentiment": An object with "score" and "label" for the text in this segment.
                - "emotion": An object with "label" and "intensity" based on the tone of this segment.
            5.  "noise_analysis": An object with "level" ('Low', 'Medium', 'High') and a "description" of background sounds.
            6.  "scene_prediction": A string describing the most likely environment (e.g., 'Quiet office meeting', 'Busy cafe').
            7.  "topics": A list of main topics and keywords discussed.
            8.  "focus_summary": A brief paragraph summarizing the main focus and recommendations.
            9.  "sound_events": An array of objects, each detecting a non-speech sound. Each object must contain "start_time" (float), "end_time" (float), and "event_description" (string, e.g., 'dog b[...]
            10. "confidence": A float from 0.0 to 1.0 indicating your overall confidence in the analysis.

            ONLY OUTPUT VALID JSON. Do not include any commentary or explanation outside the JSON.
            """


            # Call Gemini model for initial analysis
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={"response_mime_type": "application/json"},
                )
                with st.spinner("Generating analysis from Gemini (this can take up to a few minutes)..."):
                    response = model.generate_content([prompt, file_ref], request_options={"timeout": 1200})
                
                text = getattr(response, 'text', None) or str(response)
                parsed = extract_json_from_text(text)

                if parsed is None:
                    st.error("Could not parse JSON from Gemini response. Showing raw response below.")
                    st.subheader("Raw response")
                    st.code(text)
                    st.stop()
                else:
                    st.session_state.analysis = parsed

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        # --- Display Analysis and Chat ---
        analysis = st.session_state.analysis
        st.header("Transcription")
        st.write(analysis.get("transcription", ""))

        st.header("Overall Sentiment")
        overall = analysis.get("sentiment_analysis", {})
        st.metric(label="Sentiment label", value=overall.get("label", "N/A"), delta=f"Score: {overall.get('score', 'N/A')}")
        st.write(overall.get("justification", ""))

        st.header("Overall Emotion (from Tone)")
        emotion = analysis.get("emotion_analysis", {})
        st.metric(label="Detected Emotion", value=emotion.get("label", "N/A"), delta=f"Intensity: {emotion.get('intensity', 'N/A')}")
        st.write(emotion.get("justification", ""))

        # ... (display other analysis sections as before)
        st.header("Noise Analysis & Scene")
        na = analysis.get("noise_analysis", {})
        st.write(f"**Level:** {na.get('level', 'N/A')}")
        st.write(f"**Description:** {na.get('description', 'N/A')}")
        st.write(f"**Scene prediction:** {analysis.get('scene_prediction', 'N/A')}")

        st.header("Detected Sound Events (Non-Speech)")
        sound_events = analysis.get("sound_events", [])
        if sound_events:
            df_events = pd.DataFrame(sound_events)
            st.dataframe(df_events)
        else:
            st.write("No significant non-speech sound events were detected.")

        # Download button
        st.download_button(label="Download analysis (JSON)", data=json.dumps(analysis, indent=2), file_name=f"analysis_{uploaded.name}.json", mime="application/json")

        # --- CHAT FEATURE ---
        st.markdown("---")
        st.header("💬 Chat About This Audio")

        # Initialize chat if it doesn't exist
        if st.session_state.chat is None:
            chat_model = genai.GenerativeModel(model_name=model_name)
            initial_prompt = f"""
            You are a helpful assistant. The user has just analyzed an audio file.
            Here is the complete JSON analysis of that audio file:
            {json.dumps(st.session_state.analysis, indent=2)}
            The user will now ask you questions about this audio file and its analysis.
            Use the provided JSON to answer accurately.
            """

            st.session_state.chat = chat_model.start_chat(history=[
            {'role': 'user', 'parts': [initial_prompt]},
            {'role': 'model', 'parts': ["Understood. I have the analysis and I'm ready to answer questions."]}
            ])

            st.session_state.messages.append({"role": "assistant", "content": "I've analyzed the audio. What would you like to know?"})

        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input
        if prompt := st.chat_input("Ask a question about the audio..."):
            # Add user message to UI and state
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Send to Gemini and get response
            with st.spinner("Gemini is thinking..."):
                response = st.session_state.chat.send_message(prompt)
            
            # Add assistant response to UI and state
            with st.chat_message("assistant"):
                st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

else:
    st.info("Upload an audio file (mp3/wav/flac/m4a) to begin.")



# Footer
st.markdown("---")
st.caption("Built with Streamlit + Google Gemini. Keep API keys safe.")
