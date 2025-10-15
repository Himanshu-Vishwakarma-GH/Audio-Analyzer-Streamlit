import streamlit as st
import os
import time
import json
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions
import pandas as pd

# ----------------- Simple Gemini Audio Analyzer -----------------
# This version PURPOSEFULLY avoids any RAG / corpus ingestion APIs.
# It uses genai.upload_file(path=...) (the simple upload flow) and then
# sends the returned file reference with the model.generate_content call.
#
# If your deployment environment or server enforces RAG-only ingestion,
# this still may hit an error from the remote API. But if previously the
# app used genai.upload_file successfully, this restores that simple flow.

load_dotenv()
st.set_page_config(page_title="Gemini Audio Analyzer (Simple)", layout="wide")
st.title("🎧 Audio Analyzer — Simple (Upload → Analyze → Chat)")

st.markdown(
    """
    1) Upload an audio file (mp3/wav/flac/m4a).  
    2) App uploads file to Gemini (simple upload).  
    3) Gemini analyzes the file and returns a JSON.  
    4) You can chat / ask questions about the audio using the analysis context.
    """
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Gemini model", options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], index=0)
    st.markdown("---")
    st.info("Set GOOGLE_API_KEY in your environment (.env works for local).")

# --- Helpers ---
def configure_gemini_api():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Set it in environment or .env and restart the app.")
        st.stop()
    genai.configure(api_key=api_key)

def upload_file_with_retries(path, max_retries=3, initial_delay=3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            # Simple upload call expected by many genai SDK versions
            return genai.upload_file(path=path)
        except Exception as e:
            last_exc = e
            # For transient errors, retry
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                st.warning(f"Upload failed. Retrying in {delay} seconds... (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                raise last_exc
    raise last_exc

def wait_for_processing(file_ref, poll_interval=4, timeout=300):
    start = time.time()
    while True:
        # refresh file state
        try:
            file_ref = genai.get_file(getattr(file_ref, "name", file_ref))
        except Exception as e:
            raise RuntimeError(f"Failed to poll file state: {e}") from e
        state = getattr(file_ref.state, "name", None)
        # When not PROCESSING anymore, return
        if state != "PROCESSING":
            return file_ref
        if time.time() - start > timeout:
            raise TimeoutError("File processing timed out.")
        time.sleep(poll_interval)

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return None

# --- UI / Flow ---
uploaded = st.file_uploader("Drop your audio file here", type=["mp3", "wav", "flac", "m4a"], accept_multiple_files=False)

if uploaded is not None:
    st.audio(uploaded)
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    st.markdown(f"**File:** {uploaded.name} — {os.path.getsize(tmp.name)//1024} KB")

    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat" not in st.session_state:
        st.session_state.chat = None

    if st.button("Analyze with Gemini") or st.session_state.analysis:
        configure_gemini_api()

        if st.session_state.analysis is None:
            with st.spinner("Uploading and processing audio (may take a while)..."):
                try:
                    file_ref = upload_file_with_retries(tmp.name)
                    st.session_state.file_ref = file_ref
                    file_ref = wait_for_processing(file_ref, poll_interval=4, timeout=600)
                except Exception as e:
                    st.error(f"File processing error: {e}")
                    st.stop()

            st.success("Upload + processing complete. Requesting analysis from Gemini...")

            prompt = """
            You are an expert audio analyst. Analyze the provided audio file and return a single JSON object with these keys:
            - transcription: full transcript (string)
            - sentiment_analysis: {score: float (-1..1), label: string, justification: string}
            - emotion_analysis: {label: string, intensity: float (0..1), justification: string}
            - segments: list of {start_time: float, end_time: float, text: string, sentiment: {...}, emotion: {...}}
            - noise_analysis: {level: string, description: string}
            - scene_prediction: string
            - topics: [string]
            - focus_summary: string
            - sound_events: [{start_time, end_time, event_description}]
            - confidence: float (0..1)
            ONLY OUTPUT VALID JSON (no extra commentary).
            """

            try:
                model = genai.GenerativeModel(model_name=model_name, generation_config={"response_mime_type": "application/json"})
                with st.spinner("Generating analysis from Gemini (could take a few minutes)..."):
                    response = model.generate_content([prompt, file_ref], request_options={"timeout": 1200})
                text = getattr(response, "text", None) or str(response)
                parsed = extract_json_from_text(text)
                if parsed is None:
                    st.error("Gemini returned non-JSON or unparsable output. See raw response below.")
                    st.subheader("Raw response")
                    st.code(text)
                    st.stop()
                st.session_state.analysis = parsed
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        # Display results
        analysis = st.session_state.analysis
        st.header("Transcription")
        st.write(analysis.get("transcription", ""))

        st.header("Overall Sentiment")
        overall = analysis.get("sentiment_analysis", {})
        st.metric(label="Sentiment", value=overall.get("label", "N/A"), delta=f"Score: {overall.get('score', 'N/A')}")
        st.write(overall.get("justification", ""))

        st.header("Overall Emotion")
        emotion = analysis.get("emotion_analysis", {})
        st.metric(label="Emotion", value=emotion.get("label", "N/A"), delta=f"Intensity: {emotion.get('intensity', 'N/A')}")
        st.write(emotion.get("justification", ""))

        st.header("Noise & Scene")
        na = analysis.get("noise_analysis", {})
        st.write(f"Level: {na.get('level','N/A')}")
        st.write(f"Description: {na.get('description','N/A')}")
        st.write(f"Scene: {analysis.get('scene_prediction','N/A')}")

        st.header("Sound Events")
        sound_events = analysis.get("sound_events", [])
        if sound_events:
            st.dataframe(pd.DataFrame(sound_events))
        else:
            st.write("No sound events detected.")

        st.download_button("Download analysis (JSON)", data=json.dumps(analysis, indent=2), file_name=f"analysis_{uploaded.name}.json", mime="application/json")

        # Chat
        st.markdown("---")
        st.header("Chat about this audio")
        if st.session_state.chat is None:
            chat_model = genai.GenerativeModel(model_name=model_name)
            initial_prompt = f"You are a helpful assistant. Use the following JSON analysis to answer user questions:\n{json.dumps(st.session_state.analysis, indent=2)}"
            st.session_state.chat = chat_model.start_chat(history=[{"role":"user","parts":[initial_prompt]},{"role":"model","parts":["Ready."]}])
            st.session_state.messages.append({"role":"assistant","content":"I have the analysis. Ask me anything about the audio."})

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Ask a question about the audio..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.spinner("Gemini is thinking..."):
                response = st.session_state.chat.send_message(prompt)
            st.chat_message("assistant").markdown(getattr(response, "text", str(response)))
            st.session_state.messages.append({"role":"assistant","content":getattr(response, "text", str(response))})

else:
    st.info("Upload an audio file (mp3/wav/flac/m4a) to begin.")

st.markdown("---")
st.caption("Simple flow: upload -> analyze -> chat. Keep GOOGLE_API_KEY safe.")
