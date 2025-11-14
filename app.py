import io

import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder

# ---------------------------------------------------------------------
# Streamlit Cloud config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Fort Siloso Voice Guide",
    page_icon="🎙️",
    layout="centered",
)

# Use Streamlit secrets for the API key (set in Streamlit Cloud UI)
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error(
        "OPENAI_API_KEY is not set.\n\n"
        "On Streamlit Cloud, go to: **Settings → Secrets** and add:\n"
        "OPENAI_API_KEY = your_real_key_here"
    )
    st.stop()

# ---------------------------------------------------------------------
# Session state for chat history
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable, friendly voice guide for Fort Siloso "
                "on Sentosa Island, Singapore.\n\n"
                "Guidelines:\n"
                "- Answer only questions related to Fort Siloso, Sentosa history, "
                "WWII coastal defence, the tunnels, guns, exhibits and visitor experience.\n"
                "- If the user asks about something unrelated, reply briefly "
                "that you only answer questions about Fort Siloso.\n"
                "- Prefer short, spoken-style answers: 2–4 sentences, "
                "clear and conversational.\n"
                "- If the user asks for highly time-sensitive info "
                "(ticket prices, exact opening hours, special events), "
                "give general guidance and ask them to check the official website "
                "or on-site signage for exact details."
            ),
        }
    ]

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🎙️ Fort Siloso Voice Guide")
st.markdown(
    "1. Press the microphone and **ask a question out loud** about Fort Siloso.\n"
    "2. Release to stop recording.\n"
    "3. The guide will **speak back** with an answer."
)

with st.expander("About this guide", expanded=False):
    st.write(
        "This demo uses OpenAI’s speech-to-text and text-to-speech models. "
        "Voices are AI-generated."
    )

voice = st.selectbox(
    "Voice for the guide",
    options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    index=0,
)

st.markdown("### 1. Record your question")

audio_bytes = audio_recorder(
    text="Click to start / stop recording",
    recording_color="#ff4b4b",
    neutral_color="#6c757d",
    icon_size="3x",
)

if audio_bytes:
    st.success("Recording captured! Scroll down to send it.")
    st.audio(audio_bytes, format="audio/wav")

st.markdown("### 2. Send to Fort Siloso guide")

col1, col2 = st.columns(2)
with col1:
    send_now = st.button("Ask in voice", type="primary", disabled=not bool(audio_bytes))
with col2:
    clear_chat = st.button("Reset conversation")

if clear_chat:
    st.session_state.messages = st.session_state.messages[:1]  # keep only system prompt
    st.rerun()

# ---------------------------------------------------------------------
# Core pipeline: audio → text → LLM → audio
# ---------------------------------------------------------------------
def transcribe_question(audio_bytes: bytes) -> str:
    """Use OpenAI STT to transcribe the user's recorded question."""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "question.wav"

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",  # fast, good STT
        file=audio_file,
    )
    return transcript.text


def get_fort_siloso_answer(question_text: str) -> str:
    """Use LLM to answer as Fort Siloso guide, using chat history."""
    st.session_state.messages.append({"role": "user", "content": question_text})

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # bump to gpt-4.1 if you want more power
        messages=st.session_state.messages,
        temperature=0.4,
    )

    answer = completion.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    return answer


def synthesize_answer(answer_text: str, voice_name: str) -> bytes:
    """Use OpenAI TTS to generate spoken answer."""
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice_name,
        input=answer_text,
    )
    audio_out = speech.read()  # returns MP3 bytes
    return audio_out


# ---------------------------------------------------------------------
# Run the full flow
# ---------------------------------------------------------------------
if send_now and audio_bytes:
    with st.spinner("Transcribing your question..."):
        try:
            question_text = transcribe_question(audio_bytes)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

    st.markdown("**You said:**")
    st.write(question_text)

    with st.spinner("Thinking about Fort Siloso..."):
        try:
            answer_text = get_fort_siloso_answer(question_text)
        except Exception as e:
            st.error(f"LLM failed: {e}")
            st.stop()

    st.markdown("**Guide says (text):**")
    st.write(answer_text)

    with st.spinner("Generating spoken reply..."):
        try:
            answer_audio = synthesize_answer(answer_text, voice)
        except Exception as e:
            st.error(f"TTS failed: {e}")
            st.stop()

    st.markdown("### 3. Listen to the answer")
    st.audio(answer_audio, format="audio/mp3")

# ---------------------------------------------------------------------
# Chat history panel
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("Conversation history")

for msg in st.session_state.messages[1:]:  # skip system
    role = "👤 You" if msg["role"] == "user" else "🎧 Guide"
    st.markdown(f"**{role}:** {msg['content']}")
