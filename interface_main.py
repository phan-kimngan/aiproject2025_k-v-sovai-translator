import streamlit as st
from gtts import gTTS
import pandas as pd
from datetime import datetime
import requests
#from predict import translate_kor_to_vie
#from predict_2 import translate_vie_to_kor
    
import speech_recognition as sr

API_kor_to_vie = ".ngrok-free.dev link+/kor2vie"
API_vie_to_kor = ".ngrok-free.dev link+/vie2kor"
# ==============================
# VOICE INPUT
# ==============================
def record_and_transcribe(language="vi"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ ...")
        audio = r.listen(source)

    try:
        st.success("⏳ ...")
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        st.error("❗")
        return ""
    except sr.RequestError:
        st.error("❗")
        return ""


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="K-V SovAI Translator",
    page_icon="🇰🇷🇻🇳",
    layout="centered"
)

# ==============================
# SESSION STATE
# ==============================
if "mode" not in st.session_state:
    st.session_state.mode = "vi_to_kr"

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "update_trigger" not in st.session_state:
    st.session_state.update_trigger = 0

if "translation" not in st.session_state:
    st.session_state.translation = ""

if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# CSS (giữ nguyên của bạn)
# ==============================

st.markdown("""<style>.swap-container { position: relative; height: 200px; display: flex; align-items: center; justify-content: center; }</style>""", unsafe_allow_html=True)
st.markdown("""<style>body, .stApp {background: linear-gradient(145deg, #C9C3FF, #B8D7FF) !important; color: #FFFFFF;} h2 {color: #FFFFFF !important; font-weight: 800; text-shadow: 0px 1px 4px rgba(0,0,0,0.18);} textarea {background-color: #FFFFFF !important; color: #1E1E1E !important; border: 1px solid rgba(255,255,255,0.6) !important; border-radius: 14px !important; padding: 12px !important; box-shadow: 0 3px 6px rgba(0,0,0,0.08);} .stButton > button {background-color: rgba(255,255,255,0.55) !important; color: #1E1E1E !important; font-weight: 600 !important; border: 1px solid rgba(255,255,255,0.8) !important; padding: 10px 22px; border-radius: 10px; backdrop-filter: blur(8px); box-shadow: 0 3px 6px rgba(0,0,0,0.15);} .stButton > button:hover {background-color: rgba(255,255,255,0.8) !important; border: 1px solid #FFFFFF !important; transform: scale(1.07); box-shadow: 0 4px 10px rgba(0,0,0,0.22);} h2 {color: #111111 !important;}</style>""", unsafe_allow_html=True)
st.markdown("""<style>.stApp header, .stApp div[data-testid="stDecoration"]{display:none !important;}</style>""", unsafe_allow_html=True)


# 4. HEADER
st.markdown(
    """
    <h2 style='text-align:center; color:#1E3A8A;'>
        🇰🇷 K-V SovAI Translator 🇻🇳
    </h2>
    """,
    unsafe_allow_html=True
)

# ==============================
# 5. LAYOUT
# ==============================
col1, col_center, col2 = st.columns([2, 0.5, 2])

# ==============================
# 6. SWAP BUTTON
# ==============================
with col_center:
    st.markdown("<div class='swap-container'>", unsafe_allow_html=True)
    swap_clicked = st.button("↔️", key="swap_button")
    st.markdown("</div>", unsafe_allow_html=True)

if swap_clicked:
    old_in = st.session_state.input_text
    old_out = st.session_state.translation

    st.session_state.input_text = old_out
    st.session_state.translation = old_in

    st.session_state.mode = "kr_to_vi" if st.session_state.mode == "vi_to_kr" else "vi_to_kr"

    st.session_state.update_trigger += 1
    st.rerun()


# ==============================
# 7. LABEL CONFIG
# ==============================
mode = st.session_state.mode
if mode == "vi_to_kr":
    left_label = "Vietnamese"
    right_label = "Korean"
    src_tts_lang = "vi"
    tgt_tts_lang = "ko"
    translate_func = API_vie_to_kor
else:
    left_label = "Korean"
    right_label = "Vietnamese"
    src_tts_lang = "ko"
    tgt_tts_lang = "vi"
    translate_func = API_kor_to_vie


# ==============================
# 8. LEFT PANEL
# ==============================
with col1:
    st.markdown(f"<div style='color: #000; font-size:25px; font-weight:600;'>{left_label}</div>", unsafe_allow_html=True)

    input_text = st.text_area(
        "",
        st.session_state.input_text,
        height=200,
        key=f"input_widget_{st.session_state.update_trigger}"
    )

    st.session_state.input_text = input_text

    colA, colB = st.columns([3, 1])

    with colA:
        if st.button("🔊", key="speak_input"):
            if input_text.strip():
                tts = gTTS(input_text, lang=src_tts_lang)
                tts.save("input_tts.mp3")
                st.audio("input_tts.mp3")

    with colB:
        if st.button("🎤", key="voice_input"):
            text = record_and_transcribe(language=src_tts_lang)
            if text.strip():
                st.session_state.input_text = text
                st.session_state.update_trigger += 1
                st.success("✔")
                st.rerun()
            else:
                st.warning("⚠️")


# ==============================
# 9. RIGHT PANEL
# ==============================
with col2:
    st.markdown(f"<div style='color: #000; font-size:25px; font-weight:600;'>{right_label}</div>", unsafe_allow_html=True)

    st.text_area(
        " ",
        st.session_state.translation,
        height=200,
        key="output_box"
    )

    if st.button("🔊", key="speak_output"):
        if st.session_state.translation.strip():
            tts = gTTS(st.session_state.translation, lang=tgt_tts_lang)
            tts.save("output_tts.mp3")
            st.audio("output_tts.mp3")


# ==============================
# 10. TRANSLATE BUTTON
# ==============================
if st.button("🌐 Translate", use_container_width=True):
    text = st.session_state.input_text.strip()
    if text:
        with st.spinner("Translating... ⏳"):
            #result = translate_func(text)
            result = requests.get(translate_func, params={"text": text})
            result = result.json()["result"]
            st.session_state.translation = result

            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": st.session_state.mode,
                "src": text,
                "tgt": result
            })

        st.rerun()


# ==============================
# 12. HISTORY SECTION
# ==============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='color:#000; font-size:25px; font-weight:600;'>🕘 History</div>", unsafe_allow_html=True)

colH1, colH2 = st.columns([1, 1])

with colH1:
    if st.button("🧹 Clear all history"):
        st.session_state.history = []
        st.rerun()

with colH2:
    if st.button("💾 Export history to CSV"):
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            df.to_csv("translation_history.csv", index=False)
            with open("translation_history.csv", "rb") as f:
                st.download_button("⬇️ Download CSV file", f, file_name="translation_history.csv")
        else:
            st.warning("⚠️ Không có dữ liệu để export")


for item in reversed(st.session_state.history):
    direction = "🇻🇳 Vietnamese → 🇰🇷 Korean" if item["mode"] == "vi_to_kr" else "🇰🇷 Korean → 🇻🇳 Vietnamese"

    st.markdown(
        f"""
        <div style='padding:8px; background:rgba(255,255,255,0.45);
            border:1px solid rgba(0,0,0,0.18); border-radius:10px; margin-bottom:8px;
            font-size:13px; color:#000'>
            <span style='font-size:11px'>{item['time']}</span><br>
            <b>{direction}</b><br><br>
            <b>Input:</b><br>{item['src']}<br><br>
            <b>Output:</b><br>{item['tgt']}
        </div>
        """, unsafe_allow_html=True
    )
# ==============================
# 11. FOOTER
# ==============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>© 2025 K-V SovAI Translator</p>", unsafe_allow_html=True)
