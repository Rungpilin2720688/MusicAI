import streamlit as st
import torch
from songgen import (
    VoiceBpeTokenizer,
    SongGenMixedForConditionalGeneration,
    SongGenProcessor
)
import soundfile as sf
import os
from datetime import datetime
import tempfile

# Page config
st.set_page_config(
    page_title="SongGen AI - สร้างเพลงด้วย AI",
    page_icon="🎵",
    layout="wide"
)

# Title
st.title("🎵 SongGen AI - สร้างเพลงด้วย AI")
st.markdown("สร้างเพลงจากเนื้อเพลงและคำอธิบายด้วย AI")

# Initialize model (only once)
@st.cache_resource
def load_model():
    with st.spinner("กำลังโหลด AI Model..."):
        ckpt_path = "LiuZH-19/SongGen_mixed_pro"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = SongGenMixedForConditionalGeneration.from_pretrained(
            ckpt_path,
            attn_implementation='sdpa').to(device)
        processor = SongGenProcessor(ckpt_path, device)
        return model, processor

# Load model
model, processor = load_model()

# Input form
with st.form("song_form"):
    st.subheader("📝 ใส่ข้อมูลเพลง")
    
    lyrics = st.text_area(
        "เนื้อเพลง (Lyrics):",
        placeholder="ใส่เนื้อเพลงที่นี่...",
        height=150
    )
    
    description = st.text_area(
        "คำอธิบายสไตล์เพลง (Music Description):",
        placeholder="เช่น: เพลงร็อค เร็ว ใช้กีตาร์ไฟฟ้า",
        height=100
    )
    
    music_type = st.selectbox(
        "ประเภทเพลง:",
        ["Pop", "Rock", "Jazz", "Classical", "Electronic", "Country", "Hip Hop", "อื่นๆ"]
    )
    
    submitted = st.form_submit_button("🎵 สร้างเพลง", type="primary")

# Generate song
if submitted:
    if not lyrics or not description:
        st.error("กรุณาใส่เนื้อเพลงและคำอธิบาย")
    else:
        with st.spinner("กำลังสร้างเพลง... กรุณารอสักครู่ (30-60 วินาที)"):
            try:
                # Generate the song
                model_inputs = processor(text=description, lyrics=lyrics)
                generation = model.generate(**model_inputs, do_sample=True)
                
                # Save the generated audio
                audio_arr = generation.cpu().numpy().squeeze()
                
                # Create temporary file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(temp_file.name, audio_arr, model.config.sampling_rate)
                
                st.success("✅ สร้างเพลงสำเร็จ!")
                
                # Display audio player
                st.subheader("🎧 ฟังเพลง")
                with open(temp_file.name, 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/wav')
                
                # Download button
                st.download_button(
                    label="📥 ดาวน์โหลดเพลง",
                    data=audio_file.read(),
                    file_name=f"songgen_{timestamp}.wav",
                    mime="audio/wav"
                )
                
                # Clean up
                os.unlink(temp_file.name)
                
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ ข้อมูล")
    st.markdown("""
    **SongGen AI** เป็น AI ที่สามารถสร้างเพลงจาก:
    - เนื้อเพลง (Lyrics)
    - คำอธิบายสไตล์เพลง
    
    **หมายเหตุ:**
    - การสร้างเพลงครั้งแรกจะใช้เวลานาน
    - เพลงที่สร้างได้ยาวสูงสุด 30 วินาที
    - รองรับเฉพาะภาษาอังกฤษ
    """)
    
    st.header("🎵 ตัวอย่าง")
    st.markdown("""
    **เนื้อเพลง:**
    ```
    I love you so much
    You make my heart sing
    Together forever
    ```
    
    **คำอธิบาย:**
    ```
    เพลงป็อป โรแมนติก ใช้เปียโน
    ```
    """)

# Footer
st.markdown("---")
st.markdown("🎵 SongGen AI - สร้างเพลงด้วย AI | Powered by Streamlit") 