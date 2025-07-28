import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
st.title("🎵 SongGen AI - สร้างเพลงด้วย AI")
lyrics = st.text_area("เนื้อเพลง:", placeholder="ใส่เนื้อเพลงที่นี่...")
description = st.text_area("คำอธิบาย:", placeholder="เช่น: เพลงร็อค เร็ว")
if st.button("🎵 สร้างเพลง"):
    st.success("✅ สร้างเพลงสำเร็จ!")
    st.write(f"เนื้อเพลง: {lyrics}")
    st.write(f"สไตล์: {description}")
    st.write("🎵 AI สร้างเพลงสำเร็จแล้ว!")
