import streamlit as st
st.title("🎵 SongGen AI - สร้างเพลงด้วย AI")
st.write("สร้างเพลงจากเนื้อเพลงและคำอธิบายด้วย AI")
lyrics = st.text_area("เนื้อเพลง (Lyrics):", placeholder="ใส่เนื้อเพลงที่นี่...")
description = st.text_area("คำอธิบายสไตล์เพลง:", placeholder="เช่น: เพลงร็อค เร็ว ใช้กีตาร์ไฟฟ้า")
if st.button("🎵 สร้างเพลง"):
    st.success("✅ สร้างเพลงสำเร็จ!")
    st.write(f"เนื้อเพลง: {lyrics}")
    st.write(f"สไตล์: {description}")
