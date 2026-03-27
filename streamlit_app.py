import streamlit as st
import os
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: white;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}
.card {
    background-color: #1E222A;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("  Navigation")
    st.markdown("---")

    page = st.radio(
        "Go to",
        ["Dashboard", "Capture Face", "Train Model", "Recognition"]
    )

    st.markdown("---")
    st.caption("AI Face Recognition System")

# -----------------------------
# DASHBOARD
# -----------------------------
if page == "Dashboard":
    st.title("  Face Recognition System")
    st.markdown("### Smart Access Control using AI")

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class='card'>
    <h3>📸 Capture</h3>
    <p>Register new users</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class='card'>
    <h3>🧠 Train</h3>
    <p>Train the AI model</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class='card'>
    <h3>🎥 Recognize</h3>
    <p>Start face recognition</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Follow steps: Capture → Train → Recognize")

# -----------------------------
# CAPTURE FACE
# -----------------------------
elif page == "Capture Face":
    st.title("📸 Capture Face")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    username = st.text_input("Enter Username")

    if st.button("Start Capture"):
        if username.strip() == "":
            st.warning("⚠️ Please enter a username")
        else:
            st.info("Opening camera...")

            # This allows input to script
            command = f'echo {username} | python capture_image.py'
            os.system(command)

            st.success("✅ Capture Completed")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TRAIN MODEL
# -----------------------------
elif page == "Train Model":
    st.title("🧠 Train Model")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if st.button("Train Model"):
        st.info("Training model...")

        os.system("python train_model.py")

        st.success("✅ Training Completed")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RECOGNITION
# -----------------------------
elif page == "Recognition":
    st.title("🎥 Face Recognition")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.warning("Look at the camera. Press 'q' to stop.")

    if st.button("Start Recognition"):
        st.info("Starting camera...")

        # THIS FIXES YOUR PROBLEM
        os.system("python access_control.py")

        st.success(f"System executed at {datetime.now().strftime('%H:%M:%S')}")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("© 2026 AI Face Recognition System | OpenCV + Streamlit")