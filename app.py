import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Fruit Classifier AI", page_icon="🍎", layout="centered")

st.title("🍓 පලතුරු හඳුනාගන්නා AI වෙබ් ඇප් එක")
st.write("ඔයාගේ පලතුරු පින්තූරය Upload කරන්න. AI එක ඒක මොකක්ද කියලා නිවැරදිව කියයි!")

# Model එක සහ Class Names Load කරගැනීම
@st.cache_resource
def load_my_model():
    # සාදාගත් Model එක load කිරීම
    model = tf.keras.models.load_model("student_mobilenetv2_transfer_learning.keras")
    
    # Json file එකෙන් පලතුරු වර්ග 10 ලෝඩ් කරගැනීම
    with open("class_names.json", "r") as f:
        classes = json.load(f)
    return model, classes

try:
    model, class_names = load_my_model()
    st.success("AI Model එක සාර්ථකව සම්බන්ධ කලා! ✅")
except Exception as e:
    st.error("Model එක ලෝඩ් කරන්න බැරි වුණා! කරුණාකර file එක තියෙනවාදැයි බලන්න. ❌")

# පින්තූරයක් Upload කිරීමට ඉඩ දීම
uploaded_file = st.file_uploader("පලතුරක පින්තූරයක් Upload කරන්න...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='ඔයා Upload කරපු පින්තූරය', use_container_width=True)
    
    with st.spinner("AI එක පරීක්ෂා කරමින් පවතී... 🔄"):
        # Image එක AI එකට ගැලපෙන සේ සැකසීම (160x160)
        img_resized = image.resize((160, 160))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)
        
        # Predict කිරීම
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
    
    # ප්‍රතිඵලය ලස්සනට පෙන්වීම
    st.markdown(f"### 🎯 මේක බොහෝ දුරට: **{predicted_class.upper()}** එකක්.")
    st.progress(int(confidence))
    st.write(f"📊 නිවැරදිතාවයේ ප්‍රතිශතය: {confidence:.2f}%")

---

# ============================================================
# 3.2 Requirements File එක සෑදීම (requirements.txt)
# ============================================================

%%writefile requirements.txt
streamlit
tensorflow
numpy
Pillow
