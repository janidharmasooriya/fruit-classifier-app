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
    # GitHub එකේ තියෙන නම හරියටම මෙතන ලියන්න (වරහන් තිබ්බොත් ඒ විදිහටම)
    model_name = "student_mobilenetv2_transfer_learning.keras"
    
    model = tf.keras.models.load_model(model_name)
    
    with open("class_names.json", "r") as f:
        classes = json.load(f)
    return model, classes

# ගෝලීය (Global) variables මුලින්ම හිස්ව තබා ගැනීම
model = None
class_names = None

try:
    model, class_names = load_my_model()
    st.success("AI Model එක සාර්ථකව සම්බන්ධ කලා! ✅")
except Exception as e:
    st.error("Model එක ලෝඩ් කරන්න බැරි වුණා! කරුණාකර file එක තියෙනවාදැයි බලන්න. ❌")
    st.info("ඔයාගේ GitHub එකේ තියෙන .keras ෆයිල් එකේ නම සහ app.py හි 17 වෙනි පේළියේ තියෙන නම සමානදැයි බලන්න.")

# පින්තූරයක් Upload කිරීමට ඉඩ දීම
uploaded_file = st.file_uploader("පලතුරක පින්තූරයක් Upload කරන්න...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # අලුත් Streamlit සංස්කරණයට ගැළපෙන සේ width='stretch' යෙදීම
    st.image(image, caption='ඔයා Upload කරපු පින්තූරය', width='stretch')
    
    # Model එක සාර්ථකව ලෝඩ් වී ඇත්නම් පමණක් Predict කිරීම
    if model is not None and class_names is not None:
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
    else:
        st.warning("කණගාටුයි! AI Model එක හරියට Load නොවූ නිසා පරීක්ෂා කිරීමට නොහැක. ⚠️")
