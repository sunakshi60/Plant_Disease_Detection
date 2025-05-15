import streamlit as st
import tensorflow as tf
import numpy as np

# Custom CSS (no hover, more unique UI)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: #f4f9f6;
        margin: 0;
        padding: 0;
        overflow-x: hidden;
    }

    .main-header {
         background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
        padding: 2rem 1rem;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
        margin: 2rem 0 0 0;  /* Add top margin */
        border-radius: 12px 12px 0 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        white-space: nowrap;  /* Ensure heading stays in one line */
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .footer {
        background: #11998e;
        color: white;
        text-align: center;
        padding: 1.5rem;
        font-size: 1rem;
        width: 100%;
        box-sizing: border-box;
        margin-top: 2rem;
        border-radius: 0 0 16px 16px;
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.1);
    }

    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.6rem 2rem;
        border-radius: 12px 12px 0 0;
        margin: 0 0.2rem;
        background-color: #d4f4e3;
        color: #115e59;
    }

    .info-card {
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        padding: 2rem;
        margin: auto;
        max-width: 800px;
    }

    h1, h3 {
        font-size: 2rem;
        font-weight: 700;
    }

    h4 {
        color: #11998e;
        font-weight: 600;
        margin-top: 1.5rem;
    }

    p, ul, ol {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #444;
    }

    a {
        color: #11998e;
        text-decoration: underline;
    }

    button {
        background-color: #11998e;
        color: white;
        padding: 1rem 2rem;
        font-size: 1rem;
        border-radius: 12px;
        border: none;
        cursor: pointer;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Custom header with tagline
st.markdown(
    """
    <div class="main-header">
        <h1>🌿 Plant Disease Recognition System</h1>
        <p style="margin-top: 0.5rem;">Detect Early. Protect Fully. Grow Confidently.</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# TensorFlow Model Prediction
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model('trained_plant_disease_model.keras')
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        input_arr = input_arr / 255.0
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions[0])
        return result_index
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Tabs Navigation
pages = [" Home", "About", " Disease Recognition","Disease-Info","Prophylaxis"]
tabs = st.tabs(pages)

tab_home, tab_about, tab_disease ,tab_info,tab_steps= tabs

with tab_home:
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
    <div class="info-card">
    <h3 style="text-align:center;">Welcome to the Plant Disease Recognition System!🌱</h3>
    <p style="font-size:1.05rem;text-align:justify;">Identify diseases in plant leaves using our advanced deep learning model. Accurate, fast, and reliable results at your fingertips.</p>

    <h4>📌 How to Use</h4>
    <ol>
    <li><strong>Upload</strong> an image in the 'Disease Recognition' tab</li>
    <li><strong>Process</strong> the image using our model</li>
    <li><strong>Result</strong> displayed with health status</li>
    </ol>

    <h4>🎯 Benefits</h4>
    <ul>
    <li>Quick identification of plant diseases</li>
    <li>Prevents crop loss</li>
    <li>Empowers farmers and researchers</li>
    </ul>

    <h4>📧 Contact</h4>
    <p>Mail us: <a href="mailto:support@plantai.com">support@plantai.com</a></p>
    </div>
    """, unsafe_allow_html=True)

with tab_about:
    st.markdown("""
    <div class="info-card">
    <h3 style="text-align:center;">About the Dataset</h3>
    <p style="text-align:justify;">Our model is trained using a robust dataset of 87K+ images spanning 38 classes. Data is split into training, validation, and test sets.</p>

    <h4>🔍 Dataset Breakdown</h4>
    <ul>
    <li><b>Training:</b> 70,295 images</li>
    <li><b>Validation:</b> 17,572 images</li>
    <li><b>Test:</b> 33 images</li>
    </ul>

    <h4>🌍 Project Vision</h4>
    <ul>
    <li>Enable early detection for farmers</li>
    <li>Integrate AI in agriculture</li>
    <li>Raise crop health awareness</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab_disease:
    st.header("🌿Upload a Leaf Image to Detect Disease")
    test_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
        if st.button("Predict Disease"):
            with st.spinner("Analyzing image..."):
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(getattr(test_image, 'name', 'img.jpg'))[1]) as tmp_file:
                    tmp_file.write(test_image.read() if hasattr(test_image, 'read') else test_image.getvalue())
                    tmp_file_path = tmp_file.name
                result_index = model_prediction(tmp_file_path)
                if result_index is not None:
                    class_name = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
                    pred_class = class_name[result_index]
                    if 'healthy' in pred_class.lower():
                        st.success(f"✅ The plant is HEALTHY! ({pred_class.replace('_', ' ')})")
                        st.balloons()
                    else:
                        st.error(f"⚠️ The plant is DISEASED! ({pred_class.replace('_', ' ')})")
    st.markdown("""</div>""", unsafe_allow_html=True)


with tab_info:
    st.markdown("""
    <div class="info-card">
    <h3 style="text-align:center;">🌿 Common Plant Diseases & Remedies</h3>
    <p style="text-align:justify;">Below are some of the plant diseases included in our training and testing dataset along with information and treatment steps.</p>

    <h4>🍎 Apple Scab</h4>
    <p><b>Description:</b> Caused by the fungus *Venturia inaequalis*, leads to dark, scabby lesions on leaves and fruit.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Use resistant apple varieties</li>
        <li>Prune infected twigs and leaves</li>
        <li>Apply fungicides like captan or mancozeb</li>
    </ul>

    <h4>🍇 Grape Black Rot</h4>
    <p><b>Description:</b> Fungal disease causing dark lesions on leaves and rotting grapes.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Remove and destroy mummified fruits</li>
        <li>Prune infected canes</li>
        <li>Apply fungicides such as myclobutanil</li>
    </ul>

    <h4>🍅 Tomato Early Blight</h4>
    <p><b>Description:</b> Caused by *Alternaria solani*, leads to brown concentric rings on lower leaves.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Remove affected leaves</li>
        <li>Use crop rotation</li>
        <li>Spray fungicides like chlorothalonil or copper-based products</li>
    </ul>

    <h4>🌽 Corn Northern Leaf Blight</h4>
    <p><b>Description:</b> Caused by *Exserohilum turcicum*, produces long gray-green lesions on leaves.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Plant resistant hybrids</li>
        <li>Rotate with non-host crops</li>
        <li>Apply fungicides like azoxystrobin</li>
    </ul>

    <h4>🍑 Peach Bacterial Spot</h4>
    <p><b>Description:</b> Bacterial infection causing black spots on fruit and leaf drop.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Use certified disease-free trees</li>
        <li>Apply copper sprays in the dormant season</li>
        <li>Avoid overhead irrigation</li>
    </ul>

    <h4>🌶️ Bell Pepper Bacterial Spot</h4>
    <p><b>Description:</b> Bacterial infection showing water-soaked spots on leaves and fruit.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Use copper or copper + mancozeb sprays</li>
        <li>Practice crop rotation</li>
        <li>Remove infected plant debris</li>
    </ul>

    <h4>🍓 Strawberry Leaf Scorch</h4>
    <p><b>Description:</b> Causes reddish-brown lesions on leaves leading to dieback.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Plant resistant varieties</li>
        <li>Remove infected foliage</li>
        <li>Apply preventative fungicides</li>
    </ul>

    <h4>🍅 Tomato Leaf Mold</h4>
    <p><b>Description:</b> Caused by *Passalora fulva*, appears as yellow spots on upper leaf surfaces with mold underneath.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Ensure good ventilation and spacing</li>
        <li>Apply fungicides like chlorothalonil or copper</li>
        <li>Remove infected leaves promptly</li>
    </ul>

    <h4>🍠 Potato Late Blight</h4>
    <p><b>Description:</b> Caused by *Phytophthora infestans*, responsible for large, dark leaf lesions and tuber rot.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Destroy infected plants and tubers</li>
        <li>Apply fungicides like metalaxyl</li>
        <li>Ensure proper field drainage</li>
    </ul>

    <h4>🍋 Citrus Greening (HLB)</h4>
    <p><b>Description:</b> Caused by a bacterium spread by the Asian citrus psyllid, leads to yellow shoots and fruit drop.</p>
    <p><b>Treatment:</b></p>
    <ul>
        <li>Remove and destroy infected trees</li>
        <li>Control psyllid vectors using insecticides</li>
        <li>Plant disease-free certified citrus saplings</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab_steps:
    st.markdown(
        """
<div>
<br>
<h4>🟢 Prominent Steps to Get Rid of it</h4>
    <ol>
        <li><b>Identify the Disease Early:</b> Use AI tools, observe symptoms like spots or discoloration.</li>
        <li><b>Remove Infected Parts:</b> Prune and destroy affected leaves, stems, and fruits.</li>
        <li><b>Apply Suitable Treatment:</b> Use fungicides, bactericides, or pest control depending on disease type.</li>
        <li><b>Use Resistant Varieties:</b> Choose plants bred to resist common diseases.</li>
        <li><b>Practice Crop Rotation:</b> Prevent disease buildup in soil.</li>
        <li><b>Improve Plant Spacing & Airflow:</b> Reduces humidity and infection spread.</li>
        <li><b>Water Smartly:</b> Avoid wetting leaves; water early morning at the base.</li>
        <li><b>Maintain Soil Health:</b> Use compost, test soil pH and nutrients.</li>
        <li><b>Sanitize Tools and Equipment:</b> Disinfect tools after each use.</li>
        <li><b>Monitor Regularly:</b> Check plants weekly and act at first signs.</li>
        <li><b>Mulch to Control Weeds:</b> Mulching suppresses weed growth that may harbor pests.</li>
        <li><b>Support Plant Growth:</b> Use stakes or cages to keep leaves off the soil.</li>
        <li><b>Provide Adequate Sunlight:</b> Ensure plants get enough light to grow healthily.</li>
        <li><b>Remove Debris:</b> Clean up fallen leaves and fruits promptly.</li>
        <li><b>Avoid Overcrowding:</b> Give plants enough space to thrive.</li>
        <li><b>Inspect New Plants:</b> Quarantine and examine new plants before adding them to your garden.</li>
        <li><b>Use Organic Pesticides:</b> Prefer neem oil or soap-based sprays for milder infestations.</li>
        <li><b>Check Soil Drainage:</b> Prevent root rot by improving water flow.</li>
        <li><b>Educate Gardeners:</b> Share awareness about disease prevention among farmers and gardeners.</li>
        <li><b>Use Drip Irrigation:</b> Minimizes water contact with foliage, reducing fungal spread.</li>
    </ol>
</div>
        """,unsafe_allow_html=True
    )