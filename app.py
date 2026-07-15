import streamlit as st
import numpy as np
import joblib
import json
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

# --- NVIDIA LLM Configuration ---
_nvidia_client = None

# Ordered fallback list — probed live, all returned 200 OK, sorted by response time.
# Primary: meta/llama-4-maverick-17b-128e-instruct (Meta Llama 4, 0.4s)
_NVIDIA_MODELS = [
    "meta/llama-4-maverick-17b-128e-instruct",   # 0.4s ✅ primary
    "mistralai/mistral-nemotron",                 # 0.4s ✅ fallback 1
    "mistralai/mistral-small-4-119b-2603",        # 0.4s ✅ fallback 2
    "nvidia/nemotron-3-super-120b-a12b",          # 0.4s ✅ fallback 3
    "meta/llama-3.1-70b-instruct",               # 0.7s ✅ fallback 4 (proven stable)
]

def get_nvidia_client():
    """Singleton factory for the sync NVIDIA OpenAI-compatible client."""
    global _nvidia_client
    if not OPENAI_SDK_AVAILABLE:
        return None
    if _nvidia_client is None:
        api_key = os.getenv("NVIDIA_API_KEY", "")
        _nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=30.0,
        )
    return _nvidia_client


def get_disease_cure(disease_name: str, plant_name: str) -> str:
    """
    Calls the NVIDIA inference endpoint synchronously to get a structured
    treatment plan for the detected plant disease.
    Automatically falls back through _NVIDIA_MODELS on 503 / timeout errors.
    """
    client = get_nvidia_client()
    if client is None:
        return "⚠️ `openai` package not installed. Run: `pip install openai`"

    system_prompt = (
        "You are an expert plant pathologist and agricultural extension officer. Your task is to provide "
        "clear, accurate, and highly structured advice on curing and managing plant diseases. "
        "Break down your recommendations into distinct sections: Immediate Action, Organic/Biological Control, "
        "Chemical Control (if necessary), and Future Prevention. "
        "Keep explanations practical for a farmer or gardener."
    )

    user_prompt = f"""The image classification model has detected the following disease on a plant:
- **Plant Type:** {plant_name}
- **Detected Disease:** {disease_name}

Please provide a comprehensive guide on how to treat and cure this disease.
Format your response cleanly using Markdown headings and bullet points. Include:
1. A brief overview of how this disease damages the plant.
2. Biological or organic treatments.
3. Chemical solutions, specifying active ingredients if applicable.
4. Cultural practices to prevent recurrence."""

    last_error = "No models available."
    for model_id in _NVIDIA_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            # Retry with next model only on rate-limit / server-side errors
            if any(code in err_str for code in ["503", "502", "529", "timed out", "timeout"]):
                last_error = f"Model `{model_id}` unavailable (503/timeout), trying next..."
                continue
            # For any other error (auth, 404, etc.) fail immediately
            return f"⚠️ Error retrieving cure information from NVIDIA API: {err_str}"

    return f"⚠️ All NVIDIA models are currently overloaded. Please try again in a moment. (Last error: {last_error})"


# --- Page Configuration ---
st.set_page_config(
    page_title="Agri-AI Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_models():
    crop_model = None
    disease_model = None
    class_indices = None

    models_dir = 'models'
    crop_model_path = os.path.join(models_dir, 'crop_app.pkl')
    disease_model_path = os.path.join(models_dir, 'plant_disease_prediction_model.h5')
    class_indices_path = os.path.join(models_dir, 'class_indices.json')

    # 1. Load Crop Model
    try:
        crop_model = joblib.load(crop_model_path)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Crop model not found or failed to load: {e}")

    # 2. Load Disease Model
    if TF_AVAILABLE:
        try:
            disease_model = tf.keras.models.load_model(disease_model_path, compile=False)
        except Exception as e:
            st.sidebar.warning(f"⚠️ Disease model not found or failed to load: {e}")
    else:
        st.sidebar.warning("⚠️ TensorFlow is not installed. Disease detection is disabled.")

    # 3. Load Class Indices
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Class indices not found or failed to load: {e}")

    return crop_model, disease_model, class_indices


crop_model, disease_model, class_indices = load_models()


# --- Helper Functions ---
def preprocess_image(image):
    """
    Standard preprocessing for Keras image models.
    Resizes to 224x224, converts to numpy array, normalizes to [0,1],
    and expands dimensions to create a batch of 1.
    """
    image = image.resize((224, 224))
    img_array = np.array(image)

    # Handle images with an alpha channel (RGBA to RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def parse_label(raw_label: str):
    """
    Splits a raw class label of the form 'PlantName___DiseaseName'
    into (plant_name, disease_name) with human-readable formatting.

    Examples:
        'Tomato___Early_blight'          -> ('Tomato', 'Early blight')
        'Cherry_(including_sour)___...'  -> ('Cherry (including sour)', '...')
        'Apple___healthy'                -> ('Apple', 'Healthy')
    """
    if "___" in raw_label:
        plant_part, disease_part = raw_label.split("___", 1)
    else:
        plant_part = "Unknown Plant"
        disease_part = raw_label

    plant_name  = plant_part.replace("_", " ").strip()
    disease_name = disease_part.replace("_", " ").strip().title()
    return plant_name, disease_name


def clean_disease_name(name: str) -> str:
    """Cleans up the raw class name by replacing '___' and '_'."""
    name = name.replace("___", " - ")
    name = name.replace("_", " ")
    return name


# --- Sidebar Navigation ---
st.sidebar.title("🌿 Navigation")
st.sidebar.markdown("Select an AI tool below:")
page = st.sidebar.radio("", ["Crop Recommendation", "Disease Detection"])

st.sidebar.markdown("---")
st.sidebar.info("Upload models to the `models/` directory for this app to function fully.")

# --- Page 1: Crop Recommendation ---
if page == "Crop Recommendation":
    st.title("🌾 Crop Recommendation System")
    st.markdown("Enter the soil and weather parameters to get a data-driven crop recommendation.")

    with st.form("crop_form"):
        st.subheader("Field Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            N = st.number_input("Nitrogen (N)", value=0.0, format="%.2f", min_value=0.0)
            temperature = st.number_input("Temperature (°C)", value=0.0, format="%.2f")
            rainfall = st.number_input("Rainfall (mm)", value=0.0, format="%.2f", min_value=0.0)

        with col2:
            P = st.number_input("Phosphorus (P)", value=0.0, format="%.2f", min_value=0.0)
            humidity = st.number_input("Humidity (%)", value=0.0, format="%.2f", min_value=0.0, max_value=100.0)

        with col3:
            K = st.number_input("Potassium (K)", value=0.0, format="%.2f", min_value=0.0)
            ph = st.number_input("pH Level", value=0.0, format="%.2f", min_value=0.0, max_value=14.0)

        submit_button = st.form_submit_button("Predict Recommended Crop", type="primary")

    if submit_button:
        if crop_model is not None:
            input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            try:
                prediction = crop_model.predict(input_features)
                recommended_crop = prediction[0]
                st.success(f"### 🎉 Recommended Crop: **{recommended_crop.capitalize()}**")
                st.info("💡 **AI Insights:** (Coming Soon) Detailed insights about this crop, including soil requirements, planting tips, and market value will appear here.")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.error("Model is not loaded. Please ensure `models/crop_app.pkl` exists.")

# --- Page 2: Disease Detection ---
elif page == "Disease Detection":
    st.title("🔬 Plant Disease Detection")
    st.markdown("Upload a picture of a plant leaf to identify potential diseases and get a treatment plan.")

    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)

        with col2:
            st.subheader("Analysis")
            if st.button("Detect Disease", type="primary"):
                if disease_model is not None and class_indices is not None:
                    with st.spinner("Analyzing image..."):
                        try:
                            # --- Step 1: Image Classification ---
                            processed_image = preprocess_image(image)
                            predictions = disease_model.predict(processed_image)
                            predicted_class_index = np.argmax(predictions, axis=1)[0]
                            confidence = float(np.max(predictions))

                            # Map index → raw label
                            disease_name_raw = class_indices.get(str(predicted_class_index), "Unknown___Unknown Disease")

                            # Parse plant name and disease name from label
                            plant_name, disease_name = parse_label(disease_name_raw)

                            # Human-readable combined label for display
                            display_label = f"{plant_name} — {disease_name}"

                            # --- Step 2: Display Classification Result ---
                            if "healthy" in disease_name.lower():
                                st.success(f"### 🌱 **Detected:** {display_label}")
                            else:
                                st.error(f"### ⚠️ **Detected:** {display_label}")

                            st.write(f"**Confidence Score:** {confidence:.2%}")

                            # --- Step 3: Fetch NVIDIA LLM Cure ---
                            if "healthy" in disease_name.lower():
                                with st.expander("🌿 Plant Health Summary", expanded=True):
                                    st.success(
                                        f"Your **{plant_name}** plant appears **healthy**! "
                                        "No disease treatment is required. Continue with good agricultural "
                                        "practices: proper watering, balanced fertilisation, and regular monitoring."
                                    )
                            else:
                                with st.expander("🩺 Treatment & AI Insights", expanded=True):
                                    with st.spinner(f"Fetching treatment guide for **{disease_name}** from NVIDIA AI..."):
                                        cure_text = get_disease_cure(disease_name, plant_name)
                                    st.markdown(cure_text)

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                else:
                    st.error("Model or class indices not loaded. Please check the `models/` directory.")
