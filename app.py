# ==========================================
# LAPTOP PRICE PREDICTOR
# Machine Learning Web Application
# Built using Streamlit
# ==========================================


# Import Streamlit to create the web application
import streamlit as st

# Import pickle to load our previously trained ML model
import pickle

# Import NumPy for mathematical calculations
import numpy as np

# Import Pandas to create and manage DataFrames
import pandas as pd

# Import Path to handle file locations properly
from pathlib import Path


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

# Configure the browser tab and page layout
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# Display the main heading of our application
st.title("💻 Laptop Price Predictor")

# Display a short description below the heading
st.write(
    "Enter your laptop specifications below "
    "to get an estimated laptop price."
)


# ==========================================
# 2. LOAD MODEL AND DATASET
# ==========================================

# Get the directory where this app.py file is located
BASE_DIR = Path(__file__).resolve().parent


# Cache the model and dataset so they are not loaded
# repeatedly whenever Streamlit refreshes the application
@st.cache_resource
def load_model():

    # Open the saved machine learning pipeline
    with open(BASE_DIR / "pipe.pkl", "rb") as file:

        # Load the trained pipeline from the pickle file
        pipe = pickle.load(file)

    # Open the saved dataset
    with open(BASE_DIR / "df.pkl", "rb") as file:

        # Load the dataset from the pickle file
        df = pickle.load(file)

    # Return both the trained pipeline and dataset
    return pipe, df


# Try loading the model and dataset
try:

    # Store the loaded model and dataset
    pipe, df = load_model()

# Handle any error that occurs during file loading
except Exception as e:

    # Display the error message on the website
    st.error(f"Error loading model or dataset: {e}")

    # Stop the application if loading fails
    st.stop()


# ==========================================
# 3. FEATURE COLUMNS
# ==========================================

# These are the exact 14 input features used during model training
feature_columns = [
    'Company',
    'TypeName',
    'Ram',
    'Weight',
    'Touchscreen',
    'Ips',
    'ppi',
    'Cpu brand',
    'HDD',
    'SSD',
    'Hybrid',
    'Flash_Storage',
    'Gpu brand',
    'os'
]


# Check whether all required features exist in the dataset
missing_columns = [
    col for col in feature_columns if col not in df.columns
]

# Stop the application if any required feature is missing
if missing_columns:

    st.error(f"Missing features in dataset: {missing_columns}")

    st.stop()


# Check whether the saved model expects the same features
if not hasattr(pipe, "feature_names_in_"):

    st.error("The saved model does not contain feature names.")

    st.stop()


# Get the feature names stored inside the trained pipeline
model_features = pipe.feature_names_in_.tolist()


# Compare the model features with our application features
if model_features != feature_columns:

    st.error(
        "Feature mismatch detected between the application "
        "and the trained ML model."
    )

    st.stop()


# ==========================================
# 4. INPUT OPTIONS
# ==========================================

# Create a function to extract unique values from a dataset column
def get_options(column):

    # Remove missing values, extract unique values and sort them
    return sorted(df[column].dropna().unique().tolist())


# Extract available laptop brands
company_list = get_options('Company')

# Extract available laptop types
type_list = get_options('TypeName')

# Extract available CPU brands
cpu_list = get_options('Cpu brand')

# Extract available GPU brands
gpu_list = get_options('Gpu brand')

# Extract available operating systems
os_list = get_options('os')


# Create a function to extract numerical storage options
def get_storage_options(column):

    # Remove missing values from the selected column
    values = df[column].dropna().unique().tolist()

    # Convert storage values into numerical values
    values = sorted(set(float(x) for x in values))

    # Add zero if it is not already present
    if 0 not in values:

        values.insert(0, 0)

    # Return the final list of storage options
    return values


# Extract available HDD capacities
hdd_list = get_storage_options('HDD')

# Extract available SSD capacities
ssd_list = get_storage_options('SSD')

# Extract available Hybrid storage capacities
hybrid_list = get_storage_options('Hybrid')

# Extract available Flash storage capacities
flash_list = get_storage_options('Flash_Storage')


# Extract available RAM configurations from the dataset
ram_list = sorted(df['Ram'].dropna().unique().tolist())


# ==========================================
# 5. SCREEN RESOLUTION OPTIONS
# ==========================================

# These resolutions are extracted from the original
# ScreenResolution column in our Jupyter Notebook dataset.
# We use their numerical dimensions to calculate PPI.

resolution_list = [
    "1366x768",
    "1440x900",
    "1600x900",
    "1920x1080",
    "1920x1200",
    "2160x1440",
    "2304x1440",
    "2400x1600",
    "2560x1440",
    "2560x1600",
    "2736x1824",
    "2880x1800",
    "3200x1800",
    "3840x2160"
]


# ==========================================
# 6. USER INPUTS
# ==========================================

# Display a heading for the laptop configuration section
st.header("Laptop Specifications")

# Divide the input section into two columns
col1, col2 = st.columns(2)


# Create a form so that inputs are submitted together
with st.form("laptop_prediction_form"):

    # Create two columns for better input alignment
    col1, col2 = st.columns(2)

    # ------------------------------------------
    # LEFT COLUMN
    # ------------------------------------------

    with col1:

        # Let the user select the laptop brand
        company = st.selectbox(
            "Laptop Brand",
            company_list
        )

        # Let the user select the laptop category
        laptop_type = st.selectbox(
            "Laptop Type",
            type_list
        )

        # Let the user select the RAM capacity
        ram = st.selectbox(
            "RAM (GB)",
            ram_list
        )

        # Let the user enter the laptop weight
        weight = st.number_input(
            "Weight (kg)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1
        )

        # Let the user select whether the laptop has a touchscreen
        touchscreen = st.selectbox(
            "Touchscreen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        # Let the user select whether the laptop has an IPS display
        ips = st.selectbox(
            "IPS Display",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        # Let the user enter the screen size in inches
        screen_size = st.number_input(
            "Screen Size (inches)",
            min_value=10.0,
            max_value=20.0,
            value=15.6,
            step=0.1
        )


    # ------------------------------------------
    # RIGHT COLUMN
    # ------------------------------------------

    with col2:

        # Let the user select the screen resolution
        resolution = st.selectbox(
            "Screen Resolution",
            resolution_list
        )

        # Let the user select the CPU brand
        cpu = st.selectbox(
            "CPU Brand",
            cpu_list
        )

        # Let the user select HDD storage capacity
        hdd = st.selectbox(
            "HDD (GB)",
            hdd_list
        )

        # Let the user select SSD storage capacity
        ssd = st.selectbox(
            "SSD (GB)",
            ssd_list
        )

        # Let the user select Hybrid storage capacity
        hybrid = st.selectbox(
            "Hybrid Storage (GB)",
            hybrid_list
        )

        # Let the user select Flash storage capacity
        flash_storage = st.selectbox(
            "Flash Storage (GB)",
            flash_list
        )

        # Let the user select the GPU brand
        gpu = st.selectbox(
            "GPU Brand",
            gpu_list
        )

        # Let the user select the operating system
        operating_system = st.selectbox(
            "Operating System",
            os_list
        )


    # Create a button to submit all laptop specifications
    submit_button = st.form_submit_button(
        "Predict Laptop Price",
        use_container_width=True
    )


# ==========================================
# 7. PREDICTION
# ==========================================

# Execute the prediction only when the user submits the form
if submit_button:

    # Check whether the entered screen size is valid
    if screen_size <= 0:

        # Display an error if the screen size is invalid
        st.error("Please enter a valid screen size.")

    else:

        # ------------------------------------------
        # STEP 1: CALCULATE PPI
        # ------------------------------------------

        # Split the selected resolution into width and height
        x_res, y_res = map(int, resolution.split("x"))

        # Calculate Pixel Per Inch using the same formula
        # that was used during feature engineering in our notebook
        ppi = np.sqrt(x_res**2 + y_res**2) / screen_size


        # ------------------------------------------
        # STEP 2: CREATE INPUT DATA
        # ------------------------------------------

        # Store all selected laptop specifications in a dictionary
        input_data = {

            'Company': company,

            'TypeName': laptop_type,

            'Ram': ram,

            'Weight': weight,

            'Touchscreen': touchscreen,

            'Ips': ips,

            'ppi': ppi,

            'Cpu brand': cpu,

            'HDD': hdd,

            'SSD': ssd,

            'Hybrid': hybrid,

            'Flash_Storage': flash_storage,

            'Gpu brand': gpu,

            'os': operating_system
        }


        # ------------------------------------------
        # STEP 3: CREATE INPUT DATAFRAME
        # ------------------------------------------

        # Convert the dictionary into a Pandas DataFrame
        # using the exact feature order expected by our model
        query = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )


        # ------------------------------------------
        # STEP 4: HANDLE DATA TYPES
        # ------------------------------------------

        # Identify the categorical features used during training
        categorical_columns = [

            'Company',

            'TypeName',

            'Cpu brand',

            'Gpu brand',

            'os'
        ]

        # Convert categorical columns into object data type
        for column in categorical_columns:

            query[column] = query[column].astype(object)


        # Identify the numerical features used during training
        numerical_columns = [

            'Ram',

            'Weight',

            'Touchscreen',

            'Ips',

            'ppi',

            'HDD',

            'SSD',

            'Hybrid',

            'Flash_Storage'
        ]

        # Make sure all numerical columns contain numeric values
        for column in numerical_columns:

            query[column] = pd.to_numeric(query[column])


        # ------------------------------------------
        # STEP 5: PREDICT LAPTOP PRICE
        # ------------------------------------------

        try:

            # Pass the user input into the trained ML pipeline
            # The pipeline automatically performs One-Hot Encoding
            # and applies the trained Stacking Regressor
            prediction = pipe.predict(query)[0]


            # Convert the logarithmic prediction back into
            # the original laptop price using the exponential function
            predicted_price = np.exp(prediction)


            # Check whether the predicted price is a valid number
            if not np.isfinite(predicted_price):

                st.error("The model returned an invalid prediction.")

            else:

                # Display a success message
                st.success("Prediction Successful!")

                # Display the estimated laptop price in Indian Rupees
                st.subheader(
                    f"Estimated Laptop Price: ₹ {predicted_price:,.2f}"
                )

                # Display a small note below the prediction
                st.caption(
                    "This price is an estimate generated by "
                    "our trained Machine Learning model."
                )


        # Handle any error that occurs during prediction
        except Exception as e:

            # Display the prediction error on the website
            st.error(f"Prediction Error: {e}")


# ==========================================
# 8. FOOTER
# ==========================================

# Add a divider before the footer
st.divider()

# Display the project footer
st.caption(
    "Laptop Price Predictor | Built with Python, "
    "Scikit-learn and Streamlit"
)