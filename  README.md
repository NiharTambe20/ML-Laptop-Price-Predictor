💻 Laptop Price Predictor

Machine Learning-Based Laptop Price Prediction System

An interactive Machine Learning application that predicts the estimated price of a laptop based on its hardware specifications, including brand, processor, RAM, storage, GPU, display resolution, and other features.

🌐 Live Demo: Laptop Price Predictor

⸻

📌 Project Overview

Choosing a laptop can be challenging because prices vary significantly depending on hardware configurations and specifications.

This project uses Machine Learning techniques to estimate laptop prices based on user-provided specifications.

The model is trained on a laptop dataset and integrated into an interactive web application built using Streamlit.

Users can select their preferred laptop specifications and receive an estimated price instantly.

✨ Key Features

* Brand Selection: Choose from multiple laptop manufacturers.
* Processor Selection: Select CPU brands and configurations.
* Memory Configuration: Specify RAM and storage capacities.
* Display Specifications: Select screen resolution and display features.
* GPU Selection: Choose graphics processing configurations.
* Instant Prediction: Get an estimated laptop price based on selected specifications.
* Interactive Interface: User-friendly web application built with Streamlit.
* Cloud Deployment: Accessible online without requiring local installation.

🛠️ Technologies Used

Technology	Purpose
Python	Core programming language
Pandas	Data manipulation and analysis
NumPy	Numerical computations
Scikit-learn	Machine Learning and model development
XGBoost	Model-related dependency
Streamlit	Interactive web application
Jupyter Notebook	Data exploration and experimentation
Git & GitHub	Version control and project hosting
Streamlit Community Cloud	Application deployment

🧠 Machine Learning Workflow

The project follows these major steps:

1. Data Collection: Load the laptop dataset.
2. Data Preprocessing: Handle missing values and prepare the dataset.
3. Feature Engineering: Extract useful features such as PPI, touchscreen availability, IPS display, CPU brand, GPU brand, and storage configurations.
4. Exploratory Data Analysis: Understand relationships between laptop specifications and prices.
5. Model Development: Experiment with different Machine Learning algorithms.
6. Model Evaluation: Evaluate models using appropriate regression metrics.
7. Model Serialization: Save the trained pipeline using Pickle.
8. Application Development: Integrate the trained model into Streamlit.
9. Deployment: Deploy the application using Streamlit Community Cloud.

🤖 Model Information

* Problem Type: Regression
* Target Variable: Laptop Price
* Target Transformation: Natural Logarithm
* Final Saved Model: Stacking Regressor with Random Forest and Ridge Regression components
* Preprocessing: ColumnTransformer and One-Hot Encoding
* Model Serialization: Pickle (pipe.pkl)

The predicted output is converted back from the logarithmic scale to the original price scale before displaying the estimated laptop price.

📂 Project Structure

ML-Laptop-Price-Predictor/
│
├── app.py
├── main.py
│
├── pipe.pkl
├── df.pkl
│
├── Laptop-Predictor-System.ipynb
├── Laptop-Predictor-System-Final.ipynb
│
├── requirements.txt
├── Procfile
├── setup.sh
├── .gitignore
└── README.md

Important Files

* app.py – Streamlit application.
* main.py – Supporting Python implementation.
* pipe.pkl – Serialized trained Machine Learning pipeline.
* df.pkl – Processed dataset used by the application.
* Laptop-Predictor-System-Final.ipynb – Final model development notebook.
* requirements.txt – Python dependencies.

⚙️ Installation and Setup

Follow these steps to run the project locally.

1. Clone the Repository

git clone https://github.com/NiharTambe20/ML-Laptop-Price-Predictor.git

2. Navigate to the Project Directory

cd ML-Laptop-Price-Predictor

3. Install Dependencies

pip install -r requirements.txt

4. Run the Streamlit Application

streamlit run app.py

The application will open in your browser.

🚀 Live Application

Try the deployed application here:

Open Laptop Price Predictor

🔮 Future Improvements

* Improve prediction accuracy through further model experimentation.
* Expand the dataset with more recent laptop configurations.
* Add additional laptop specifications and filtering options.
* Improve the user interface and visualization.
* Integrate real-time laptop market pricing data.

👨‍💻 Author

Nihar Tambe

B.Tech – Information Technology

Vidyalankar Institute of Technology, Mumbai

* GitHub: @NiharTambe20
* Project Repository: ML-Laptop-Price-Predictor

⸻

⭐ If you find this project interesting, consider giving the repository a star!