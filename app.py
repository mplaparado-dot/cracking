import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Import the corrected prediction function
from thermal_prediction_wrapper import predict_thermal_crack

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Thermal Crack Risk Prediction System",
    page_icon="🏗️",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Prediction",
        "Dataset & Model Info",
        "Program Relevance"
    ]
)

# =========================
# HOME PAGE
# =========================
if page == "Home":

    st.title("🏗️ AI Thermal Crack Risk Prediction System")

    st.subheader(
        "For Reinforced Concrete Wall-on-Slab Structures"
    )

    st.write("""
    This web-based system uses Artificial Intelligence to predict
    thermal cracking behavior in reinforced concrete wall-on-slab structures.

    The system provides:
    - Crack / No-Crack Classification
    - Maximum Crack Width Prediction
    - Structural Risk Assessment

    The objective is to assist civil engineers, contractors,
    researchers, and students in evaluating thermal cracking risks.
    """)

    st.image(
        "https://images.unsplash.com/photo-1504307651254-35680f356dfd",
        use_container_width=True
    )

# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":

    st.title("🔍 Thermal Crack Prediction (20-Feature Model)")

    st.write("Enter all required structural parameters:")

    col1, col2, col3 = st.columns(3)

    # =========================
    # CATEGORICAL FEATURES
    # =========================
    with col1:

        structural_element = st.text_input(
            "Structural Element (e.g., highway bridge segment 1A)",
            "highway bridge segment 1A"
        )

        type_structure = st.selectbox(
            "Type of Structure",
            ["bridge", "containment", "laboratory", "tank"]
        )

        type_restraint = st.selectbox(
            "Type of Restraint",
            ["base", "base + 1 side", "base + 1 side + top", "base + 2 sides", "base + top"]
        )

        cement_type = st.selectbox(
            "Cement Type",
            ["CEM I 42.5N", "CEM I 52.5N", "CEM II 32.5 R", "CEM II/A 42.5R", 
             "CEM II/B-S 32.5R", "CEM II/B-S 32.5R ", "CEM III/A 32.5N", 
             "CEM III/A 42.5N ", "CEM III/A 42.5R"]
        )

        aggregate_type = st.selectbox(
            "Aggregate Type",
            ["crushed", "gravel", "sand"]
        )

        concrete_class = st.selectbox(
            "Concrete Class",
            ["C20/25", "C25/30", "C30/37", "C35/45", "C50/60"]
        )

    # =========================
    # GEOMETRY
    # =========================
    with col2:

        wall_thickness = st.number_input("Wall Thickness (m)", 0.1, 5.0, 0.3)
        wall_length = st.number_input("Wall Length (m)", 0.1, 100.0, 5.0)
        wall_height = st.number_input("Wall Height (m)", 0.1, 50.0, 3.0)

        foundation_depth = st.selectbox(
            "Foundation Depth (m)",
            ["-", "0.4", "0.6", "0.7", "0.75", "0.9", "1.3"]
        )
        
        foundation_width = st.selectbox(
            "Foundation Width (m)",
            ["-", "1.7", "15", "39.5", "4.5", "8", "9", "96.4"]
        )

        # Convert categorical values to numeric for Af calculation
        try:
            foundation_width_num = float(foundation_width) if foundation_width != "-" else wall_length
        except:
            foundation_width_num = wall_length

        L_H = wall_length / wall_height
        Aw = wall_thickness * wall_height
        Af = foundation_width_num * wall_length
        Aw_Af = Aw / Af if Af != 0 else 0

    # =========================
    # MATERIAL + REBAR
    # =========================
    with col3:

        cement_amount = st.number_input("Cement Amount (kg/m³)", 100, 600, 350)

        fcm_measured = st.selectbox(
            "fcm Measured (MPa)",
            ["32", "32.13", "33 (7x7x7 sample)", "34", "34 (7x7x7 sample)", 
             "35 (7x7x7 sample)", "36 (7x7x7 sample)", "37 (7x7x7 sample)", "37.4",
             "38 (7x7x7 sample)", "39 (7x7x7 sample)", "40 (7x7x7 sample)", "40.2",
             "41 (7x7x7 sample)", "41.1", "41.7", "42 (7x7x7 sample)", "43 (7x7x7 sample)",
             "44 (7x7x7 sample)", "45 (7x7x7 sample)", "46 (7x7x7 sample)", "47 (7x7x7 sample)",
             "48 (7x7x7 sample)", "49 (7x7x7 sample)", "50 (7x7x7 sample)", "51 (7x7x7 sample)",
             "52 (7x7x7 sample)", "53 (7x7x7 sample)", "54 (7x7x7 sample)", "55",
             "55 (7x7x7 sample)", "56 (7x7x7 sample)", "57 (7x7x7 sample)", "58 (7x7x7 sample)",
             "59 (7x7x7 sample)", "60 (7x7x7 sample)", "61 (7x7x7 sample)", "62 (7x7x7 sample)",
             "63 (7x7x7 sample)", "64 (7x7x7 sample)", "64.5", "65", "65 (7x7x7 sample)",
             "66 (7x7x7 sample)", "67 (7x7x7 sample)", "68 (7x7x7 sample)", "69 (7x7x7 sample)",
             "70 (7x7x7 sample)", "71 (7x7x7 sample)"]
        )

        concrete_cover = st.number_input("Concrete Cover (mm)", 10, 100, 40)

        rebar_diameter = st.number_input("Reinforcement Diameter (mm)", 6, 40, 16)

        rebar_spacing = st.selectbox(
            "Reinforcement Spacing (mm)",
            ["-", "10", "14", "16", "21", "22", "43", "47", "62.5", "75", "100", "125", "150", "200"]
        )

        reinforcement_ratio = st.number_input("Reinforcement Ratio (%)", 0.0, 10.0, 1.0)

    # =========================
    # ADDITIONAL PROPERTIES
    # =========================
    with col3:
        st.subheader("Additional Properties")
        
        Iw = st.number_input("Moment of Inertia Iw [m⁴]", 0.0, 10000.0, 100.0)
        If = st.number_input("Moment of Inertia If [m⁴]", 0.0, 10000.0, 100.0)
        modulus_elasticity = st.number_input("Modulus of Elasticity of Steel [GPa]", 0.0, 250.0, 200.0)
        wall_curvature = st.number_input("Wall Curvature [1/m]", 0.0, 1.0, 0.0)
        no_cracks = st.number_input("No. of Cracks (reference)", 0, 1000, 0)
        crack_width_range = st.number_input("Crack Width Range [mm] (reference)", 0.0, 10.0, 0.0)

    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict"):

        # Create input dictionary with all 20 features
        input_data = {
            'type of structure': type_structure,
            'type of restraint': type_restraint,
            'wall thickness [m]': wall_thickness,
            'wall length [m]': wall_length,
            'wall height [m]': wall_height,
            'foundation depth [m]': foundation_depth,
            'foundation width [m]': foundation_width,
            'L/H': L_H,
            'Aw [m2]': Aw,
            'Af [m2]': Af,
            'Aw/Af': Aw_Af,
            'cement type': cement_type,
            'cement amount [kg/m3]': cement_amount,
            'aggregate type': aggregate_type,
            'concrete class': concrete_class,
            'fcm measured [MPa]': fcm_measured,
            'concrete cover [mm]': concrete_cover,
            'reinforcement diameter [mm]': rebar_diameter,
            'reinforcement spacing [mm]': rebar_spacing,
            'reinforcement ratio [%]': reinforcement_ratio
        }

        try:
            # Use the corrected prediction wrapper
            crack_class, crack_width = predict_thermal_crack(input_data)

            st.success("Prediction Completed")

            if crack_class == 1:
                st.error("⚠ Crack Detected")
            else:
                st.success("✅ No Crack Detected")

            st.metric("Max Crack Width (mm)", round(float(crack_width), 3))

            # Risk classification
            if crack_width < 0.2:
                st.info("Risk Level: LOW")
            elif crack_width < 0.4:
                st.warning("Risk Level: MODERATE")
            else:
                st.error("Risk Level: HIGH")

        except Exception as e:
            st.error("Model loading failed.")
            st.code(str(e))
# =========================
# DATASET PAGE
# =========================
elif page == "Dataset & Model Info":

    st.title("📊 Dataset and Model Information")

    st.subheader("Dataset Source")

    st.write("""
    Database for Thermal Cracking of Wall-on-Slab Structures

    Source:
    Zenodo Research Database

    Developed by:
    Agnieszka Jędrzejewska and Mariusz Zych
    """)

    st.subheader("Dataset Features")

    st.write("""
    - Wall Thickness
    - Wall Height
    - Reinforcement Ratio
    - Concrete Strength
    - Thermal Strain
    - Shrinkage Strain
    - Maximum Crack Width
    - Crack / No-Crack Classification
    """)

    st.subheader("Machine Learning Models")

    st.write("""
    Classification Model:
    - Random Forest Classifier

    Regression Model:
    - Random Forest Regressor
    """)

    st.subheader("Evaluation Metrics")

    st.write("""
    Classification:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

    Regression:
    - MAE
    - RMSE
    - R² Score
    """)

# =========================
# PROGRAM RELEVANCE PAGE
# =========================
elif page == "Program Relevance":

    st.title("🎓 Program Relevance")

    st.write("""
    This project supports Civil Engineering through the
    application of Artificial Intelligence in structural analysis.

    Areas covered:
    - Reinforced Concrete Structures
    - Structural Safety
    - Crack Analysis
    - Construction Materials
    - Infrastructure Maintenance

    The system assists engineers in making faster and more
    informed decisions regarding thermal cracking risks.
    """)

    st.info("""
    Target Users:

    • Civil Engineers
    • Contractors
    • Researchers
    • Infrastructure Owners
    • Students
    """)