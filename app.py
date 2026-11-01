import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import os

# Configure the page
st.set_page_config(page_title="EEG Seizure Prediction Portfolio", layout="wide")

# Helper functions to safely load assets and data
@st.cache_data
def load_table(filename):
    file_path = Path(f"data/results/{filename}")
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame({"Notice": [f"Table {filename} not found in data/results/"]})

def show_image(filename, caption):
    img_path = Path(f"assets/{filename}")
    if img_path.exists():
        st.image(Image.open(img_path), caption=caption, use_container_width=True)
    else:
        st.info(f"Image '{filename}' not found in assets/ folder. (Download from your Drive's figures_final folder).")

# Sidebar Navigation
st.sidebar.title("Dissertation Navigation")
page = st.sidebar.radio("Select Chapter/Section:", [
    "1. Architecture & Methodology", 
    "2. Dataset Characteristics", 
    "3. CHB-MIT Generalization (LOPO)", 
    "4. Cross-Dataset Transfer (Siena)", 
    "5. Clinical Event-Level Burden",
    "6. Appendix & Metadata"
])

# Main Header
st.title("Generalising EEG-Based Seizure Prediction")
st.markdown("**A Cross-Patient and Cross-Dataset Study Using CHB-MIT and Siena Scalp EEG**")
st.markdown("---")

# ---------------------------------------------------------
# PAGE 1: Architecture & Methodology
# ---------------------------------------------------------
if page == "1. Architecture & Methodology":
    st.header("Pipeline Architecture & Methodology")
    
    st.subheader("Data Processing Pipeline")
    show_image("figure_03_preprocessing_pipeline.png", "Full Preprocessing and Labelling Pipeline")
    
    st.subheader("EEG Phase Timeline & Windowing")
    col1, col2 = st.columns(2)
    with col1:
        show_image("figure_01_eeg_phase_timeline.png", "EEG Phase Timeline")
    with col2:
        show_image("figure_04_windowing_labelling.png", "Windowing and Label Generation (30s Windows)")
        
    st.subheader("Channel Harmonization")
    show_image("figure_05_channel_harmonisation.png", "Harmonization across CHB-MIT and Siena")
    
    st.subheader("1D-CNN Architecture")
    show_image("figure_08_cnn_architecture.png", "1D-CNN Architecture")
    
    st.subheader("Preprocessing Parameters")
    st.dataframe(load_table("table_03_preprocessing_labelling_parameters.csv"), use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: Dataset Characteristics
# ---------------------------------------------------------
elif page == "2. Dataset Characteristics":
    st.header("Dataset Exploration")
    
    st.subheader("Overall Dataset Comparison")
    st.dataframe(load_table("table_02_dataset_characteristics.csv"), use_container_width=True)
    show_image("figure_02_chb_vs_siena_comparison.png", "CHB-MIT vs Siena Comparison")
    
    st.subheader("CHB-MIT Label & Patient Distribution")
    col1, col2 = st.columns(2)
    with col1:
        show_image("figure_06_chb_label_distribution.png", "Class Imbalance: Preictal vs Interictal")
    with col2:
        show_image("figure_06b_chb_windows_by_patient.png", "Windows Distributed by Patient")
        
    st.subheader("Valid CHB-MIT Patient Cohort Summary")
    st.dataframe(load_table("table_04_valid_chb_patient_cohort_summary.csv"), use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: CHB-MIT Generalization
# ---------------------------------------------------------
elif page == "3. CHB-MIT Generalization (LOPO)":
    st.header("Patient-Independent Performance (CHB-MIT)")
    
    st.subheader("Baseline Models vs 1D-CNN")
    col1, col2 = st.columns(2)
    with col1:
        show_image("figure_07_baseline_model_comparison.png", "Baseline Model Comparison")
    with col2:
        show_image("figure_09_chb_lopo_model_comparison.png", "Final LOPO Model Comparison")
        
    st.subheader("Patient-Specific vs Patient-Independent")
    show_image("figure_15_patient_specific_vs_independent.png", "Upper-bound vs Realistic Generalization")
    
    st.subheader("Final CHB-MIT Comparison Metrics")
    st.dataframe(load_table("table_08_final_chb_comparison.csv"), use_container_width=True)
    
    with st.expander("View Detailed Per-Patient Results (LOPO)"):
        st.dataframe(load_table("table_05_baseline_lopo_per_patient_results.csv"), use_container_width=True)
        st.dataframe(load_table("table_06_baseline_summary_statistics.csv"), use_container_width=True)
        st.dataframe(load_table("table_07_cnn_summary_statistics.csv"), use_container_width=True)
        
    st.subheader("Training Loss Curve")
    show_image("figure_16_chb_source_training_loss.png", "CNN Training Loss")

# ---------------------------------------------------------
# PAGE 4: Cross-Dataset Transfer (Siena)
# ---------------------------------------------------------
elif page == "4. Cross-Dataset Transfer (Siena)":
    st.header("Domain Shift & Transfer Learning (CHB-MIT → Siena)")
    
    st.subheader("The Impact of Domain Shift")
    show_image("figure_12_chb_to_siena_drop.png", "Performance Drop due to Domain Shift")
    st.dataframe(load_table("table_09_chb_to_siena_transfer_results.csv"), use_container_width=True)
    
    st.subheader("Recovery via Fine-Tuning")
    show_image("figure_13_finetuning_improvement.png", "Improvement Post Fine-Tuning")
    st.dataframe(load_table("table_10_fine_tuning_results.csv"), use_container_width=True)
    
    st.subheader("Fine-Tuning Loss Curve")
    show_image("figure_16b_siena_finetune_training_loss.png", "Siena Fine-Tuning Loss")

# ---------------------------------------------------------
# PAGE 5: Event-Level Clinical Burden
# ---------------------------------------------------------
elif page == "5. Clinical Event-Level Burden":
    st.header("Real-World Clinical Viability")
    st.write("Measuring event-level sensitivity against the approximate false positive burden per hour.")
    
    show_image("figure_14_event_level_comparison.png", "Event-Level Metric Comparison")
    st.dataframe(load_table("table_11_event_level_results.csv"), use_container_width=True)

# ---------------------------------------------------------
# PAGE 6: Appendix
# ---------------------------------------------------------
elif page == "6. Appendix & Metadata":
    st.header("Project Metadata & Literature Review")
    
    st.subheader("Research Question Mapping")
    st.dataframe(load_table("table_12_chapter_to_research_question_mapping.csv"), use_container_width=True)
    
    st.subheader("Literature Comparison")
    st.dataframe(load_table("table_01_literature_comparison_template.csv"), use_container_width=True)