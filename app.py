import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

# Configure the page
st.set_page_config(
    page_title="EEG Seizure Prediction | Project Story",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_data
def load_table(filename):
    file_path = Path(f"data/results/{filename}")
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame({"Notice": [f"Table '{filename}' not found in data/results/"]})

def show_image(filename, caption):
    img_path = Path(f"assets/{filename}")
    if img_path.exists():
        st.image(Image.open(img_path), caption=caption, use_container_width=True)
    else:
        st.info(f"Image '{filename}' not found in assets/ folder.")

# ---------------------------------------------------------
# Sidebar Navigation & Plain-English Glossary
# ---------------------------------------------------------
st.sidebar.title("🧭 Project Storyline")
page = st.sidebar.radio("Follow the investigation:", [
    "0. Start Here: The Story",
    "1. How the Data & Model Work",
    "2. The Two Hospital Datasets",
    "3. Act I: Unseen Patients (CHB-MIT)",
    "4. Act II: Unseen Hospital (Siena)",
    "5. Act III: Real-World Clinical Burden",
    "6. Full Appendix & Research Map"
])

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Quick Glossary (Plain English)", expanded=True):
    st.markdown("""
    * **Interictal:** Normal, baseline brain activity far from any seizure.
    * **Preictal:** The **5 to 30 minute window** right before a seizure starts—our target warning zone.
    * **LOPO (Leave-One-Patient-Out):** Training on 9 patients and testing on the 10th unseen patient to mimic real hospital deployment.
    * **Domain Shift:** The accuracy drop that happens when a model trained on one hospital's equipment is tested on another hospital.
    """)

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.title("🧠 Generalising EEG-Based Seizure Prediction")
st.markdown("**Can a deep learning model warn patients up to 25 minutes before an epileptic seizure—even across unseen patients and different hospitals?**")
st.markdown("---")

# =========================================================
# PAGE 0: START HERE (THE STORY & HERO METRICS)
# =========================================================
if page == "0. Start Here: The Story":
    st.header("The Problem & Why Most Models Fail in the Real World")
    
    st.write("""
    For people living with drug-resistant epilepsy, the unpredictability of seizures is one of the greatest dangers. 
    An early warning system—even **25 minutes in advance**—gives patients time to get to safety or take fast-acting medication.
    
    **The Catch:** Many academic studies report >95% accuracy because they train and test on the *same* patient. 
    In reality, a hospital needs a model that works out-of-the-box on **new, unseen patients** and across **different EEG machines**. 
    This project stress-tests a 1D Convolutional Neural Network (1D-CNN) against both challenges.
    """)

    # Hero KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total EEG Windows", "31,147", "30-sec brainwave clips")
    col2.metric("Patients Studied", "18 Patients", "2 hospitals (US & Italy)")
    col3.metric("Early Warning Lead", "Up to 25 Min", "5–30 min pre-seizure")
    col4.metric("Fine-Tuned Sensitivity", "75.0%", "+21.7% over zero-shot")

    st.markdown("---")
    st.subheader("🗺️ How to Read This Dashboard (3-Act Narrative)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
        **Act I: Unseen Patients (Boston)**  
        We test whether a 1D-CNN can detect pre-seizure brain patterns in a brand-new patient it has never seen before, beating traditional ML baselines.
        """)
    with c2:
        st.warning("""
        **Act II: Unseen Hospital (Italy)**  
        We take the Boston-trained model and test it cold (*zero-shot*) on an Italian hospital dataset. Performance drops due to equipment/population shift—until we fine-tune it.
        """)
    with c3:
        st.success("""
        **Act III: Clinical Reality Check**  
        High window accuracy means nothing if an alarm goes off every 5 minutes. We measure **seizure events caught vs. false alarms per hour**.
        """)

    st.markdown("👉 **Next step:** Select **1. How the Data & Model Work** in the left sidebar to see how raw brainwaves are turned into predictions.")

# =========================================================
# PAGE 1: ARCHITECTURE & METHODOLOGY
# =========================================================
# =========================================================
    # STEP A: DEFINING THE WARNING ZONE (FIGURE 1 & FIGURE 4)
    # =========================================================
    st.subheader("Step A: Defining the Early Warning Zone & Slicing the Signal")
    st.write("Before training any model, we must teach it which parts of a patient's brainwave recording count as 'normal', which parts count as an 'impending seizure warning', and which parts must be thrown away.")

    # --- FIGURE 1 ROW ---
    st.markdown("#### 1. How a Patient's Timeline is Divided (`Figure 1`)")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_01_eeg_phase_timeline.png", "Figure 1: The five clinical phases around an epileptic seizure")
    with col2:
        st.info("""
        **How to read this diagram (Left → Right):**
        * **What do the numbers `0` to `7` mean?** They are an **illustrative timeline** (not drawn to literal minute scale) showing the order of brain states from left to right. There is no vertical axis because this is a single timeline bar.
        * 🟦 **Blue (`0` to `2`) — Inter-ictal (`Label = 0`):** Normal, everyday brain activity recorded **at least 2 hours away** from any seizure. Kept as our "safe" baseline.
        * 🟧 **Orange (`2` to `4.5`) — Pre-ictal (`Label = 1`):** The **30 to 5 minute window before a seizure**. This 25-minute stretch is our target early-warning zone.
        * 🟩 **Green (`4.5` to `5`) — Excluded Buffer:** The final **5 minutes** right before the seizure starts. Thrown away so the model learns *early* warning signs rather than cheating by detecting the seizure starting.
        * 🟥 **Red (`5` to `5.8`) — Ictal:** The actual seizure happening in real time. Thrown away (detecting an active seizure is too late to warn the patient).
        * 🟪 **Purple (`5.8` to `7`) — Post-ictal:** Recovery brainwaves right after a seizure ends. Thrown away so tired/recovering brainwaves aren't mistaken for normal baseline.
        """)

    st.markdown("---")

    # --- FIGURE 4 ROW ---
    st.markdown("#### 2. Slicing Continuous Brainwaves into Overlapping Clips (`Figure 4`)")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_04_windowing_labelling.png", "Figure 4: Sliding overlapping windows across the timeline")
    with col2:
        st.info("""
        **How to read every part of this diagram:**
        * **Horizontal Axis (`0` to `200`):** Time moving forward in minutes (illustrative scale).
        * **Vertical Dashed Line (`x = 180`):** **Seizure Onset**—the exact moment the seizure begins.
        * **Light-Blue Shaded Band (`x = 120` to `150`):** **Pre-ictal Warning Zone**—a 30-minute illustrative warning window ending well before onset at `180` (in our real pipeline, this zone is **30 to 5 minutes** before onset).
        * **White Gap (`x = 150` to `180`):** The **excluded buffer zone** right before the seizure—any clip falling here is discarded.
        * **Why does the horizontal bar change colors?** Each colored block represents one **sliding time window** (`30 seconds` in our real model). Because each new window slides forward by only **half a window (`15 seconds` / 50% overlap)**, each color transition shows where the next overlapping clip begins.
        * **How clips get labeled:** Clips inside the light-blue shaded zone (olive & cyan segments) are labeled **`1` (Warning)**. Clips far to the left (at least 2 hours away in real data) are labeled **`0` (Normal)**. Clips touching the buffer or dashed seizure line (far right) are dropped.
        """)

    st.markdown("---")

    # =========================================================
    # STEP B: HARMONIZING TWO DIFFERENT HOSPITALS (FIGURE 3 & 5)
    # =========================================================
    st.subheader("Step B: Cleaning & Standardizing Two Different Hospitals")
    st.write("Hospital EEG files are huge, noisy, and recorded differently depending on the country and equipment. Before any AI model can learn from them, every recording passes through an automated assembly line.")

    # --- FIGURE 3 ROW ---
    st.markdown("#### 1. The 7-Step Signal Assembly Line (`Figure 3`)")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_03_preprocessing_pipeline.png", "Figure 3: Step-by-step pipeline from raw hospital files to model-ready data")
    with col2:
        st.info("""
        **How to read this diagram (Follow the arrows Left → Right):**
        * *(Note: There are no numbers or axes here—each blue box is one processing stage in chronological order.)*
        * **Box 1 — Raw EDF loading:** Opens the raw hospital brainwave recordings (stored in **EDF**, the standard medical file format for EEG).
        * **Box 2 — Channel cleaning / harmonisation:** Strips out non-brain sensors (like heart-rate or eye-movement channels) and standardizes electrode names so both hospitals match.
        * **Box 3 — Resampling to 256 Hz:** Adjusts recording speeds so every file captures exactly **256 brainwave snapshots per second** (Siena was recorded twice as fast at 512 Hz, so we downsample it to match Boston's 256 Hz).
        * **Box 4 — Band-pass filtering (0.5–40 Hz):** Acts like an audio equalizer—blocks slow baseline drift (`< 0.5 Hz`, like breathing/sweat) and high-frequency electrical hum (`> 40 Hz`, like wall outlets), keeping only true brainwave frequencies.
        * **Box 5 — 30 s windows with 50% overlap:** Slices the continuous recording into **30-second clips**, stepping forward every **15 seconds** (as shown in Figure 4).
        * **Box 6 — Label generation:** Tags each 30-second clip as **`1` (Pre-ictal warning)** or **`0` (Inter-ictal normal)** based on its distance from a seizure.
        * **Box 7 — Chunk caching / feature extraction:** Saves the cleaned clips onto disk as compressed `.npz` chunks (for the 1D-CNN) and extracts 144 summary stats (for the baseline models) so the computer never runs out of memory.
        """)

    st.markdown("---")

    # --- FIGURE 5 ROW ---
    st.markdown("#### 2. Making Boston & Italy Speak the Same Sensor Language (`Figure 5`)")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_05_channel_harmonisation.png", "Figure 5: Unifying two different electrode wiring methods into one 16-channel standard")
    with col2:
        st.info("""
        **How to read the three boxes & inward arrows:**
        * **Why do both outer arrows point toward the center?** Because two completely different hospital setups (left and right) are being converted into **one shared middle format** so a single AI model can read both.
        * **Left Box — CHB-MIT (Bipolar channels):** The Boston hospital recorded brainwaves as **pairs** (e.g., `FP1-F7`), measuring the electrical voltage *difference* between two neighboring scalp sensors.
        * **Right Box — Siena (Referential → bipolar):** The Italian hospital recorded each sensor individually against a single reference point (e.g., `Fp1`, `F7`) and used older naming labels (like `T3` instead of `T7`). Our code renames the sensors and subtracts pairs (`FP1 minus F7`) to reconstruct the exact same paired signals.
        * **Center Box — Common 16-channel harmonised montage:** A **"montage"** is simply the map of sensor connections on a patient's head. Both datasets end up with the **exact same 16 paired brain channels** in the exact same order.
        """)

    st.markdown("---")

   # =========================================================
    # STEP C: THE 1D-CNN ARCHITECTURE (FIGURE 8)
    # =========================================================
    st.subheader("Step C: How the AI Model Reads Raw Brainwaves (`Figure 8`)")
    st.write("Traditional machine learning requires humans to manually calculate summary statistics (like average wave power) before training. Instead, our **1D Convolutional Neural Network (1D-CNN)** reads the raw brainwave curves directly from left to right[cite: 6].")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_08_cnn_architecture.png", "Figure 8: High-level flow of a 30-second brainwave clip through the 1D-CNN")
    with col2:
        st.success("""
        **How to read this diagram (Follow the arrows Left → Right):**
        * *(Note: There are no axes here—each blue box shows how one 30-second brainwave clip is transformed step by step into a final warning score.)*
        * **Box 1 — `Input (16 × 7680)`:** What do these two numbers mean?
          * **`16`** = The **16 harmonized brain channels** (sensor pairs across the scalp)[cite: 6].
          * **`7680`** = The exact number of brainwave measurements in one 30-second clip (`30 seconds × 256 snapshots per second = 7,680 time steps`)[cite: 6].
        * **Box 2 — `Conv Block 1` (Early Pattern Detector):** Slides small digital filters across the raw waves to spot basic, fast shapes—like sharp spikes, sudden voltage jumps, or rhythmic ripples.
        * **Box 3 — `Conv Block 2` (Deep Pattern Detector):** Combines those basic shapes into higher-level patterns across multiple brain channels that signal early pre-seizure brain changes[cite: 5, 6].
        * **Box 4 — `Pooling / Flatten` (Summarizer):** Compresses thousands of time-step readings down into a compact list of the **128 strongest pattern scores** found anywhere in that 30-second window[cite: 5, 6].
        * **Box 5 — `Dense + Dropout` (Decision Maker & Overfitting Guard):** Weighs all 128 pattern scores together to make a decision[cite: 5, 6]. **`Dropout` (30%)** randomly switches off 30% of connections during training so the model cannot simply memorize one patient's quirks[cite: 5, 6].
        * **Box 6 — `Sigmoid output` (Final Risk Score):** Squashes the final calculation into a clean probability between **`0.0` (0% risk — Normal)** and **`1.0` (100% risk — Impending Seizure)**[cite: 5]. If this score crosses our alert threshold (`0.5` or `0.1`), an early warning is triggered[cite: 6].
        """)

    with st.expander("📋 View Exact Preprocessing & Labelling Parameters (Table 3)"):
        st.dataframe(load_table("table_03_preprocessing_labelling_parameters.csv"), use_container_width=True)

    st.markdown("👉 **Next step:** Go to **2. The Two Hospital Datasets** in the sidebar to see how much data we collected and why severe class imbalance makes accuracy misleading.")

# =========================================================
# PAGE 2: DATASET CHARACTERISTICS
# =========================================================
elif page == "2. The Two Hospital Datasets":
    st.header("2. Exploring the Data: Boston vs. Italy")
    st.write("Seizures are rare events compared to normal daily life. Understanding the data balance across patients is critical before judging any model's score.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_02_chb_vs_siena_comparison.png", "Comparison of patients and window counts across both datasets")
    with col2:
        st.info("""
        **Key Contrast Between Datasets:**
        * **CHB-MIT (28,025 windows):** Heavily dominated by normal *interictal* windows (26,283 vs. 1,742 preictal), reflecting real-world rarity.
        * **Siena (3,122 windows):** Smaller cohort (8 patients) with a higher proportion of preictal clips (2,359 preictal vs. 763 interictal).
        """)

    st.dataframe(load_table("table_02_dataset_characteristics.csv"), use_container_width=True)
    st.markdown("---")

    st.subheader("Patient-by-Patient Breakdown (CHB-MIT)")
    col1, col2 = st.columns(2)
    with col1:
        show_image("figure_06_chb_label_distribution.png", "Severe class imbalance: ~94% normal vs ~6% preictal")
    with col2:
        show_image("figure_06b_chb_windows_by_patient.png", "Recording volume varies widely across individual patients")

    st.warning("**Takeaway:** Because only ~6% of CHB-MIT windows are preictal, a lazy model that predicts 'No Seizure' every time would still get 94% accuracy—which is why we evaluate using **Recall (Sensitivity)** and **ROC-AUC**, and tune probability thresholds (e.g., `0.1` instead of `0.5`).")

    with st.expander("📋 View Valid CHB-MIT Patient Cohort Summary (Table 4)"):
        st.dataframe(load_table("table_04_valid_chb_patient_cohort_summary.csv"), use_container_width=True)

    st.markdown("👉 **Next step:** Go to **3. Act I: Unseen Patients (CHB-MIT)** to see how well the models generalize to brand-new patients.")

# =========================================================
# PAGE 3: CHB-MIT GENERALIZATION
# =========================================================
elif page == "3. Act I: Unseen Patients (CHB-MIT)":
    st.header("3. Act I: Can the Model Generalize to Unseen Patients?")
    st.write("We used strict **Leave-One-Patient-Out (LOPO)** cross-validation: the model is trained on 9 patients and tested on the 10th patient it has never seen.")

    col1, col2 = st.columns(2)
    with col1:
        show_image("figure_07_baseline_model_comparison.png", "Traditional ML baselines (Logistic Regression & Random Forest)")
    with col2:
        show_image("figure_09_chb_lopo_model_comparison.png", "1D-CNN vs. statistical baselines across LOPO folds")

    st.success("""
    **Act I Takeaway:**  
    Traditional models (Logistic Regression and Random Forest) struggle to find patterns that hold across different human brains. 
    The **1D-CNN** learns richer temporal features from the raw EEG signal, outperforming baselines across unseen patients—especially when lowering the alert threshold to `0.1` to catch subtle pre-seizure shifts.
    """)

    st.dataframe(load_table("table_08_final_chb_comparison.csv"), use_container_width=True)
    st.markdown("---")

    st.subheader("Why Patient-Specific Papers Look 'Too Good to Be True'")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_15_patient_specific_vs_independent.png", "Same-patient upper bound vs. realistic unseen-patient generalization")
    with col2:
        st.info("""
        **Sanity Check Finding:**  
        When a model is trained and tested on the **same patient** (left bars), scores shoot up because the model memorizes that specific person's brainwave signature. 
        Testing on **unseen patients** (right bars) is much harder—and represents the true clinical benchmark.
        """)

    col_loss, col_tables = st.columns([1, 1])
    with col_loss:
        st.subheader("Training Stability")
        show_image("figure_16_chb_source_training_loss.png", "Smooth convergence of the source 1D-CNN on CHB-MIT")
    with col_tables:
        st.subheader("Detailed Per-Patient Breakdowns")
        with st.expander("📋 Table 5: Baseline LOPO Per-Patient Results"):
            st.dataframe(load_table("table_05_baseline_lopo_per_patient_results.csv"), use_container_width=True)
        with st.expander("📋 Table 6: Baseline Summary Statistics"):
            st.dataframe(load_table("table_06_baseline_summary_statistics.csv"), use_container_width=True)
        with st.expander("📋 Table 7: CNN Summary Statistics"):
            st.dataframe(load_table("table_07_cnn_summary_statistics.csv"), use_container_width=True)

    st.markdown("👉 **Next step:** Go to **4. Act II: Unseen Hospital (Siena)** to see what happens when we deploy this Boston model to a hospital in Italy.")

# =========================================================
# PAGE 4: CROSS-DATASET TRANSFER (SIENA)
# =========================================================
elif page == "4. Act II: Unseen Hospital (Siena)":
    st.header("4. Act II: What Happens When We Switch Hospitals?")
    st.write("Even after standardizing channels and sampling rates, different hospitals have different patient demographics, recording hardware, and clinical protocols—a challenge known as **Domain Shift**.")

    # Zero-Shot Drop
    st.subheader("Phase 1: Zero-Shot Transfer (No Italian Training Data)")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_12_chb_to_siena_drop.png", "AUC drop when moving from CHB-MIT (Boston) to Siena (Italy)")
    with col2:
        st.warning("""
        **The Domain Shift Reality:**  
        When the Boston-trained CNN is tested cold (*zero-shot*) on Siena patients, recall sits at **53.3%**. 
        Every model experiences an AUC drop across datasets—proving that lab accuracy on one dataset does not automatically transfer to another clinic.
        """)

    st.dataframe(load_table("table_09_chb_to_siena_transfer_results.csv"), use_container_width=True)
    st.markdown("---")

    # Fine-Tuning Recovery
    st.subheader("Phase 2: Recovery via Target-Domain Fine-Tuning")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_13_finetuning_improvement.png", "Performance jump after fine-tuning on a small Siena subset")
    with col2:
        st.success("""
        **Act II Takeaway (53.3% → 75.0% Recall):**  
        By fine-tuning the pre-trained CNN on a small subset of Siena patient data and evaluating on held-out Siena patients, **sensitivity jumped from 53.3% to 75.0%**. 
        Transfer learning allows a hospital to adapt a pre-trained model without needing massive local datasets.
        """)

    st.dataframe(load_table("table_10_fine_tuning_results.csv"), use_container_width=True)

    with st.expander("📈 View Siena Fine-Tuning Loss Curve (Figure 16b)"):
        show_image("figure_16b_siena_finetune_training_loss.png", "Training loss progression during Siena adaptation")

    st.markdown("👉 **Next step:** Go to **5. Act III: Real-World Clinical Burden** to check whether these predictions are practical for a real patient to live with.")

# =========================================================
# PAGE 5: EVENT-LEVEL CLINICAL BURDEN
# =========================================================
elif page == "5. Act III: Real-World Clinical Burden":
    st.header("5. Act III: Clinical Reality — Catching Seizures vs. Alarm Fatigue")
    st.write("""
    In a hospital or wearable device, doctors don't care about isolated 30-second window scores—they care about two practical questions:
    1. **Event Sensitivity:** Did the system warn the patient *at least once* during the 25-minute lead-up to an actual seizure?
    2. **False Positive Burden:** How many false alarms per hour does the patient have to endure?
    """)

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_14_event_level_comparison.png", "Event-level sensitivity vs. false positive burden on Siena")
    with col2:
        st.info("""
        **Act III Takeaway:**  
        Evaluating at the **seizure-event level** reveals the real-world trade-off between catching impending seizures early and minimizing false alarm fatigue—highlighting why post-processing smoothing and calibration are essential next steps before clinical deployment.
        """)

    st.dataframe(load_table("table_11_event_level_results.csv"), use_container_width=True)
    st.markdown("👉 **Next step:** Check **6. Full Appendix & Research Map** to see how every chapter maps to the formal dissertation research questions.")

# =========================================================
# PAGE 6: APPENDIX & METADATA
# =========================================================
elif page == "6. Full Appendix & Research Map":
    st.header("6. Dissertation Structure & Literature Context")
    st.write("This table maps each section of the dashboard back to the formal Master's dissertation chapters and research questions (RQ1–RQ3).")

    st.subheader("Chapter-to-Research-Question Mapping")
    st.dataframe(load_table("table_12_chapter_to_research_question_mapping.csv"), use_container_width=True)

    st.subheader("Literature Comparison Framework")
    st.dataframe(load_table("table_01_literature_comparison_template.csv"), use_container_width=True)