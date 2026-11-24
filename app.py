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
page = st.sidebar.radio("Select Chapter/Section:", [
    "1. Architecture & Methodology",
    "2. Dataset Characteristics",
    "3. Act I: Unseen Patients (CHB-MIT)",
    "4. Act II: Unseen Hospital (Siena)",
    "5. Act III: Clinical Viability & Burden",
    "6. Appendix & Metadata"
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
elif page == "1. How the Data & Model Work":
    st.header("1. Turning Raw Brainwaves into Early Warnings")
    st.write("Before training any model, raw hospital EEG recordings must be cleaned, sliced into time windows, and standardized so two completely different hospitals speak the same data language.")

    # Quick at-a-glance math strip
    m1, m2, m3 = st.columns(3)
    m1.metric("Clip Length", "30 Seconds", "7,680 samples @ 256 Hz")
    m2.metric("Sliding Step", "15 Seconds", "50% window overlap")
    m3.metric("Sensor Standard", "16 Channels", "Unified 10-20 bipolar pairs")

    st.markdown("---")

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

    with st.expander("🔍 View the Exact 16 Harmonized 10-20 Bipolar Channels"):
        st.code("['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',\n 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'P8-O2', 'FZ-CZ']", language="python")

    st.markdown("---")

    # =========================================================
    # STEP C: THE 1D-CNN ARCHITECTURE (FIGURE 8)
    # =========================================================
    st.subheader("Step C: How the AI Model Reads Raw Brainwaves (`Figure 8`)")
    st.write("Traditional machine learning requires humans to manually calculate summary statistics (like average wave power) before training. Instead, our **1D Convolutional Neural Network (1D-CNN)** reads the raw brainwave curves directly from left to right.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_08_cnn_architecture.png", "Figure 8: High-level flow of a 30-second brainwave clip through the 1D-CNN")
    with col2:
        st.success("""
        **How to read this diagram (Follow the arrows Left → Right):**
        * *(Note: There are no axes here—each blue box shows how one 30-second brainwave clip is transformed step by step into a final warning score.)*
        * **Box 1 — `Input (16 × 7680)`:** What do these two numbers mean?
          * **`16`** = The **16 harmonized brain channels** (sensor pairs across the scalp).
          * **`7680`** = The exact number of brainwave measurements in one 30-second clip (`30 seconds × 256 snapshots per second = 7,680 time steps`).
        * **Box 2 — `Conv Block 1` (Early Pattern Detector):** Slides small digital filters across the raw waves to spot basic, fast shapes—like sharp spikes, sudden voltage jumps, or rhythmic ripples.
        * **Box 3 — `Conv Block 2` (Deep Pattern Detector):** Combines those basic shapes into higher-level patterns across multiple brain channels that signal early pre-seizure brain changes.
        * **Box 4 — `Pooling / Flatten` (Summarizer):** Compresses thousands of time-step readings down into a compact list of the **128 strongest pattern scores** found anywhere in that 30-second window.
        * **Box 5 — `Dense + Dropout` (Decision Maker & Overfitting Guard):** Weighs all 128 pattern scores together to make a decision. **`Dropout` (30%)** randomly switches off 30% of connections during training so the model cannot simply memorize one patient's quirks.
        * **Box 6 — `Sigmoid output` (Final Risk Score):** At prediction time, converts the network's raw score into a clean probability between **`0.0` (0% risk — Normal)** and **`1.0` (100% risk — Impending Seizure)**. If this score crosses our alert threshold (`0.5` or `0.1`), an early warning is triggered.
        """)

    with st.expander("📋 View Exact Preprocessing & Labelling Parameters (Table 3)"):
        st.dataframe(load_table("table_03_preprocessing_labelling_parameters.csv"), use_container_width=True)

    st.markdown("👉 **Next step:** Go to **2. The Two Hospital Datasets** in the sidebar to see how much data we collected and why severe class imbalance makes accuracy misleading.")

# =========================================================
# PAGE 2: DATASET CHARACTERISTICS
# =========================================================
elif page == "2. The Two Hospital Datasets":
    st.header("2. Exploring the Data: Boston vs. Italy")
    st.write("Before judging any AI model's score, we need to look at how much brainwave data was collected from each hospital and how rare pre-seizure warning clips actually are compared to normal brain activity.")

    # Quick at-a-glance dataset summary strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CHB-MIT (Boston)", "28,025 Clips", "10 patients (5 LOPO-valid)")
    m2.metric("Siena (Italy)", "3,122 Clips", "8 patients (cross-hospital)")
    m3.metric("CHB-MIT Imbalance", "15 : 1 Ratio", "93.8% Normal vs 6.2% Warning")
    m4.metric("Siena Balance", "1 : 3.1 Ratio", "24.4% Normal vs 75.6% Warning")

    st.markdown("---")

    # =========================================================
    # SECTION A: BOSTON VS ITALY COMPARISON (FIGURE 2 & TABLE 2)
    # =========================================================
    st.subheader("Step A: Comparing Both Hospital Datasets Side-by-Side (`Figure 2`)")
    st.write("Both datasets were standardized into 30-second clips across 16 identical channels at 256 Hz, but their size and class balance differ dramatically.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_02_chb_vs_siena_comparison.png", "Figure 2: Side-by-side comparison of patient counts and 30-second window counts across CHB-MIT and Siena")
    with col2:
        st.info("""
        **How to read every bar and axis on this chart:**
        * **Horizontal Axis (`Dataset`):** Two hospital groups—**CHB-MIT** (Boston, left) and **Siena** (Italy, right).
        * **Vertical Axis (`Count`):** Total count from `0` up to `28,025`.
        * **What the 4 colored bars mean in each group (Left → Right):**
          * 🟦 **Blue (`Patients used`):** **`10`** in CHB-MIT vs. **`8`** in Siena. *(Why do these bars look invisible? Because `10` and `8` sit right on the bottom line when plotted on a `0–28,000` scale!)*
          * 🟧 **Orange (`Windows` — Total clips):** **`28,025`** in CHB-MIT vs. **`3,122`** in Siena. CHB-MIT has nearly **9× more** total data.
          * 🟩 **Green (`Preictal windows` — Warning clips):** **`1,742`** in CHB-MIT vs. **`2,359`** in Siena.
          * 🟥 **Red (`Interictal windows` — Normal clips):** **`26,283`** in CHB-MIT vs. **`763`** in Siena.
        * **Key Visual Contrast:** On **CHB-MIT**, the red normal bar towers over the green warning bar. On **Siena**, the pattern reverses—the green warning bar (`2,359`) is taller than the red normal bar (`763`).
        """)

    with st.expander("📋 View Exact Dataset Characteristics Table (Table 2)", expanded=True):
        st.dataframe(load_table("table_02_dataset_characteristics.csv"), use_container_width=True)

    st.markdown("---")

    # =========================================================
    # SECTION B: SEVERE CLASS IMBALANCE IN CHB-MIT (FIGURE 6)
    # =========================================================
    st.subheader("Step B: Why 'Accuracy' Can Be Misleading — Severe Class Imbalance (`Figure 6`)")
    st.write("In real life, patients spend almost all their time in a normal baseline state rather than about to have a seizure. Even after capping normal clips at a maximum of 80 per recording file, CHB-MIT remains heavily skewed.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_06_chb_label_distribution.png", "Figure 6: Total Interictal (normal) vs. Preictal (warning) 30-second clips in CHB-MIT")
    with col2:
        st.warning("""
        **How to read every part of this chart:**
        * **Horizontal Axis:** The two brain states in CHB-MIT—**Interictal** (left) and **Preictal** (right).
        * **Vertical Axis (`Number of windows`):** Runs from `0` to `25,000+` in steps of `5,000`.
        * **Left Bar (`Interictal` — Normal):** Reaches just above `25,000`—specifically **`26,283` windows (`93.8%`)**.
        * **Right Bar (`Preictal` — Warning):** Sits near the bottom well below `5,000`—specifically **`1,742` windows (`6.2%`)**.
        * **Why this matters for evaluating AI:** With a **15-to-1 imbalance**, a useless model that simply predicts *"No Seizure"* 100% of the time would still score **93.8% accuracy** while missing every seizure! That is why we judge models using **Recall (Sensitivity)** and **ROC-AUC**, and lower our alert threshold to **`0.1`**.
        """)

    st.markdown("---")

    # =========================================================
    # SECTION C: PATIENT-BY-PATIENT VOLUME (FIGURE 6B & TABLE 4)
    # =========================================================
    st.subheader("Step C: Recording Volume Across Individual Patients (`Figure 6b`)")
    st.write("Hospital recordings vary in length from patient to patient, and not every patient's recordings contain usable pre-seizure warning windows.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_06b_chb_windows_by_patient.png", "Figure 6b: Total preprocessed 30-second windows per patient across all 10 CHB-MIT subjects")
    with col2:
        st.info("""
        **How to read every part of this chart:**
        * **Horizontal Axis:** All **10 preprocessed CHB-MIT patients**, sorted from the largest recording volume on the left (`chb04`) to the smallest on the right (`chb12`).
        * **Vertical Axis (`Number of windows`):** Runs from `0` to `3,500` in steps of `500`.
        * **Highest vs. Lowest Bars:**
          * **Tallest (`chb04`):** **`3,488`** windows (just under the `3,500` gridline).
          * **Shortest (`chb12`):** **`1,680`** windows (just above `1,500`, because 3 corrupted files failed the 16-channel check during cleaning).
        * **Why does Figure 6b show 10 patients while Table 4 below lists 5?**
          * Patients **`chb01` through `chb05`** (5 patients) all have both normal and pre-seizure warning clips (`204` to `523` preictal windows each), making them valid for **Leave-One-Patient-Out (LOPO)** testing.
          * Patients **`chb11` through `chb15`** had **`0` valid preictal windows** after applying the strict 30-to-5 minute window rules, so they are excluded from LOPO test folds.
        """)

    with st.expander("📋 View Valid LOPO Patient Cohort Breakdown (Table 4)", expanded=True):
        st.dataframe(load_table("table_04_valid_chb_patient_cohort_summary.csv"), use_container_width=True)

    st.markdown("👉 **Next step:** Go to **3. Act I: Unseen Patients (CHB-MIT)** in the sidebar to see how well the 1D-CNN and baseline models perform when tested on a brand-new patient.")

# =========================================================
# PAGE 3: CHB-MIT GENERALIZATION (ACT I)
# =========================================================
elif page == "3. Act I: Unseen Patients (CHB-MIT)":
    st.header("3. Act I: Can the Model Generalize to Unseen Patients?")
    st.write("To simulate real hospital deployment, we used strict **Leave-One-Patient-Out (LOPO)** cross-validation: the model trains on 4 valid Boston patients and is tested on the **5th patient it has never seen before**, repeated across all 5 patients.")

    # Quick at-a-glance Act I KPI strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Unseen-Patient AUC", "0.568", "+0.130 vs best baseline")
    m2.metric("Top CNN Recall (@ 0.1)", "54.9%", "Catches most warning clips")
    m3.metric("Top CNN Accuracy (@ 0.5)", "71.0%", "Lowest false-alarm rate")
    m4.metric("Same-Patient Ceiling", "0.990 AUC", "Upper-bound sanity check")

    st.markdown("---")

    # =========================================================
    # STEP A: STATISTICAL BASELINES (FIGURE 7)
    # =========================================================
    st.subheader("Step A: How Do Traditional ML Models Perform on Unseen Patients? (`Figure 7`)")
    st.write("Before testing deep learning, we tested two classic machine learning models—**Logistic Regression** and **Random Forest**—using 144 manually calculated brainwave statistics.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_07_baseline_model_comparison.png", "Figure 7: Leave-One-Patient-Out (LOPO) mean scores for traditional statistical baselines")
    with col2:
        st.info("""
        **Questions & Answers to Read This Chart:**
        * **What are the axes?** The horizontal axis (`model`) compares our two baseline algorithms; the vertical axis (`Score`) shows their average score (`0.0` to `0.65`) across all 5 unseen patients.
        * **Why is Logistic Regression's blue `accuracy_mean` bar (`0.614`) tall while its green `recall_mean` bar (`0.239`) is low?** Because 93.8% of the data is normal, Logistic Regression plays it safe by mostly guessing *"Normal"*—missing **76%** of pre-seizure warning clips.
        * **What does `(0.1)` mean next to Random Forest?** By default, models only trigger an alert when risk exceeds `50%` (`0.5`). Lowering that trigger to `10%` (`0.1`) makes Random Forest much more sensitive—pushing its green Recall bar up to **`0.504`**, though extra false alarms pull its blue Accuracy bar down to **`0.433`**.
        * **Why are both red `auc_mean` bars (`0.416` and `0.438`) below `0.50`?** An AUC of `0.50` equals random coin-flipping. Both traditional models struggle to separate pre-seizure from normal brainwaves on brand-new patients because hand-crafted summary stats don't transfer well across different brains.
        """)

    st.markdown("---")

    # =========================================================
    # STEP B: 1D-CNN VS BASELINES (FIGURE 9 & TABLE 8)
    # =========================================================
    st.subheader("Step B: Does the 1D-CNN Beat the Baselines on Unseen Patients? (`Figure 9`)")
    st.write("Next, we tested our **1D-CNN** (which reads raw brainwave curves directly without hand-crafted features) under the exact same Leave-One-Patient-Out rules.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_09_chb_lopo_model_comparison.png", "Figure 9: Full LOPO comparison across Baselines and 1D-CNN at thresholds 0.1 and 0.5")
    with col2:
        st.success("""
        **Questions & Answers to Read This Chart:**
        * **What is the biggest takeaway?** Look at the **red `auc_mean` bars** across all 4 models: the 1D-CNN jumps to **`0.568`**, beating random chance (`0.50`) and outperforming both Logistic Regression (`0.416`) and Random Forest (`0.438`).
        * **Why are the red AUC bars identical (`0.568`) for `CNN (0.1)` and `CNN (0.5)`?** ROC-AUC measures the model's overall ranking ability across *all* possible thresholds—it stays the same regardless of where you set the alarm trigger.
        * **Why show both `CNN (0.1)` and `CNN (0.5)`?** They are the exact same trained neural network with two different alert dials:
          * **`CNN (0.1)` (Sensitive Dial):** Triggers an alarm at `10%` risk—achieving the highest seizure warning **Recall (`0.549` / 54.9%)** and highest **F1 score (`0.192`)** across every model tested.
          * **`CNN (0.5)` (Strict Dial):** Waits until `50%` risk—achieving the highest overall **Accuracy (`0.710` / 71.0%)**, but catching fewer subtle pre-seizure clips (`0.329` Recall).
        """)

    with st.expander("📋 View Exact Mean ± Std Comparison Table Across All 4 Models (Table 8)", expanded=True):
        st.dataframe(load_table("table_08_final_chb_comparison.csv"), use_container_width=True)

    st.markdown("---")

    # =========================================================
    # STEP C: PATIENT-SPECIFIC VS INDEPENDENT (FIGURE 15)
    # =========================================================
    st.subheader("Step C: Why Academic Papers Report '99% Accuracy' — The Reality Check (`Figure 15`)")
    st.write("Many published papers report near-perfect seizure prediction because they train and test on clips from the **same patient**. We ran a sanity-check experiment to compare that easy setup against our strict unseen-patient test.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_15_patient_specific_vs_independent.png", "Figure 15: Unseen-patient LOPO generalization (left) vs. same-patient upper bound (right)")
    with col2:
        st.warning("""
        **Questions & Answers to Read This Chart:**
        * **What is being compared on the horizontal axis (`Setting`)?**
          * **Left (`Patient-independent CNN (0.1)`):** Tested on a **brand-new patient** excluded from training (our realistic LOPO benchmark).
          * **Right (`Patient-specific LR`):** Trained on early recordings from a patient and tested on later recordings from that **exact same patient**.
        * **Why do all 4 bars shoot up near `1.0` on the right (`0.944` Accuracy, `0.842` F1, `0.948` Recall, `0.990` AUC)?** Every human brain has a unique electrical signature. When tested on the same person, even a simple linear model memorizes that patient's personal brainwaves.
        * **Why does this matter?** A new patient arriving at a hospital does not have hours of pre-recorded seizures to train a personal model. The left bars show the **honest clinical challenge**—and prove why cross-patient generalization is the true benchmark.
        """)

    st.markdown("---")

    # =========================================================
    # STEP D: SOURCE CNN TRAINING STABILITY (FIGURE 16)
    # =========================================================
    st.subheader("Step D: Training the Final Boston Source Model (`Figure 16`)")
    st.write("After validating the 1D-CNN architecture via LOPO, we trained one final source model across all 5 valid CHB-MIT patients to prepare for our cross-hospital test in Italy.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_16_chb_source_training_loss.png", "Figure 16: Training loss progression across 5 epochs on the combined CHB-MIT cohort")
    with col2:
        st.info("""
        **Questions & Answers to Read This Chart:**
        * **What does `Epoch` on the horizontal axis (`1.0` to `5.0`) mean?** One **Epoch** is one complete pass where the 1D-CNN reads through every training window across all 5 Boston patients. The 5 blue dots mark the error score after each full pass (`1, 2, 3, 4, 5`).
        * **What does `Training loss` on the vertical axis (`0.50` to `0.80`) mean?** It measures how far off the model's predictions are from the true labels (`BCEWithLogitsLoss`)—**lower means fewer mistakes**.
        * **What does the smooth downward curve prove?** Loss drops sharply from **`~0.805` (Epoch 1)** to **`~0.613` (Epoch 2)** and continues declining smoothly down to **`~0.482` (Epoch 5)**. This confirms stable learning without erratic jumps before saving the model for Act II.
        """)

    st.subheader("Detailed Per-Patient & Fold Tables")
    t1, t2, t3 = st.columns(3)
    with t1:
        with st.expander("📋 Table 5: Baseline Per-Patient Results"):
            st.dataframe(load_table("table_05_baseline_lopo_per_patient_results.csv"), use_container_width=True)
    with t2:
        with st.expander("📋 Table 6: Baseline Summary Stats"):
            st.dataframe(load_table("table_06_baseline_summary_statistics.csv"), use_container_width=True)
    with t3:
        with st.expander("📋 Table 7: CNN Summary Stats"):
            st.dataframe(load_table("table_07_cnn_summary_statistics.csv"), use_container_width=True)

    st.markdown("👉 **Next step:** Go to **4. Act II: Unseen Hospital (Siena)** in the sidebar to see what happens when we deploy this Boston-trained model to an unseen hospital in Italy.")

# =========================================================
# PAGE 4: CROSS-DATASET TRANSFER — SIENA (ACT II)
# =========================================================
elif page == "4. Act II: Unseen Hospital (Siena)":
    st.header("4. Act II: What Happens at a Completely Different Hospital? (Siena, Italy)")
    st.write("""
    Even when two hospitals use the same 16 electrode pairs and 256 Hz sampling rate, real-world clinical deployment faces **domain shift**: 
    differences in EEG recording hardware, reference-to-bipolar signal conversion, and adult (Siena, Italy) vs. pediatric (CHB-MIT, Boston) brain physiology. 
    In Act II, we tested whether our Boston-trained models survive this jump—and how much light local calibration is needed to adapt them.
    """)

    # Quick at-a-glance Act II KPI strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Zero-Shot CNN AUC", "0.395", "-0.173 drop from Boston (0.568)")
    m2.metric("Fine-Tuned CNN AUC", "0.589", "+0.194 recovery (beats Boston)")
    m3.metric("Fine-Tuned Recall (@ 0.1)", "76.2%", "+22.9% vs zero-shot (53.3%)")
    m4.metric("Fine-Tuned Accuracy (@ 0.1)", "67.9%", "+19.1% vs zero-shot (48.8%)")

    st.markdown("---")

    # =========================================================
    # STEP A: ZERO-SHOT CROSS-HOSPITAL DROP (FIGURE 12 & TABLE 9)
    # =========================================================
    st.subheader("Step A: Zero-Shot Deployment — Testing Boston Models Directly on Italy (`Figure 12`)")
    st.write("""
    **What we did:** We froze our Boston-trained models (**Logistic Regression**, **Random Forest**, and **1D-CNN**) and tested them **zero-shot**—meaning with *zero* Italian training examples—on **3,122 harmonized EEG windows** across 8 Siena patients.  
    **Why we did it:** To measure how badly a medical AI degrades when shipped "out-of-the-box" to a new hospital without local calibration.
    """)

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_12_chb_to_siena_drop.png", "Figure 12: ROC-AUC comparison when transferring models directly from CHB-MIT (blue) to Siena (orange)")
    with col2:
        st.warning("""
        **Questions & Answers to Read This Chart:**
        * **What do the two bars compare?** For each model on the horizontal axis (`LR`, `RF`, `CNN`), the **blue bar (`CHB AUC`)** shows its average score in Boston, and the **orange bar (`Siena AUC`)** shows its zero-shot score in Italy.
        * **Why did the `CNN` plummet from `0.568` (blue) to `0.395` (orange)?** Because the 1D-CNN reads raw waveform shapes directly. Subtle hardware and adult-vs-pediatric waveform shifts in Siena's recordings threw off its Boston-learned filters (`-0.173` AUC drop).
        * **Why did `LR` (`0.416` → `0.476`) and `RF` (`0.438` → `0.546`) look flat or slightly higher?** Hand-crafted summary statistics are less sensitive to raw waveform shifts, but as Table 9 shows, they still fail clinically because they trigger constant false alarms on Siena patients.
        """)

    with st.expander("📋 View Full Zero-Shot CHB-MIT → Siena Transfer Metrics (Table 9)", expanded=False):
        st.dataframe(load_table("table_09_chb_to_siena_transfer_results.csv"), use_container_width=True)

    st.markdown("---")

    # =========================================================
    # STEP B: TARGET-DOMAIN FINE-TUNING PROCESS (FIGURE 16b)
    # =========================================================
    st.subheader("Step B: Fixing Domain Shift — Light Local Fine-Tuning on 4 Siena Patients (`Figure 16b`)")
    st.write("""
    **What we did:** Instead of training a new model from scratch, we took the Boston-trained 1D-CNN and **fine-tuned it for just 5 epochs** using a small calibration group of **4 Siena patients (`PN00, PN01, PN03, PN12`)**, while keeping the remaining **4 Siena patients (`PN05, PN09, PN10, PN14` — 2,063 windows)** completely locked away as an unseen test set.  
    **Why we did it:** In a real hospital rollout, a clinic only has limited local recordings. Splitting strictly *by patient* prevents data leakage and tests whether calibrating on a few local patients generalizes to **brand-new patients** in that same hospital.
    """)

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_16b_siena_finetune_training_loss.png", "Figure 16b: Training-loss progression over 5 fine-tuning epochs on the 4 Siena calibration patients")
    with col2:
        st.info("""
        **Questions & Answers to Read This Chart:**
        * **Why does Epoch 1 start at a very high loss (`1.669`)?** In Boston (`Figure 16`), training started at `0.805`. Starting at **`1.669`** here is visual proof of **domain shift**: the Boston-trained neural network was initially surprised by Siena's signal amplitude and adult EEG patterns.
        * **What happens between Epoch 1 and Epoch 5?** Each dot marks one full pass (`Epoch 1` to `5`) over the 4 calibration patients. The error drops steeply to **`0.950` (Epoch 2)** and **`0.623` (Epoch 3)**, then settles smoothly down to **`0.411` (Epoch 5)**.
        * **Why stop at 5 epochs?** Stopping early lets the network adapt its filters to the Italian hospital's equipment without memorizing or overfitting the 4 calibration patients.
        """)

    st.markdown("---")

    # =========================================================
    # STEP C: ZERO-SHOT VS FINE-TUNED RECOVERY (FIGURE 13 & TABLE 10)
    # =========================================================
    st.subheader("Step C: Did Fine-Tuning Recover Performance on Unseen Siena Patients? (`Figure 13`)")
    st.write("""
    **What we did:** We evaluated the fine-tuned 1D-CNN on the **4 completely unseen held-out Siena patients** and compared it side-by-side against the unadapted zero-shot CNN at both alert thresholds (`0.5` strict vs. `0.1` sensitive).  
    **Why we did it:** To prove whether brief local domain adaptation restores reliable seizure prediction on brand-new patients in a new hospital.
    """)

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_13_finetuning_improvement.png", "Figure 13: Zero-shot vs. fine-tuned 1D-CNN performance on Siena at thresholds 0.5 and 0.1")
    with col2:
        st.success("""
        **Questions & Answers to Read This Chart:**
        * **What do the 4 model groups on the horizontal axis show?** The left two bars show the **zero-shot CNN** (before seeing any Siena data); the right two bars show the **fine-tuned CNN** (after 5 epochs of calibration on 4 patients).
        * **Look at the red `auc` bars (`0.395` → `0.589`):** Fine-tuning boosts ROC-AUC by **`+0.194`**, completely erasing the cross-hospital drop and outperforming our original Boston LOPO benchmark (`0.568`).
        * **Why is `CNN CHB→Siena fine-tuned (0.1)` the clear winner?** Combining local fine-tuning with the sensitive `0.1` alert threshold drives massive gains across every metric:
          * **Recall (green):** Jumps from `53.3%` to **`76.2%`** (`+22.9%` more pre-seizure warnings caught).
          * **Accuracy (blue):** Jumps from `48.8%` to **`67.9%`** (`+19.1%`).
          * **F1-Score (orange):** Jumps from `0.611` to **`0.797`** *(note: Siena's test windows contain a higher proportion of pre-ictal segments than Boston, which naturally lifts F1)*.
        """)

    with st.expander("📋 View Full Zero-Shot vs. Fine-Tuned Comparison Table (Table 10)", expanded=True):
        st.dataframe(load_table("table_10_fine_tuning_results.csv"), use_container_width=True)

    st.markdown("👉 **Next step:** Go to **5. Act III: Clinical Viability & Burden** in the sidebar to see how these window scores translate into real hospital bedside alarms per hour.")

# =========================================================
# PAGE 5: CLINICAL EVENT-LEVEL BURDEN (ACT III)
# =========================================================
elif page == "5. Act III: Clinical Viability & Burden":
    st.header("5. Act III: Is This Ready for a Hospital Bedside? (Clinical Burden Analysis)")
    st.write("""
    High window-level recall does not automatically make a medical device usable in an intensive care unit. 
    In Act III, we translated our Siena predictions into **event-level clinical metrics** to measure the real trade-off between 
    **catching every upcoming seizure** and **causing nurse alarm fatigue**.
    """)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Checks / Hour", "240 windows", "30s window, 15s step")
    m2.metric("Baseline False Burden", "~234–239 / hr", "Alarms on ~98% of windows")
    m3.metric("Fine-Tuned CNN Burden", "~170 / hr", "-29% fewer false alarms vs LR")
    m4.metric("Event Sensitivity", "~1.0 (100%)", "Catches advance event warnings")

    st.markdown("---")

    st.subheader("Step A: Advance Seizure Warnings vs. False Alarms Per Hour (`Figure 14`)")
    st.write("""
    **What we did:** We grouped continuous 30-second window predictions on Siena into seizure events and measured two clinical metrics: 
    **`event_sensitivity`** (fraction of seizure events preceded by at least one warning) and **`approx_false_positive_burden_per_hour`** 
    (how many false alarms fire per hour during normal interictal periods).  
    **Why we did it:** A model that predicts "Seizure!" 240 times an hour technically catches 100% of seizures, but is useless in a hospital. 
    Event-level evaluation exposes models that cheat via constant false alarms.
    """)

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_14_event_level_comparison.png", "Figure 14: Event-level sensitivity (blue) vs. approximate false-positive alarms per hour (orange) on Siena")
    with col2:
        st.warning("""
        **Questions & Answers to Read This Chart:**
        * **Where are the blue `event_sensitivity` bars?** Look right along the bottom baseline (`0` to `1`): the thin blue slivers sit at **`~1.0` (100%)** for all four models. They look tiny only because they are plotted on the same `0–250` vertical axis as the hourly false-alarm counts.
        * **Why do `LR` (`~239/hr`) and `RF (0.1)` (`~234/hr`) hit nearly `240`?** With 30-second windows overlapping by 50% (one step every 15 seconds), there are **240 windows in an hour**. Both traditional baselines simply trigger an alarm on almost *every single window*—proving their high sensitivity is an illusion caused by constant false alarms.
        * **How much better is the 1D-CNN?** The 1D-CNN slashes false-alarm burden by **~29% to 41%**, dropping to **`~142/hr`** (zero-shot) and **`~170/hr`** (fine-tuned) while still warning before seizures occur.
        """)

    with st.expander("📋 View Exact Event-Level Sensitivity & False-Alarm Metrics (Table 11)", expanded=True):
        st.dataframe(load_table("table_11_event_level_results.csv"), use_container_width=True)

    st.markdown("---")

    st.subheader("Step B: Key Takeaway & Next Steps for Real-World Deployment")
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        **Why `~170` Raw Window Alerts/Hour Is Expected (and Honest):**
        * Each 30-second window is scored independently without temporal smoothing.
        * Lowering the decision dial to `0.1` prioritizes catching subtle pre-seizure changes (`76.2%` window recall), which naturally allows isolated noisy windows to trigger single-step alerts.
        """)
    with c2:
        st.success("""
        **How to Bring False Alarms Below `< 1 / Hour` in Production:**
        * **Persistence Filtering ($k$-of-$n$ rule):** Only sound a bedside alarm when at least *4 out of 6* consecutive windows exceed the risk threshold.
        * **Refractory Cooldown Period:** Suppress duplicate alerts for 15–30 minutes after an initial warning is raised.
        """)

    st.markdown("👉 **Next step:** Go to **6. Appendix & Metadata** in the sidebar to view how every chapter maps to our research questions and literature benchmarks.")

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