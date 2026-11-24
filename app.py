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
# Helper Functions & Navigation State
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

PAGES = [
    "0. Executive Summary & Takeaways",
    "1. Architecture & Methodology",
    "2. Dataset Characteristics",
    "3. Act I: Unseen Patients (CHB-MIT)",
    "4. Act II: Unseen Hospital (Siena)",
    "5. Act III: Clinical Viability & Burden",
    "6. Appendix & Metadata"
]

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]

def go_to_page(target_page):
    st.session_state.current_page = target_page

def render_chapter_nav(current_idx):
    st.markdown("---")
    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
    with nav_left:
        if current_idx > 0:
            st.button(
                f"← Previous: {PAGES[current_idx - 1].split(':')[0]}",
                on_click=go_to_page,
                args=(PAGES[current_idx - 1],),
                use_container_width=True
            )
    with nav_mid:
        st.caption(f"Chapter {current_idx + 1} of {len(PAGES)} • {PAGES[current_idx]}")
    with nav_right:
        if current_idx < len(PAGES) - 1:
            st.button(
                f"Next: {PAGES[current_idx + 1].split(':')[0]} →",
                on_click=go_to_page,
                args=(PAGES[current_idx + 1],),
                type="primary",
                use_container_width=True
            )

# ---------------------------------------------------------
# Sidebar Navigation & Searchable Plain-English Glossary
# ---------------------------------------------------------
st.sidebar.title("🧭 Project Storyline")
page = st.sidebar.radio("Select Chapter/Section:", PAGES, key="current_page")

# Format: (Abbreviation / Term, Stands For (Full Form), Category, Plain-English Meaning)
GLOSSARY_TERMS = [
    # --- Brain & Medical ---
    ("EEG", "Electroencephalogram", "Brain & Medical",
     "A non-invasive test using small scalp sensors to record the brain's electrical waves over time."),
    ("CHB-MIT", "Children's Hospital Boston – Massachusetts Institute of Technology", "Brain & Medical",
     "Our primary pediatric scalp EEG dataset recorded in Boston, USA (10 patients preprocessed; 5 valid for LOPO)."),
    ("Siena", "Siena Scalp EEG Database (University of Siena, Italy)", "Brain & Medical",
     "Our external adult scalp EEG dataset recorded in Italy (8 patients), used to test cross-hospital transfer."),
    ("Interictal (Label = 0)", "Between Seizures (Clinical Term)", "Brain & Medical",
     "Normal, everyday brain activity recorded at least 2 hours away from any seizure."),
    ("Preictal (Label = 1)", "Before a Seizure (Clinical Term)", "Brain & Medical",
     "The 5-to-30-minute warning window right before a seizure starts—what the AI tries to detect."),
    ("Ictal & Post-ictal", "During Seizure & After Seizure (Clinical Terms)", "Brain & Medical",
     "Ictal is the active seizure itself; Post-ictal is the recovery period right after. Both are excluded so the model learns early warnings rather than active seizures."),

    # --- Signal & Data Engineering ---
    ("EDF", "European Data Format", "Signal & Data",
     "The standard medical file format hospitals use to store raw brainwave recordings."),
    ("Hz", "Hertz (Samples / Cycles per Second)", "Signal & Data",
     "Measurement speed. 256 Hz means the machine records 256 brainwave snapshots every second."),
    ("10-20 System (16 Channels)", "International 10–20 Electrode Placement System", "Signal & Data",
     "The standard scalp map for placing EEG sensors (spaced at 10% and 20% intervals). Both hospitals were aligned to the exact same 16 sensor pairs."),
    ("FP, F, C, P, O, T, Z", "Frontal-Pole, Frontal, Central, Parietal, Occipital, Temporal, Zero (Midline)", "Signal & Data",
     "The brain region letters in channel names (e.g., FP1-F7 connects the left front-pole sensor to the left frontal sensor)."),
    ("Bipolar vs. Referential", "Two-Electrode Difference vs. Single-Reference Recording", "Signal & Data",
     "Bipolar measures voltage difference between two neighboring sensors (e.g., FP1-F7); Referential measures each sensor against one shared point."),
    ("Band-pass Filter (0.5–40 Hz)", "Frequency Range Filter", "Signal & Data",
     "A digital filter that blocks slow sweat/breathing drift (<0.5 Hz) and wall-outlet electrical hum (>40 Hz), keeping true brainwaves."),
    ("Windowing & 50% Overlap", "Sliding Time-Clip Segmentation", "Signal & Data",
     "Slicing long recordings into 30-second clips, stepping forward every 15 seconds so no transition is missed."),
    (".npz & OOM", "NumPy Zipped Archive & Out-Of-Memory", "Signal & Data",
     "Saving processed clips in compressed .npz disk chunks so 31,147 windows load safely without crashing computer memory (OOM)."),

    # --- AI & Modeling ---
    ("1D-CNN", "One-Dimensional Convolutional Neural Network", "AI & Modeling",
     "A deep learning model that slides pattern detectors directly across raw brainwave curves over time."),
    ("ML & AI", "Machine Learning & Artificial Intelligence", "AI & Modeling",
     "Computer algorithms that learn patterns from historical patient data to make predictions on new recordings."),
    ("LR & RF", "Logistic Regression & Random Forest", "AI & Modeling",
     "Classic baseline ML models trained on 144 hand-crafted summary statistics rather than raw brainwave curves."),
    ("LOPO", "Leave-One-Patient-Out (Cross-Validation)", "AI & Modeling",
     "Training on 4 patients and testing on the 5th completely unseen patient to mimic a brand-new hospital admission."),
    ("Domain Shift", "Cross-Hospital Distribution Shift", "AI & Modeling",
     "The accuracy drop that happens when a model trained on one hospital's equipment and age group is tested on a different hospital."),
    ("Zero-Shot vs. Fine-Tuning", "Uncalibrated Transfer vs. Light Local Calibration", "AI & Modeling",
     "Zero-Shot tests the Boston model on Italy with zero local training; Fine-Tuning adapts it for 5 passes on 4 local Italian patients."),
    ("Epoch & BCEWithLogitsLoss", "Full Training Pass & Binary Cross-Entropy Loss", "AI & Modeling",
     "One Epoch is one complete pass through the training data; Training Loss is the error score (lower means fewer mistakes)."),
    ("Dropout & Sigmoid", "Regularization Layer & Probability Activation Function", "AI & Modeling",
     "Dropout randomly disables 30% of connections during training to prevent memorization; Sigmoid turns the final output into a 0%–100% risk score."),

    # --- Metrics & Alerts ---
    ("ROC-AUC", "Receiver Operating Characteristic – Area Under the Curve", "Metrics & Alerts",
     "Overall ability to rank warning clips higher than normal clips across all thresholds (0.50 = random coin flip; 1.00 = perfect)."),
    ("Recall (Sensitivity)", "True Positive Rate", "Metrics & Alerts",
     "Out of all true pre-seizure warning clips, the percentage the model successfully caught."),
    ("F1-Score", "Harmonic Mean of Precision and Recall", "Metrics & Alerts",
     "A single combined score balancing how many warnings are caught (Recall) against false alarms (Precision)."),
    ("Std", "Standard Deviation", "Metrics & Alerts",
     "Measures how much a model's score varies from patient to patient across LOPO folds."),
    ("Threshold (0.1 vs. 0.5)", "Decision Probability Cutoff", "Metrics & Alerts",
     "The alarm trigger dial: 0.1 triggers an alert at 10% risk (catches more seizures, more false alarms); 0.5 waits until 50% risk."),
    ("Event Sensitivity & Burden/hr", "Seizure-Level Detection Rate & False Positives per Hour", "Metrics & Alerts",
     "Event Sensitivity checks if at least one warning fired before a seizure; Burden/hr counts false alarms per hour during normal periods."),

    # --- Clinical & Future Terms ---
    ("ICU & EMU", "Intensive Care Unit & Epilepsy Monitoring Unit", "Clinical & Research",
     "Specialized hospital wards where continuous bedside EEG monitoring is used to watch critically ill or epilepsy patients."),
    ("ICA & EMG", "Independent Component Analysis & Electromyography", "Clinical & Research",
     "ICA is a signal-cleaning method used to separate and remove muscle tension (EMG) and eye-blink noise from brainwaves."),
    ("RNS & VNS", "Responsive Neurostimulation & Vagus Nerve Stimulation", "Clinical & Research",
     "Implanted medical devices that deliver targeted electrical pulses to stop a seizure before it spreads."),
    ("RQ (RQ1–RQ3)", "Research Question", "Clinical & Research",
     "The three core questions guiding this dissertation: unseen-patient generalization (RQ1), cross-hospital transfer (RQ2), and clinical burden (RQ3).")
]

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Plain-English Glossary (Searchable)", expanded=True):
    search_term = st.text_input("🔍 Filter terms (e.g. AUC, EDF, LOPO):", placeholder="Type any abbreviation...").strip().lower()
    if search_term:
        matches = [
            item for item in GLOSSARY_TERMS
            if search_term in item[0].lower()
            or search_term in item[1].lower()
            or search_term in item[2].lower()
            or search_term in item[3].lower()
        ]
        if matches:
            for term, stands_for, cat, desc in matches:
                st.markdown(f"**{term}** — *{stands_for}*  \n{desc}")
        else:
            st.caption("No matching term found. Try 'AUC', 'EDF', 'Hz', 'CNN', or 'LOPO'.")
    else:
        st.caption("Showing core abbreviations — type above to search all 32 terms:")
        core_terms = {"EEG", "CHB-MIT", "1D-CNN", "LOPO", "ROC-AUC", "Domain Shift"}
        for term, stands_for, _, desc in GLOSSARY_TERMS:
            if term in core_terms:
                st.markdown(f"**{term}** (*{stands_for}*): {desc}")

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.title("🧠 Generalising EEG-Based Seizure Prediction")
st.markdown("**Can a deep learning model warn patients up to 25 minutes before an epileptic seizure—even across unseen patients and different hospitals?**")
st.markdown("---")

# =========================================================
# PAGE 0: EXECUTIVE SUMMARY & PROJECT TAKEAWAYS
# =========================================================
if page == PAGES[0]:
    st.header("Executive Summary: End-to-End Project Overview")
    st.write("""
    This project investigates whether deep learning can predict epileptic seizures **5 to 30 minutes before onset** 
    not only on **brand-new patients**, but also across **completely different hospitals and recording equipment**.
    """)

    with st.expander("💡 New to Brainwaves & Medical AI? Read This 30-Second Weather Analogy", expanded=True):
        st.markdown("""
        Think of seizure prediction like a **storm barometer**:
        - ☀️ **Interictal (`Label = 0`):** Clear, calm weather—normal brain activity at least **2 hours away** from any seizure.
        - 🌦️ **Preictal (`Label = 1`):** Falling barometric pressure **30 to 5 minutes before a storm**—subtle electrical shifts our AI tries to spot early so a patient can sit down or take rescue medication.
        - ⛈️ **Ictal:** The active thunderstorm itself—excluded from training because warning someone *during* a seizure is too late.
        - 🎛️ **Alert Threshold (`0.1` vs `0.5`):** How sensitive you set the storm siren—a low setting (`0.1`) catches almost every approaching storm (`76.2%` recall) at the cost of more false alarms.
        """)

    # Top-level KPI Summary Strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total EEG Data", "31,147 Windows", "CHB-MIT (28,025) + Siena (3,122)")
    k2.metric("Unseen-Patient AUC", "0.568", "1D-CNN vs 0.438 Best Baseline")
    k3.metric("Cross-Hospital Recovery", "0.589 AUC", "+0.194 after 5-epoch fine-tuning")
    k4.metric("Fine-Tuned Sensitivity", "76.2% Window / 100% Event", "At 0.1 clinical alert threshold")

    st.markdown("---")

    # =========================================================
    # 1. WHAT WE DID
    # =========================================================
    st.subheader("1. What We Did (Methodology & Experimental Design)")
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        **Data Engineering & Harmonization**
- **Unified Two Distinct Hospitals:** Combined pediatric scalp EEG from **CHB-MIT (Boston, USA)** and adult scalp EEG from **Siena (Italy)**.
- **Signal & Channel Standardization:** Mapped differing electrode setups into a common **16-channel bipolar montage**, resampled all signals to **256 Hz**, and applied a **0.5–40 Hz bandpass filter**.
- **Leakage-Free Windowing:** Sliced recordings into **30-second windows (50% overlap)**, labeling **30 to 5 minutes before seizure onset** as *pre-ictal* and enforcing a strict **2-hour exclusion buffer** around seizures for *interictal* (normal) baseline data.
- **Memory-Safe Pipeline:** Engineered `.npz` chunk caching to process **31,147 windows** (~236 MB per channel tensor set) without Out-Of-Memory crashes.
        """)
    with c2:
        st.info("""
        **Three-Act Validation Strategy (RQ1 → RQ3)**
- **Act I (`RQ1`) — Unseen Patients (CHB-MIT LOPO):** Evaluated Logistic Regression, Random Forest (144 hand-crafted features), and a raw-waveform **1D-CNN** using strict **Leave-One-Patient-Out (LOPO)** cross-validation across all 5 valid seizure patients (`chb01`–`chb05`), plus a same-patient sanity check.
- **Act II (`RQ2`) — Unseen Hospital (CHB-MIT → Siena):** Tested the Boston-trained models **zero-shot** on 8 Italian patients (`3,122` windows), then **fine-tuned the 1D-CNN for 5 epochs** on 4 calibration patients (`PN00, PN01, PN03, PN12` — `1,059` windows) and tested on **4 completely unseen holdout patients** (`PN05, PN09, PN10, PN14` — `2,063` windows).
- **Act III (`RQ3`) — Clinical Bedside Viability:** Converted window scores into **event-level seizure sensitivity** and **false-positive alarms per hour**.
        """)

    st.markdown("---")

    # =========================================================
    # 2. WHAT OUTPUT WE GOT & 3. WHY WE GOT IT
    # =========================================================
    st.subheader("2. What Output We Got — And Why We Got It")

    summary_results = pd.DataFrame([
        {
            "Experiment Stage": "Same-Patient Sanity Check (CHB-MIT)",
            "Best Model": "Patient-Specific LR",
            "ROC-AUC": "0.990",
            "Recall / Sensitivity": "94.8%",
            "Key Takeaway": "Upper-bound ceiling when training & testing on the same person"
        },
        {
            "Experiment Stage": "Act I (RQ1): Unseen Patients (CHB-MIT LOPO)",
            "Best Model": "1D-CNN (Threshold 0.1)",
            "ROC-AUC": "0.568",
            "Recall / Sensitivity": "54.9%",
            "Key Takeaway": "Beats LR (0.416 AUC) & RF (0.438 AUC) on brand-new patients"
        },
        {
            "Experiment Stage": "Act IIa (RQ2): Zero-Shot New Hospital (Siena)",
            "Best Model": "1D-CNN Zero-Shot (0.1)",
            "ROC-AUC": "0.395",
            "Recall / Sensitivity": "53.3%",
            "Key Takeaway": "-0.173 AUC drop proves severe cross-hospital domain shift"
        },
        {
            "Experiment Stage": "Act IIb (RQ2): Fine-Tuned New Hospital (Siena)",
            "Best Model": "1D-CNN Fine-Tuned (0.1)",
            "ROC-AUC": "0.589",
            "Recall / Sensitivity": "76.2%",
            "Key Takeaway": "+0.194 AUC recovery; outperforms original Boston benchmark"
        },
        {
            "Experiment Stage": "Act III (RQ3): Clinical Event Burden (Siena)",
            "Best Model": "1D-CNN Fine-Tuned (0.1)",
            "ROC-AUC": "0.589",
            "Recall / Sensitivity": "100% Event",
            "Key Takeaway": "Cuts false alerts to ~170.6/hr vs ~239.4/hr for LR (max 240/hr)"
        }
    ])
    st.dataframe(summary_results, use_container_width=True, hide_index=True)

    col_why1, col_why2 = st.columns(2)
    with col_why1:
        st.warning("""
        **Why Did Performance Drop Across Patients & Hospitals?**
- **Unique Brain Signatures (`0.990` vs `0.568` AUC):** Every individual has distinct baseline EEG rhythms. Same-patient splits allow models to memorize personal signatures, whereas unseen-patient LOPO forces true generalization.
- **Hardware & Cohort Domain Shift (`0.568` → `0.395` AUC):** Even with identical 16-channel names and 256 Hz sampling rates, Siena uses adult patients and referential-to-bipolar signal conversion—causing uncalibrated Boston filters to misfire.
        """)
    with col_why2:
        st.success("""
        **Why Did 1D-CNN + Light Fine-Tuning Recover Performance?**
- **Raw Temporal Filters Beat Summary Stats:** Unlike Logistic Regression or Random Forest (which rely on 144 static summary numbers), the 1D-CNN learns wave shapes directly from raw EEG.
- **Rapid Target Adaptation (`0.395` → `0.589` AUC):** Fine-tuning for just 5 epochs on 4 local Siena patients realigned the network's convolutional filters to Italian hospital hardware without overfitting.
- **Threshold & Window Independence:** Lowering the alert dial to `0.1` catches subtle pre-ictal shifts (`76.2%` recall), while scoring 30s windows independently leaves ~170 raw alerts/hr before smoothing.
        """)

    st.markdown("---")

    # =========================================================
    # 4. WHAT DO WE UNDERSTAND IN THE END
    # =========================================================
    st.subheader("3. What Do We Understand From This Project in the End?")
    st.success("""
1. **'99% Accuracy' Papers Are Misleading for New Patients:** True clinical readiness must be measured via **Leave-One-Patient-Out (LOPO)** and **cross-dataset** testing, not overlapping splits from the same patient.
2. **Data Harmonization Alone Does Not Solve Domain Shift:** Standardizing channels and sampling rates is necessary, but physiological and hardware differences across hospitals still degrade zero-shot models.
3. **Light Local Calibration Is Practical & Effective:** A pre-trained neural network does not need to be rebuilt from scratch for every hospital—brief fine-tuning on a small local patient cohort restores and even improves predictive power.
4. **Window Metrics Are Not Enough for Hospital Deployment:** Evaluating **event-level sensitivity alongside hourly false-alarm burden** is essential to bridge the gap between ML benchmarks and bedside usability.
    """)

    st.markdown("---")

    # =========================================================
    # 5. HOW IT CAN BE IMPROVED & 6. FUTURE APPLICATIONS
    # =========================================================
    col_imp, col_app = st.columns(2)

    with col_imp:
        st.subheader("4. How This Project Can Be Improved")
        st.markdown("""
- **Temporal Alarm Smoothing ($k$-of-$n$ Persistence):** Require at least *4 out of 6 consecutive windows* to exceed the risk threshold plus a 15–30 minute cooldown window—reducing raw alerts (`~170/hr`) to **< 1 false alarm per hour**.
- **Advanced Architectures & Domain Adaptation:** Explore hybrid **CNN-Transformer / Conformer** models, adversarial domain adaptation, and per-patient adaptive normalization to improve zero-shot transfer without labels.
- **Automated Artifact Rejection:** Integrate Independent Component Analysis (ICA) or wavelet denoising to strip muscle (EMG) and eye-blink artifacts before inference.
- **Multi-Center Scale & Uncertainty Estimation:** Train across larger repositories (e.g., Temple University Hospital EEG Corpus) and add conformal prediction to flag noisy/unreliable readings.
        """)

    with col_app:
        st.subheader("5. Future Real-World Applications")
        st.markdown("""
- **Wearable Seizure Warning Headsets:** Pair lightweight, dry-electrode EEG headbands with a smartphone app to give epilepsy patients a **5–30 minute safety alert** to sit down, alert caregivers, or take fast-acting rescue medication.
- **Hospital ICU & Epilepsy Monitoring Units (EMUs):** Provide real-time bedside risk dashboards that warn neurologists and nurses ahead of dangerous seizure clusters or status epilepticus.
- **Plug-and-Calibrate Hospital Onboarding:** Deploy a foundation seizure-prediction model that hospitals can rapidly calibrate to their local EEG machines using just a handful of local patient recordings.
- **Closed-Loop Neuromodulation Devices:** Serve as an intelligent trigger for Responsive Neurostimulation (RNS) or Vagus Nerve Stimulation (VNS) implants to deliver targeted pulses and abort seizures before clinical onset.
        """)

    render_chapter_nav(0)

# =========================================================
# PAGE 1: ARCHITECTURE & METHODOLOGY
# =========================================================
elif page == PAGES[1]:
    st.header("1. Architecture & Methodology: Turning Raw Brainwaves into Early Warnings")
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
    st.caption("💡 **TL;DR Takeaway:** Only calm baseline brainwaves (≥2 hrs away) and early warning waves (30–5 mins before onset) are kept; the active seizure and recovery phases are excluded.")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_01_eeg_phase_timeline.png", "Figure 1: The five clinical phases around an epileptic seizure")
    with col2:
        st.info("""
        **How to read this diagram (Left → Right):**
- **What do the numbers `0` to `7` mean?** They are an **illustrative timeline** (not drawn to literal minute scale) showing the order of brain states from left to right. There is no vertical axis because this is a single timeline bar.
- 🟦 **Blue (`0` to `2`) — Inter-ictal (`Label = 0`):** Normal, everyday brain activity recorded **at least 2 hours away** from any seizure. Kept as our "safe" baseline.
- 🟧 **Orange (`2` to `4.5`) — Pre-ictal (`Label = 1`):** The **30 to 5 minute window before a seizure**. This 25-minute stretch is our target early-warning zone.
- 🟩 **Green (`4.5` to `5`) — Excluded Buffer:** The final **5 minutes** right before the seizure starts. Thrown away so the model learns *early* warning signs rather than cheating by detecting the seizure starting.
- 🟥 **Red (`5` to `5.8`) — Ictal:** The actual seizure happening in real time. Thrown away (detecting an active seizure is too late to warn the patient).
- 🟪 **Purple (`5.8` to `7`) — Post-ictal:** Recovery brainwaves right after a seizure ends. Thrown away so tired/recovering brainwaves aren't mistaken for normal baseline.
        """)

    st.markdown("---")

    # --- FIGURE 4 ROW ---
    st.markdown("#### 2. Slicing Continuous Brainwaves into Overlapping Clips (`Figure 4`)")
    st.caption("💡 **TL;DR Takeaway:** Sliding 30-second windows with 50% overlap doubles our training clips and ensures no pre-seizure transition gets cut in half.")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_04_windowing_labelling.png", "Figure 4: Sliding overlapping windows across the timeline")
    with col2:
        st.info("""
        **How to read every part of this diagram:**
- **Horizontal Axis (`0` to `200`):** Time moving forward in minutes (illustrative scale).
- **Vertical Dashed Line (`x = 180`):** **Seizure Onset**—the exact moment the seizure begins.
- **Light-Blue Shaded Band (`x = 120` to `150`):** **Pre-ictal Warning Zone**—a 30-minute illustrative warning window ending well before onset at `180` (in our real pipeline, this zone is **30 to 5 minutes** before onset).
- **White Gap (`x = 150` to `180`):** The **excluded buffer zone** right before the seizure—any clip falling here is discarded.
- **Why does the horizontal bar change colors?** Each colored block represents one **sliding time window** (`30 seconds` in our real model). Because each new window slides forward by only **half a window (`15 seconds` / 50% overlap)**, each color transition shows where the next overlapping clip begins.
- **How clips get labeled:** Clips inside the light-blue shaded zone (olive & cyan segments) are labeled **`1` (Warning)**. Clips far to the left (at least 2 hours away in real data) are labeled **`0` (Normal)**. Clips touching the buffer or dashed seizure line (far right) are dropped.
        """)

    st.markdown("---")

    # =========================================================
    # STEP B: HARMONIZING TWO DIFFERENT HOSPITALS (FIGURE 3 & 5)
    # =========================================================
    st.subheader("Step B: Cleaning & Standardizing Two Different Hospitals")
    st.write("Hospital EEG files are huge, noisy, and recorded differently depending on the country and equipment. Before any AI model can learn from them, every recording passes through an automated assembly line.")

    # --- FIGURE 3 ROW ---
    st.markdown("#### 1. The 7-Step Signal Assembly Line (`Figure 3`)")
    st.caption("💡 **TL;DR Takeaway:** Every raw hospital file goes through identical cleaning, resampling (256 Hz), noise filtering (0.5–40 Hz), and disk caching before reaching the model.")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_03_preprocessing_pipeline.png", "Figure 3: Step-by-step pipeline from raw hospital files to model-ready data")
    with col2:
        st.info("""
        **How to read this diagram (Follow the arrows Left → Right):**
- *(Note: There are no numbers or axes here—each blue box is one processing stage in chronological order.)*
- **Box 1 — Raw EDF loading:** Opens the raw hospital brainwave recordings (stored in **EDF**, the standard medical file format for EEG).
- **Box 2 — Channel cleaning / harmonisation:** Strips out non-brain sensors (like heart-rate or eye-movement channels) and standardizes electrode names so both hospitals match.
- **Box 3 — Resampling to 256 Hz:** Adjusts recording speeds so every file captures exactly **256 brainwave snapshots per second** (Siena was recorded twice as fast at 512 Hz, so we downsample it to match Boston's 256 Hz).
- **Box 4 — Band-pass filtering (0.5–40 Hz):** Acts like an audio equalizer—blocks slow baseline drift (`< 0.5 Hz`, like breathing/sweat) and high-frequency electrical hum (`> 40 Hz`, like wall outlets), keeping only true brainwave frequencies.
- **Box 5 — 30 s windows with 50% overlap:** Slices the continuous recording into **30-second clips**, stepping forward every **15 seconds** (as shown in Figure 4).
- **Box 6 — Label generation:** Tags each 30-second clip as **`1` (Pre-ictal warning)** or **`0` (Inter-ictal normal)** based on its distance from a seizure.
- **Box 7 — Chunk caching / feature extraction:** Saves the cleaned clips onto disk as compressed `.npz` chunks (for the 1D-CNN) and extracts 144 summary stats (for the baseline models) so the computer never runs out of memory.
        """)

    st.markdown("---")

    # --- FIGURE 5 ROW ---
    st.markdown("#### 2. Making Boston & Italy Speak the Same Sensor Language (`Figure 5`)")
    st.caption("💡 **TL;DR Takeaway:** Siena's single-sensor recordings are mathematically subtracted into pairs so both hospitals share the exact same 16 bipolar channels.")
    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_05_channel_harmonisation.png", "Figure 5: Unifying two different electrode wiring methods into one 16-channel standard")
    with col2:
        st.info("""
        **How to read the three boxes & inward arrows:**
- **Why do both outer arrows point toward the center?** Because two completely different hospital setups (left and right) are being converted into **one shared middle format** so a single AI model can read both.
- **Left Box — CHB-MIT (Bipolar channels):** The Boston hospital recorded brainwaves as **pairs** (e.g., `FP1-F7`), measuring the electrical voltage *difference* between two neighboring scalp sensors.
- **Right Box — Siena (Referential → bipolar):** The Italian hospital recorded each sensor individually against a single reference point (e.g., `Fp1`, `F7`) and used older naming labels (like `T3` instead of `T7`). Our code renames the sensors and subtracts pairs (`FP1 minus F7`) to reconstruct the exact same paired signals.
- **Center Box — Common 16-channel harmonised montage:** A **"montage"** is simply the map of sensor connections on a patient's head. Both datasets end up with the **exact same 16 paired brain channels** in the exact same order.
        """)

    with st.expander("🔍 View the Exact 16 Harmonized 10-20 Bipolar Channels"):
        st.code("['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',\n 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'P8-O2', 'FZ-CZ']", language="python")

    st.markdown("---")

    # =========================================================
    # STEP C: THE 1D-CNN ARCHITECTURE (FIGURE 8)
    # =========================================================
    st.subheader("Step C: How the AI Model Reads Raw Brainwaves (`Figure 8`)")
    st.caption("💡 **TL;DR Takeaway:** Rather than relying on human-guessed summary numbers, the 1D-CNN scans all 16 raw brainwave curves simultaneously and outputs a 0%–100% seizure risk score.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_08_cnn_architecture.png", "Figure 8: High-level flow of a 30-second brainwave clip through the 1D-CNN")
    with col2:
        st.success("""
        **How to read this diagram (Follow the arrows Left → Right):**
- *(Note: There are no axes here—each blue box shows how one 30-second brainwave clip is transformed step by step into a final warning score.)*
- **Box 1 — `Input (16 × 7680)`:** What do these two numbers mean?
  - **`16`** = The **16 harmonized brain channels** (sensor pairs across the scalp).
  - **`7680`** = The exact number of brainwave measurements in one 30-second clip (`30 seconds × 256 snapshots per second = 7,680 time steps`).
- **Box 2 — `Conv Block 1` (Early Pattern Detector):** Slides small digital filters across the raw waves to spot basic, fast shapes—like sharp spikes, sudden voltage jumps, or rhythmic ripples.
- **Box 3 — `Conv Block 2` (Deep Pattern Detector):** Combines those basic shapes into higher-level patterns across multiple brain channels that signal early pre-seizure brain changes.
- **Box 4 — `Pooling / Flatten` (Summarizer):** Compresses thousands of time-step readings down into a compact list of the **128 strongest pattern scores** found anywhere in that 30-second window.
- **Box 5 — `Dense + Dropout` (Decision Maker & Overfitting Guard):** Weighs all 128 pattern scores together to make a decision. **`Dropout` (30%)** randomly switches off 30% of connections during training so the model cannot simply memorize one patient's quirks.
- **Box 6 — `Sigmoid output` (Final Risk Score):** At prediction time, converts the network's raw score into a clean probability between **`0.0` (0% risk — Normal)** and **`1.0` (100% risk — Impending Seizure)**. If this score crosses our alert threshold (`0.5` or `0.1`), an early warning is triggered.
        """)

    with st.expander("📋 View Exact Preprocessing & Labelling Parameters (Table 3)"):
        st.dataframe(load_table("table_03_preprocessing_labelling_parameters.csv"), use_container_width=True)

    render_chapter_nav(1)

# =========================================================
# PAGE 2: DATASET CHARACTERISTICS
# =========================================================
elif page == PAGES[2]:
    st.header("2. Dataset Characteristics: Boston vs. Italy")
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
    st.caption("💡 **TL;DR Takeaway:** CHB-MIT has nearly 9× more windows and is dominated by normal recordings, whereas Siena is smaller and richer in pre-seizure clips.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_02_chb_vs_siena_comparison.png", "Figure 2: Side-by-side comparison of patient counts and 30-second window counts across CHB-MIT and Siena")
    with col2:
        st.info("""
        **How to read every bar and axis on this chart:**
- **Horizontal Axis (`Dataset`):** Two hospital groups—**CHB-MIT** (Boston, left) and **Siena** (Italy, right).
- **Vertical Axis (`Count`):** Total count from `0` up to `28,025`.
- **What the 4 colored bars mean in each group (Left → Right):**
  - 🟦 **Blue (`Patients used`):** **`10`** in CHB-MIT vs. **`8`** in Siena. *(Why do these bars look invisible? Because `10` and `8` sit right on the bottom line when plotted on a `0–28,000` scale!)*
  - 🟧 **Orange (`Windows` — Total clips):** **`28,025`** in CHB-MIT vs. **`3,122`** in Siena. CHB-MIT has nearly **9× more** total data.
  - 🟩 **Green (`Preictal windows` — Warning clips):** **`1,742`** in CHB-MIT vs. **`2,359`** in Siena.
  - 🟥 **Red (`Interictal windows` — Normal clips):** **`26,283`** in CHB-MIT vs. **`763`** in Siena.
- **Key Visual Contrast:** On **CHB-MIT**, the red normal bar towers over the green warning bar. On **Siena**, the pattern reverses—the green warning bar (`2,359`) is taller than the red normal bar (`763`).
        """)

    with st.expander("📋 View Exact Dataset Characteristics Table (Table 2)", expanded=True):
        st.dataframe(load_table("table_02_dataset_characteristics.csv"), use_container_width=True)

    st.markdown("---")

    # =========================================================
    # SECTION B: SEVERE CLASS IMBALANCE IN CHB-MIT (FIGURE 6)
    # =========================================================
    st.subheader("Step B: Why 'Accuracy' Can Be Misleading — Severe Class Imbalance (`Figure 6`)")
    st.caption("💡 **TL;DR Takeaway:** Because 93.8% of CHB-MIT clips are normal, a lazy model that never predicts a seizure still gets 93.8% accuracy—which is why we rely on Recall and ROC-AUC.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_06_chb_label_distribution.png", "Figure 6: Total Interictal (normal) vs. Preictal (warning) 30-second clips in CHB-MIT")
    with col2:
        st.warning("""
        **How to read every part of this chart:**
- **Horizontal Axis:** The two brain states in CHB-MIT—**Interictal** (left) and **Preictal** (right).
- **Vertical Axis (`Number of windows`):** Runs from `0` to `25,000+` in steps of `5,000`.
- **Left Bar (`Interictal` — Normal):** Reaches just above `25,000`—specifically **`26,283` windows (`93.8%`)**.
- **Right Bar (`Preictal` — Warning):** Sits near the bottom well below `5,000`—specifically **`1,742` windows (`6.2%`)**.
- **Why this matters for evaluating AI:** With a **15-to-1 imbalance**, a useless model that simply predicts *"No Seizure"* 100% of the time would still score **93.8% accuracy** while missing every seizure! That is why we judge models using **Recall (Sensitivity)** and **ROC-AUC**, and lower our alert threshold to **`0.1`**.
        """)

    st.markdown("---")

    # =========================================================
    # SECTION C: PATIENT-BY-PATIENT VOLUME (FIGURE 6B & TABLE 4)
    # =========================================================
    st.subheader("Step C: Recording Volume Across Individual Patients (`Figure 6b`)")
    st.caption("💡 **TL;DR Takeaway:** Only patients `chb01`–`chb05` contained valid pre-seizure warning windows after strict 30-to-5 minute filtering, forming our 5-patient LOPO cohort.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_06b_chb_windows_by_patient.png", "Figure 6b: Total preprocessed 30-second windows per patient across all 10 CHB-MIT subjects")
    with col2:
        st.info("""
        **How to read every part of this chart:**
- **Horizontal Axis:** All **10 preprocessed CHB-MIT patients**, sorted from the largest recording volume on the left (`chb04`) to the smallest on the right (`chb12`).
- **Vertical Axis (`Number of windows`):** Runs from `0` to `3,500` in steps of `500`.
- **Highest vs. Lowest Bars:**
  - **Tallest (`chb04`):** **`3,488`** windows (just under the `3,500` gridline).
  - **Shortest (`chb12`):** **`1,680`** windows (just above `1,500`, because 3 corrupted files failed the 16-channel check during cleaning).
- **Why does Figure 6b show 10 patients while Table 4 below lists 5?**
  - Patients **`chb01` through `chb05`** (5 patients) all have both normal and pre-seizure warning clips (`204` to `523` preictal windows each), making them valid for **Leave-One-Patient-Out (LOPO)** testing.
  - Patients **`chb11` through `chb15`** had **`0` valid preictal windows** after applying the strict 30-to-5 minute window rules, so they are excluded from LOPO test folds.
        """)

    with st.expander("📋 View Valid LOPO Patient Cohort Breakdown (Table 4)", expanded=True):
        st.dataframe(load_table("table_04_valid_chb_patient_cohort_summary.csv"), use_container_width=True)

    render_chapter_nav(2)

# =========================================================
# PAGE 3: CHB-MIT GENERALIZATION (ACT I)
# =========================================================
elif page == PAGES[3]:
    st.header("3. Act I (RQ1): Can the Model Generalize to Unseen Patients?")
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
    st.caption("💡 **TL;DR Takeaway:** Both Logistic Regression (`0.416` AUC) and Random Forest (`0.438` AUC) score below random chance (`0.50`) on unseen patients.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_07_baseline_model_comparison.png", "Figure 7: Leave-One-Patient-Out (LOPO) mean scores for traditional statistical baselines")
    with col2:
        st.info("""
        **Questions & Answers to Read This Chart:**
- **What are the axes?** The horizontal axis (`model`) compares our two baseline algorithms; the vertical axis (`Score`) shows their average score (`0.0` to `0.65`) across all 5 unseen patients.
- **Why is Logistic Regression's blue `accuracy_mean` bar (`0.614`) tall while its green `recall_mean` bar (`0.239`) is low?** Because 93.8% of the data is normal, Logistic Regression plays it safe by mostly guessing *"Normal"*—missing **76%** of pre-seizure warning clips.
- **What does `(0.1)` mean next to Random Forest?** By default, models only trigger an alert when risk exceeds `50%` (`0.5`). Lowering that trigger to `10%` (`0.1`) makes Random Forest much more sensitive—pushing its green Recall bar up to **`0.504`**, though extra false alarms pull its blue Accuracy bar down to **`0.433`**.
- **Why are both red `auc_mean` bars (`0.416` and `0.438`) below `0.50`?** An AUC of `0.50` equals random coin-flipping. Both traditional models struggle to separate pre-seizure from normal brainwaves on brand-new patients because hand-crafted summary stats don't transfer well across different brains.
        """)

    st.markdown("---")

    # =========================================================
    # STEP B: 1D-CNN VS BASELINES (FIGURE 9 & TABLE 8)
    # =========================================================
    st.subheader("Step B: Does the 1D-CNN Beat the Baselines on Unseen Patients? (`Figure 9`)")
    st.caption("💡 **TL;DR Takeaway:** Learning directly from raw waveforms pushes the 1D-CNN to `0.568` ROC-AUC—beating both statistical baselines on every ranking metric.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_09_chb_lopo_model_comparison.png", "Figure 9: Full LOPO comparison across Baselines and 1D-CNN at thresholds 0.1 and 0.5")
    with col2:
        st.success("""
        **Questions & Answers to Read This Chart:**
- **What is the biggest takeaway?** Look at the **red `auc_mean` bars** across all 4 models: the 1D-CNN jumps to **`0.568`**, beating random chance (`0.50`) and outperforming both Logistic Regression (`0.416`) and Random Forest (`0.438`).
- **Why are the red AUC bars identical (`0.568`) for `CNN (0.1)` and `CNN (0.5)`?** ROC-AUC measures the model's overall ranking ability across *all* possible thresholds—it stays the same regardless of where you set the alarm trigger.
- **Why show both `CNN (0.1)` and `CNN (0.5)`?** They are the exact same trained neural network with two different alert dials:
  - **`CNN (0.1)` (Sensitive Dial):** Triggers an alarm at `10%` risk—achieving the highest seizure warning **Recall (`0.549` / 54.9%)** and highest **F1 score (`0.192`)** across every model tested.
  - **`CNN (0.5)` (Strict Dial):** Waits until `50%` risk—achieving the highest overall **Accuracy (`0.710` / 71.0%)**, but catching fewer subtle pre-seizure clips (`0.329` Recall).
        """)

    # Interactive Per-Patient LOPO Explorer
    st.markdown("#### 🔍 Interactive Explorer: How Did Each Unseen Patient Score?")
    st.write("Averages hide how much human brains differ from one another. Select a patient below to inspect their individual unseen-test score:")

    lopo_patient_data = pd.DataFrame([
        {"Patient": "chb01", "Warning Clips": "523 / 3,282", "ROC-AUC": 0.551, "CNN (0.1) Recall": "81.5%", "CNN (0.1) Acc": "39.5%", "CNN (0.5) Recall": "32.9%", "CNN (0.5) Acc": "61.7%", "Takeaway": "Strong sensitivity (81.5% caught at 0.1 threshold)."},
        {"Patient": "chb02", "Warning Clips": "204 / 2,844", "ROC-AUC": 0.753, "CNN (0.1) Recall": "94.1%", "CNN (0.1) Acc": "41.2%", "CNN (0.5) Recall": "68.1%", "CNN (0.5) Acc": "65.3%", "Takeaway": "Best generalization: 0.753 AUC and 94.1% warning recall."},
        {"Patient": "chb03", "Warning Clips": "444 / 2,924", "ROC-AUC": 0.519, "CNN (0.1) Recall": "68.0%", "CNN (0.1) Acc": "41.2%", "CNN (0.5) Recall": "53.4%", "CNN (0.5) Acc": "53.5%", "Takeaway": "Balanced detection across both 0.1 and 0.5 alert dials."},
        {"Patient": "chb04", "Warning Clips": "204 / 3,488", "ROC-AUC": 0.455, "CNN (0.1) Recall": "24.0%", "CNN (0.1) Acc": "78.2%", "CNN (0.5) Recall": "8.8%", "CNN (0.5) Acc": "86.8%", "Takeaway": "Hard outlier patient whose pre-seizure waves differed from peers."},
        {"Patient": "chb05", "Warning Clips": "367 / 3,087", "ROC-AUC": 0.565, "CNN (0.1) Recall": "6.8%", "CNN (0.1) Acc": "85.6%", "CNN (0.5) Recall": "1.4%", "CNN (0.5) Acc": "88.0%", "Takeaway": "Good ranking (0.565 AUC), but conservative risk scores sit below 0.1."}
    ])

    selected_pat = st.selectbox(
        "Choose an unseen test patient:",
        ["All 5 Patients (Side-by-Side)"] + lopo_patient_data["Patient"].tolist()
    )
    if selected_pat == "All 5 Patients (Side-by-Side)":
        st.dataframe(lopo_patient_data, use_container_width=True, hide_index=True)
    else:
        row = lopo_patient_data[lopo_patient_data["Patient"] == selected_pat].iloc[0]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Unseen ROC-AUC", f"{row['ROC-AUC']:.3f}")
        p2.metric("Recall @ 0.1 Dial", row["CNN (0.1) Recall"])
        p3.metric("Accuracy @ 0.5 Dial", row["CNN (0.5) Acc"])
        p4.metric("Warning / Total Clips", row["Warning Clips"])
        st.caption(f"**Clinical Interpretation for `{selected_pat}`:** {row['Takeaway']}")

    with st.expander("📋 View Exact Mean ± Std Comparison Table Across All 4 Models (Table 8)", expanded=False):
        st.dataframe(load_table("table_08_final_chb_comparison.csv"), use_container_width=True)

    st.markdown("---")

    # =========================================================
    # STEP C: PATIENT-SPECIFIC VS INDEPENDENT (FIGURE 15)
    # =========================================================
    st.subheader("Step C: Why Academic Papers Report '99% Accuracy' — The Reality Check (`Figure 15`)")
    st.caption("💡 **TL;DR Takeaway:** Testing on the same person yields `0.990` AUC because the model memorizes personal brainwaves; unseen-patient LOPO (`0.568` AUC) reveals the true clinical challenge.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_15_patient_specific_vs_independent.png", "Figure 15: Unseen-patient LOPO generalization (left) vs. same-patient upper bound (right)")
    with col2:
        st.warning("""
        **Questions & Answers to Read This Chart:**
- **What is being compared on the horizontal axis (`Setting`)?**
  - **Left (`Patient-independent CNN (0.1)`):** Tested on a **brand-new patient** excluded from training (our realistic LOPO benchmark).
  - **Right (`Patient-specific LR`):** Trained on early recordings from a patient and tested on later recordings from that **exact same patient**.
- **Why do all 4 bars shoot up near `1.0` on the right (`0.944` Accuracy, `0.842` F1, `0.948` Recall, `0.990` AUC)?** Every human brain has a unique electrical signature. When tested on the same person, even a simple linear model memorizes that patient's personal brainwaves.
- **Why does this matter?** A new patient arriving at a hospital does not have hours of pre-recorded seizures to train a personal model. The left bars show the **honest clinical challenge**—and prove why cross-patient generalization is the true benchmark.
        """)

    st.markdown("---")

    # =========================================================
    # STEP D: SOURCE CNN TRAINING STABILITY (FIGURE 16)
    # =========================================================
    st.subheader("Step D: Training the Final Boston Source Model (`Figure 16`)")
    st.caption("💡 **TL;DR Takeaway:** Training loss drops smoothly from `0.805` to `0.482` across 5 epochs, confirming stable convergence before cross-hospital testing.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_16_chb_source_training_loss.png", "Figure 16: Training loss progression across 5 epochs on the combined CHB-MIT cohort")
    with col2:
        st.info("""
        **Questions & Answers to Read This Chart:**
- **What does `Epoch` on the horizontal axis (`1.0` to `5.0`) mean?** One **Epoch** is one complete pass where the 1D-CNN reads through every training window across all 5 Boston patients. The 5 blue dots mark the error score after each full pass (`1, 2, 3, 4, 5`).
- **What does `Training loss` on the vertical axis (`0.50` to `0.80`) mean?** It measures how far off the model's predictions are from the true labels (`BCEWithLogitsLoss`)—**lower means fewer mistakes**.
- **What does the smooth downward curve prove?** Loss drops sharply from **`~0.805` (Epoch 1)** to **`~0.613` (Epoch 2)** and continues declining smoothly down to **`~0.482` (Epoch 5)**. This confirms stable learning without erratic jumps before saving the model for Act II.
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

    render_chapter_nav(3)

# =========================================================
# PAGE 4: CROSS-DATASET TRANSFER — SIENA (ACT II)
# =========================================================
elif page == PAGES[4]:
    st.header("4. Act II (RQ2): What Happens at a Completely Different Hospital? (Siena, Italy)")
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
    st.caption("💡 **TL;DR Takeaway:** Shipping the Boston 1D-CNN directly to Italy without local calibration drops ROC-AUC from `0.568` to `0.395`—proving severe cross-hospital domain shift.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_12_chb_to_siena_drop.png", "Figure 12: ROC-AUC comparison when transferring models directly from CHB-MIT (blue) to Siena (orange)")
    with col2:
        st.warning("""
        **Questions & Answers to Read This Chart:**
- **What do the two bars compare?** For each model on the horizontal axis (`LR`, `RF`, `CNN`), the **blue bar (`CHB AUC`)** shows its average score in Boston, and the **orange bar (`Siena AUC`)** shows its zero-shot score in Italy.
- **Why did the `CNN` plummet from `0.568` (blue) to `0.395` (orange)?** Because the 1D-CNN reads raw waveform shapes directly. Subtle hardware and adult-vs-pediatric waveform shifts in Siena's recordings threw off its Boston-learned filters (`-0.173` AUC drop).
- **Why did `LR` (`0.416` → `0.476`) and `RF` (`0.438` → `0.546`) look flat or slightly higher?** Hand-crafted summary statistics are less sensitive to raw waveform shifts, but as Table 9 shows, they still fail clinically because they trigger constant false alarms on Siena patients.
        """)

    with st.expander("📋 View Full Zero-Shot CHB-MIT → Siena Transfer Metrics (Table 9)", expanded=False):
        st.dataframe(load_table("table_09_chb_to_siena_transfer_results.csv"), use_container_width=True)

    st.markdown("---")

    # =========================================================
    # STEP B: TARGET-DOMAIN FINE-TUNING PROCESS (FIGURE 16b)
    # =========================================================
    st.subheader("Step B: Fixing Domain Shift — Light Local Fine-Tuning on 4 Siena Patients (`Figure 16b`)")
    st.caption("💡 **TL;DR Takeaway:** Fine-tuning for just 5 epochs on 4 local Siena patients (`1,059` windows) rapidly drives loss down from `1.669` to `0.411` while keeping 4 unseen patients locked away for testing.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_16b_siena_finetune_training_loss.png", "Figure 16b: Training-loss progression over 5 fine-tuning epochs on the 4 Siena calibration patients")
    with col2:
        st.info("""
        **Questions & Answers to Read This Chart:**
- **Why does Epoch 1 start at a very high loss (`1.669`)?** In Boston (`Figure 16`), training started at `0.805`. Starting at **`1.669`** here is visual proof of **domain shift**: the Boston-trained neural network was initially surprised by Siena's signal amplitude and adult EEG patterns.
- **What happens between Epoch 1 and Epoch 5?** Each dot marks one full pass (`Epoch 1` to `5`) over the 4 calibration patients (`PN00, PN01, PN03, PN12`). The error drops steeply to **`0.950` (Epoch 2)** and **`0.623` (Epoch 3)**, then settles smoothly down to **`0.411` (Epoch 5)**.
- **Why stop at 5 epochs?** Stopping early lets the network adapt its filters to the Italian hospital's equipment without memorizing or overfitting the 4 calibration patients—leaving **`PN05, PN09, PN10, PN14` (`2,063` windows)** completely unseen for evaluation.
        """)

    st.markdown("---")

    # =========================================================
    # STEP C: ZERO-SHOT VS FINE-TUNED RECOVERY (FIGURE 13 & TABLE 10)
    # =========================================================
    st.subheader("Step C: Did Fine-Tuning Recover Performance on Unseen Siena Patients? (`Figure 13`)")
    st.caption("💡 **TL;DR Takeaway:** Light calibration boosts ROC-AUC by `+0.194` (to `0.589`), and pairing it with the `0.1` alert dial reaches `76.2%` warning recall.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_13_finetuning_improvement.png", "Figure 13: Zero-shot vs. fine-tuned 1D-CNN performance on Siena at thresholds 0.5 and 0.1")
    with col2:
        st.success("""
        **Questions & Answers to Read This Chart:**
- **What do the 4 model groups on the horizontal axis show?** The left two bars show the **zero-shot CNN** (before seeing any Siena data); the right two bars show the **fine-tuned CNN** (after 5 epochs of calibration on 4 patients).
- **Look at the red `auc` bars (`0.395` → `0.589`):** Fine-tuning boosts ROC-AUC by **`+0.194`**, completely erasing the cross-hospital drop and outperforming our original Boston LOPO benchmark (`0.568`).
- **Why is `CNN CHB→Siena fine-tuned (0.1)` the clear winner?** Combining local fine-tuning with the sensitive `0.1` alert threshold drives massive gains across every metric:
  - **Recall (green):** Jumps from `53.3%` to **`76.2%`** (`+22.9%` more pre-seizure warnings caught).
  - **Accuracy (blue):** Jumps from `48.8%` to **`67.9%`** (`+19.1%`).
  - **F1-Score (orange):** Jumps from `0.611` to **`0.797`** *(note: Siena's holdout windows contain a higher proportion of pre-ictal segments than Boston, which naturally lifts F1)*.
        """)

    # Interactive Threshold Dial Simulator (Real Notebook Sweep Data)
    st.markdown("#### 🎛️ Interactive Simulator: How the Alert Threshold Dial (`0.1` → `0.9`) Controls Warnings")
    st.write("Move the slider below to see how changing the decision cutoff on the **Fine-Tuned Siena 1D-CNN** trades off seizure sensitivity against strictness (using exact values from your evaluation sweep):")

    siena_ft_sweep = {
        0.1: {"acc": "67.9%", "f1": "0.797", "recall": "76.2%", "auc": "0.589", "note": "Recommended clinical setting: catches 3 out of 4 pre-seizure warning clips."},
        0.2: {"acc": "58.2%", "f1": "0.704", "recall": "60.4%", "auc": "0.589", "note": "Moderate sensitivity: catches 60.4% of pre-seizure warning clips."},
        0.3: {"acc": "51.7%", "f1": "0.624", "recall": "48.6%", "auc": "0.589", "note": "Balanced cutoff: misses roughly half of early warning clips."},
        0.5: {"acc": "39.3%", "f1": "0.439", "recall": "28.8%", "auc": "0.589", "note": "Default ML cutoff (50%): too strict for subtle pre-seizure signals."},
        0.7: {"acc": "25.1%", "f1": "0.169", "recall": "9.2%",  "auc": "0.589", "note": "High cutoff: misses over 90% of pre-seizure warnings."},
        0.9: {"acc": "18.2%", "f1": "0.015", "recall": "0.8%",  "auc": "0.589", "note": "Extreme cutoff: almost never triggers an alarm."}
    }
    chosen_thresh = st.select_slider(
        "Select Alert Trigger Threshold (Risk Probability Required to Sound Alarm):",
        options=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        value=0.1
    )
    t_stats = siena_ft_sweep[chosen_thresh]
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Warning Recall", t_stats["recall"])
    tc2.metric("Window Accuracy", t_stats["acc"])
    tc3.metric("F1-Score", t_stats["f1"])
    tc4.metric("ROC-AUC (Unchanged)", t_stats["auc"])
    st.caption(f"**Dial `{chosen_thresh}` Behavior:** {t_stats['note']} Notice that **ROC-AUC stays locked at `0.589`** because it measures overall ranking across all thresholds.")

    with st.expander("📋 View Full Zero-Shot vs. Fine-Tuned Comparison Table (Table 10)", expanded=False):
        st.dataframe(load_table("table_10_fine_tuning_results.csv"), use_container_width=True)

    render_chapter_nav(4)

# =========================================================
# PAGE 5: CLINICAL EVENT-LEVEL BURDEN (ACT III)
# =========================================================
elif page == PAGES[5]:
    st.header("5. Act III (RQ3): Is This Ready for a Hospital Bedside? (Clinical Burden Analysis)")
    st.write("""
    High window-level recall does not automatically make a medical device usable in an intensive care unit. 
    In Act III, we translated our Siena predictions into **event-level clinical metrics** to measure the real trade-off between 
    **catching every upcoming seizure** and **causing nurse alarm fatigue**.
    """)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Checks / Hour", "240 windows", "30s window, 15s step")
    m2.metric("Baseline False Burden", "~234–239 / hr", "Alarms on ~98% of windows")
    m3.metric("Fine-Tuned CNN Burden", "~170.6 / hr", "-29% fewer false alarms vs LR")
    m4.metric("Event Sensitivity", "100% (1.0)", "Warns ahead of every seizure")

    st.markdown("---")

    st.subheader("Step A: Advance Seizure Warnings vs. False Alarms Per Hour (`Figure 14`)")
    st.caption("💡 **TL;DR Takeaway:** Traditional baselines achieve 100% event sensitivity only by sounding false alarms on ~98% of windows (`~234–239/hr`), whereas the 1D-CNN cuts false alerts by ~29–41%.")

    col1, col2 = st.columns([3, 2])
    with col1:
        show_image("figure_14_event_level_comparison.png", "Figure 14: Event-level sensitivity (blue) vs. approximate false-positive alarms per hour (orange) on Siena")
    with col2:
        st.warning("""
        **Questions & Answers to Read This Chart:**
- **Where are the blue `event_sensitivity` bars?** Look right along the bottom baseline (`0` to `1`): the thin blue slivers sit at **`~1.0` (100%)** for all four models (`0.958` for zero-shot CNN, `1.000` for fine-tuned CNN). They look tiny only because they are plotted on the same `0–250` vertical axis as the hourly false-alarm counts.
- **Why do `LR` (`~239.4/hr`) and `RF (0.1)` (`~234.3/hr`) hit nearly `240`?** With 30-second windows overlapping by 50% (one step every 15 seconds), there are **240 windows in an hour**. Both traditional baselines simply trigger an alarm on almost *every single window*—proving their high sensitivity is an illusion caused by constant false alarms.
- **How much better is the 1D-CNN?** The 1D-CNN slashes false-alarm burden by **~29% to 41%**, dropping to **`~141.9/hr`** (zero-shot) and **`~170.6/hr`** (fine-tuned) while warning before seizures occur.
        """)

    st.caption("ℹ️ **Methodological Note:** Because normal baseline clips were subsampled (`max 80 windows per file`), hourly false-positive burden is estimated over the retained evaluation windows (where 240 windows = 1 hour of 15-second steps).")

    with st.expander("📋 View Exact Event-Level Sensitivity & False-Alarm Metrics (Table 11)", expanded=True):
        st.dataframe(load_table("table_11_event_level_results.csv"), use_container_width=True)

    st.markdown("---")

    st.subheader("Step B: Key Takeaway & Next Steps for Real-World Deployment")
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        **Why `~170.6` Raw Window Alerts/Hour Is Expected (and Honest):**
- Each 30-second window is scored independently without temporal smoothing.
- Lowering the decision dial to `0.1` prioritizes catching subtle pre-seizure changes (`76.2%` window recall), which naturally allows isolated noisy windows to trigger single-step alerts.
        """)
    with c2:
        st.success("""
        **How to Bring False Alarms Below `< 1 / Hour` in Production:**
- **Persistence Filtering ($k$-of-$n$ rule):** Only sound a bedside alarm when at least *4 out of 6* consecutive windows exceed the risk threshold.
- **Refractory Cooldown Period:** Suppress duplicate alerts for 15–30 minutes after an initial warning is raised.
        """)

    render_chapter_nav(5)

# =========================================================
# PAGE 6: APPENDIX & METADATA
# =========================================================
elif page == PAGES[6]:
    st.header("6. Appendix & Metadata: Dissertation Structure, Code Architecture & Glossary")
    st.write("This reference hub connects every dashboard chapter to the Master's dissertation research questions (`RQ1`–`RQ3`), the modular repository structure, and the complete abbreviation decoder.")

    st.subheader("1. Chapter-to-Research-Question Mapping")
    st.dataframe(load_table("table_12_chapter_to_research_question_mapping.csv"), use_container_width=True)

    st.markdown("---")
    st.subheader("2. Reproducible Codebase Architecture")
    st.write("Engineered for memory-safe execution across 31,147 multi-channel EEG windows without Out-Of-Memory (OOM) bottlenecks:")
    st.code("""eeg-seizure-prediction/
├── data/               # Raw .edf hospital recordings & cached .npz window chunks
├── notebooks/          # Exploratory data analysis, LOPO experiments & figure generation
├── src/                # Modular Python package
│   ├── config.py       # Centralized hyperparameters, 16-channel montage & paths
│   ├── data_utils.py   # Band-pass filtering (0.5–40 Hz), resampling (256 Hz) & windowing
│   ├── dataset.py      # PyTorch Dataset with memory-safe .npz chunk indexing
│   ├── models/cnn_1d.py# 1D Convolutional Neural Network architecture
│   └── evaluation.py   # LOPO cross-validation, threshold sweeps & event-level metrics
└── scripts/            # End-to-end CLI pipelines (01_preprocess.py, 02_train_cnn.py)""", language="text")

    st.markdown("---")
    st.subheader("3. Complete Abbreviation & Plain-English Glossary")
    st.write("A full reference guide showing what every abbreviation stands for and what it means in plain English.")

    glossary_df = pd.DataFrame(
        GLOSSARY_TERMS,
        columns=["Abbreviation / Term", "Stands For (Full Form)", "Category", "Plain-English Meaning"]
    )
    cat_filter = st.selectbox("Filter by Category:", ["All Categories"] + sorted(glossary_df["Category"].unique().tolist()))
    if cat_filter != "All Categories":
        glossary_df = glossary_df[glossary_df["Category"] == cat_filter]

    st.dataframe(glossary_df, use_container_width=True, hide_index=True)

    render_chapter_nav(6)