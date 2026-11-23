import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from vibration_analysis import VibrationAnalysis, waterfall

st.set_page_config(page_title="Vibration Trainer", layout="wide")
st.title("Online Vibration Trainer")
st.write("Upload vibration data, enter machine details, and analyze bearing and gear condition.")

uploaded_file = st.file_uploader("Upload vibration CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("CSV preview")
    st.write(data.head())
    st.write("Available columns:", list(data.columns))

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns were found in the CSV file.")
        st.stop()

    default_column = "waveform" if "waveform" in data.columns else numeric_columns[-1]
    vibration_column = st.selectbox("Select vibration column", numeric_columns, index=numeric_columns.index(default_column))

    st.subheader("Machine settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        fs = st.number_input("Sampling rate (Hz)", min_value=1.0, value=2000.0, step=100.0)
        rpm = st.number_input("Machine RPM", min_value=1.0, value=1780.0, step=10.0)
    with col2:
        bearing_rollers = st.number_input("Number of rolling elements", min_value=1, value=9)
        roller_diameter = st.number_input("Roller diameter (mm)", min_value=0.1, value=8.0)
    with col3:
        pitch_diameter = st.number_input("Pitch diameter (mm)", min_value=0.1, value=40.0)
        contact_angle = st.number_input("Contact angle (degrees)", min_value=0.0, max_value=90.0, value=0.0)

    st.subheader("Gear settings")
    col4, col5 = st.columns(2)
    with col4:
        gear_teeth = st.number_input("Gear teeth", min_value=1, value=24)
    with col5:
        shaft_ratio = st.number_input("Shaft ratio", min_value=0.001, value=1.0)

    max_band = fs / 2
    band_low = st.number_input("Envelope band lower limit (Hz)", min_value=1.0, max_value=max_band, value=min(200.0, max_band))
    band_high = st.number_input("Envelope band upper limit (Hz)", min_value=2.0, max_value=max_band, value=min(800.0, max_band))

    if st.button("Run Analysis"):
        x = data[vibration_column].values
        x = x[~np.isnan(x)]

        if len(x) < 100:
            st.error("Not enough vibration data was found.")
            st.stop()

        analysis = VibrationAnalysis(x, fs=float(fs), rpm=float(rpm))

        st.subheader("Trend metrics")
        metrics = analysis.time_features()
        metric_cols = st.columns(3)
        for i, (name, value) in enumerate(metrics.items()):
            metric_cols[i % 3].metric(label=name, value=f"{value:.4f}")

        f, a = analysis.spectrum()
        fe, ae = analysis.envelope(band=(float(band_low), float(band_high)))

        st.subheader("FFT spectrum")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(f, a)
        ax1.set_xlim(0, fs / 2)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("Bearing envelope spectrum")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(fe, ae)
        ax2.set_xlim(0, min(200, fs / 2))
        ax2.set_xlabel("Envelope frequency (Hz)")
        ax2.set_ylabel("Envelope amplitude")
        ax2.grid(True)
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Bearing fault analysis")
        bearing_hits = analysis.diagnose_bearing(
            z=int(bearing_rollers),
            d=float(roller_diameter),
            D=float(pitch_diameter),
            band=(float(band_low), float(band_high))
        )
        if bearing_hits:
            st.dataframe(pd.DataFrame(bearing_hits))
        else:
            st.info("No bearing fault-frequency matches were found within the selected tolerance.")

        st.subheader("Gear fault analysis")
        gear_hits = analysis.diagnose_gear(teeth=int(gear_teeth), ratio=float(shaft_ratio))
        if gear_hits:
            st.dataframe(pd.DataFrame(gear_hits))
        else:
            st.info("No gear fault-frequency matches were found within the selected tolerance.")

        st.subheader("Waterfall data")
        spectra = []
        labels = []
        for day in ["Day 1", "Day 2", "Day 3"]:
            an = VibrationAnalysis(x, fs=float(fs), rpm=float(rpm))
            ff, aa = an.spectrum()
            spectra.append(aa)
            labels.append(day)

        wf = waterfall(spectra, f, labels)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for spectrum, label in zip(wf["spectra"], wf["labels"]):
            ax3.plot(wf["frequencies"], spectrum, label=label)
        ax3.set_xlim(0, fs / 2)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Amplitude")
        ax3.set_title("Waterfall comparison")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)
        plt.close(fig3)

        st.success("Analysis complete.")
else:
    st.info("Upload a CSV file to begin.")