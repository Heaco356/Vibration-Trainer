from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import io
import random

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home(
    speed_rpm: float = 1800,
    unbalance: float = 0.5,
    misalignment: float = 0.0,
    looseness: float = 0.0,
    bearing_defect: float = 0.0,
    fault_type: str = "none",
    balls: int = 9,
    ball_diameter: float = 7.94,
    pitch_diameter: float = 39.04,
    contact_angle: float = 0.0,
    bearing_location: str = "drive_end",
    gear_enabled: str = "no",
    input_teeth: int = 20,
    output_teeth: int = 40,
    gear_defect: float = 0.0,
    quiz_mode: str = "off",
    quiz_answer: str = "",
    score_action: str = "none",
):
    rpm = max(speed_rpm, 1)
    rotation_hz = rpm / 60
    shaft_hz = rotation_hz
    contact_angle_rad = np.deg2rad(contact_angle)
    ratio = (ball_diameter / pitch_diameter) * np.cos(contact_angle_rad)

    bpfo = (balls / 2) * shaft_hz * (1 - ratio)
    bpfi = (balls / 2) * shaft_hz * (1 + ratio)
    bsf = (pitch_diameter / (2 * ball_diameter)) * shaft_hz * (1 - ratio**2)
    ftf = 0.5 * shaft_hz * (1 - ratio)

    input_gear_mesh_frequency = shaft_hz * input_teeth
    output_shaft_hz = shaft_hz * input_teeth / output_teeth
    output_gear_mesh_frequency = output_shaft_hz * output_teeth

    fault_frequencies = {
        "none": 0.0,
        "outer_race": bpfo,
        "inner_race": bpfi,
        "ball": bsf,
        "cage": ftf,
    }

    fault_names = {
        "none": "No selected bearing fault",
        "outer_race": "Outer-race defect (BPFO)",
        "inner_race": "Inner-race defect (BPFI)",
        "ball": "Ball/rolling-element defect (BSF)",
        "cage": "Cage defect (FTF)",
    }

    selected_fault_frequency = fault_frequencies.get(fault_type, 0.0)
    selected_fault_name = fault_names.get(fault_type, "No selected bearing fault")

    duration = 2.0
    samples = 4000
    t = np.linspace(0, duration, samples, endpoint=False)
    fs = samples / duration

    one_x = unbalance * np.sin(2 * np.pi * rotation_hz * t)
    two_x = misalignment * 0.5 * np.sin(2 * np.pi * 2 * rotation_hz * t)
    harmonics = looseness * 0.25 * (
        np.sin(2 * np.pi * 2 * rotation_hz * t)
        + 0.6 * np.sin(2 * np.pi * 3 * rotation_hz * t)
        + 0.4 * np.sin(2 * np.pi * 4 * rotation_hz * t)
    )

    location_factors = {
        "drive_end": 1.0,
        "non_drive_end": 0.75,
        "motor": 0.5,
    }

    location_factor = location_factors.get(bearing_location, 1.0)

    bearing = (
        bearing_defect
        * 0.3
        * location_factor
        * np.random.normal(0, 1, samples)
    )

    fault_wave = np.zeros(samples)

    if selected_fault_frequency > 0 and bearing_defect > 0:
        fault_omega = 2 * np.pi * selected_fault_frequency
        fault_wave = (
            bearing_defect
            * 0.4
            * location_factor
            * np.sin(fault_omega * t)
        )

        for sideband in [-1, 1]:
            sideband_frequency = selected_fault_frequency + sideband * shaft_hz
            if sideband_frequency > 0:
                fault_wave += (
                    bearing_defect
                    * 0.15
                    * location_factor
                    * np.sin(2 * np.pi * sideband_frequency * t)
                )

    gear_wave = np.zeros(samples)

    if gear_enabled == "yes" and gear_defect > 0:
        gear_wave = (
            gear_defect
            * 0.25
            * np.sin(2 * np.pi * input_gear_mesh_frequency * t)
        )

        gear_wave += (
            gear_defect
            * 0.15
            * np.sin(2 * np.pi * output_gear_mesh_frequency * t)
        )

        for sideband in [-1, 1]:
            sideband_frequency = input_gear_mesh_frequency + sideband * shaft_hz
            if sideband_frequency > 0:
                gear_wave += (
                    gear_defect
                    * 0.10
                    * np.sin(2 * np.pi * sideband_frequency * t)
                )

    waveform = one_x + two_x + harmonics + bearing + fault_wave + gear_wave
    healthy_waveform = one_x + two_x + harmonics

    velocity_rms = np.sqrt(np.mean(waveform**2)) * 10
    healthy_velocity_rms = np.sqrt(np.mean(healthy_waveform**2)) * 10

    spectrum = np.abs(np.fft.rfft(waveform))**2
    healthy_spectrum = np.abs(np.fft.rfft(healthy_waveform))**2
    frequencies = np.fft.rfftfreq(samples, d=1 / fs)

    if velocity_rms < 1.8:
        zone, color, advice = "A", "#2e7d32", "Excellent condition."
    elif velocity_rms < 4.5:
        zone, color, advice = "B", "#f9a825", "Acceptable for long-term operation."
    elif velocity_rms < 11.2:
        zone, color, advice = "C", "#ef6c00", "Unsatisfactory for continuous operation; plan maintenance."
    else:
        zone, color, advice = "D", "#c62828", "Severe vibration; damage risk. Stop and investigate."

    quiz_questions = {
        "unbalance": {
            "question": "A strong 1X frequency peak is most likely caused by which fault?",
            "options": ["Unbalance", "Misalignment", "Bearing defect", "Gear defect"],
            "correct": "Unbalance",
        },
        "misalignment": {
            "question": "An elevated 2X frequency peak is most likely caused by which fault?",
            "options": ["Unbalance", "Misalignment", "Looseness", "Cage defect"],
            "correct": "Misalignment",
        },
        "looseness": {
            "question": "Multiple harmonics of the running speed are most likely caused by which fault?",
            "options": ["Unbalance", "Misalignment", "Mechanical looseness", "Gear defect"],
            "correct": "Mechanical looseness",
        },
        "bearing": {
            "question": "A peak at BPFO, BPFI, BSF, or FTF is most likely caused by which fault?",
            "options": ["Unbalance", "Bearing defect", "Misalignment", "Looseness"],
            "correct": "Bearing defect",
        },
        "gear": {
            "question": "A strong gear-mesh frequency with sidebands is most likely caused by which fault?",
            "options": ["Unbalance", "Bearing defect", "Gear defect", "Looseness"],
            "correct": "Gear defect",
        },
    }

    possible_questions = []

    if unbalance > 0.2:
        possible_questions.append("unbalance")
    if misalignment > 0.2:
        possible_questions.append("misalignment")
    if looseness > 0.2:
        possible_questions.append("looseness")
    if bearing_defect > 0.2:
        possible_questions.append("bearing")
    if gear_enabled == "yes" and gear_defect > 0.2:
        possible_questions.append("gear")

    if not possible_questions:
        possible_questions = [
            "unbalance",
            "misalignment",
            "looseness",
            "bearing",
            "gear",
        ]

    selected_quiz_key = random.choice(possible_questions)
    selected_quiz = quiz_questions[selected_quiz_key]

    diagnoses = []

    if unbalance > 0.2:
        diagnoses.append("Strong 1X peak: possible unbalance.")
    if misalignment > 0.2:
        diagnoses.append("Elevated 2X peak: possible misalignment.")
    if looseness > 0.2:
        diagnoses.append("Multiple harmonics: possible mechanical looseness.")
    if bearing_defect > 0.2:
        if selected_fault_frequency > 0:
            diagnoses.append(
                f"{selected_fault_name}: strong peak near "
                f"{selected_fault_frequency:.2f} Hz, with sidebands at "
                f"±{shaft_hz:.2f} Hz."
            )
        else:
            diagnoses.append(
                "Broadband random energy: possible bearing defect. "
                "Check peaks near BPFO, BPFI, BSF, and FTF."
            )
    if gear_enabled == "yes" and gear_defect > 0.2:
        diagnoses.append(
            f"Gear defect detected: strong gear-mesh energy near "
            f"{input_gear_mesh_frequency:.2f} Hz and "
            f"{output_gear_mesh_frequency:.2f} Hz."
        )

    if not diagnoses:
        diagnoses.append("No dominant fault pattern detected.")

    quiz_result = ""
    quiz_is_correct = False

    if quiz_mode == "on" and quiz_answer:
        if quiz_answer == selected_quiz["correct"]:
            quiz_result = "Correct! You identified the likely fault."
            quiz_is_correct = True
        else:
            quiz_result = (
                f"Incorrect. The best answer was: "
                f"{selected_quiz['correct']}."
            )

    if zone == "A":
        maintenance = "No action required. Continue normal monitoring."
    elif zone == "B":
        maintenance = "Continue operation. Recheck vibration at the next planned inspection."
    elif zone == "C":
        maintenance = "Schedule maintenance. Investigate the dominant frequency and monitor closely."
    else:
        maintenance = "Reduce load or stop the machine if safe. Investigate immediately."

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Machine Animation",
            "Frequency Spectrum: Healthy vs Faulty",
            "Fault Sidebands",
            "Maintenance Advice",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "indicator"}],
        ],
    )

    shaft_angle = 0
    shaft_x = [0, np.cos(shaft_angle)]
    shaft_y = [0, np.sin(shaft_angle)]

    fig.add_trace(
        go.Scatter(
            x=[-1, 1],
            y=[0, 0],
            mode="lines",
            line=dict(color="#90a4ae", width=8),
            name="Machine body",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=shaft_x,
            y=shaft_y,
            mode="lines+markers",
            line=dict(color="#1565c0", width=5),
            marker=dict(size=18, color="#c62828"),
            name="Shaft",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[-0.7, 0.7],
            y=[0, 0],
            mode="markers",
            marker=dict(size=30, color="#455a64", symbol="square"),
            name="Bearings",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=healthy_spectrum,
            mode="lines",
            line=dict(color="#2e7d32", width=1.5),
            name="Healthy",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=spectrum,
            mode="lines",
            line=dict(color="#c62828", width=1.5),
            name="Faulty",
        ),
        row=1,
        col=2,
    )

    fault_lines = [
        (rotation_hz, "1X"),
        (2 * rotation_hz, "2X"),
        (3 * rotation_hz, "3X"),
        (bpfo, "BPFO"),
        (bpfi, "BPFI"),
        (bsf, "BSF"),
        (ftf, "FTF"),
    ]

    for frequency, label in fault_lines:
        if frequency > 0:
            fig.add_vline(
                x=frequency,
                line_dash="dash",
                line_color="#c62828",
                annotation_text=label,
                annotation_position="top",
                row=1,
                col=2,
            )

    if selected_fault_frequency > 0:
        for sideband in [-2, -1, 1, 2]:
            sideband_frequency = selected_fault_frequency + sideband * shaft_hz
            if sideband_frequency > 0:
                fig.add_vline(
                    x=sideband_frequency,
                    line_dash="dot",
                    line_color="#6a1b9a",
                    annotation_text=f"SB{sideband:+d}",
                    annotation_position="top",
                    row=2,
                    col=1,
                )

        fig.add_vline(
            x=selected_fault_frequency,
            line_dash="dash",
            line_color="#c62828",
            annotation_text="Fault",
            annotation_position="top",
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=waveform,
            mode="lines",
            line=dict(color="#1565c0", width=1.2),
            name="Faulty waveform",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=velocity_rms,
            title={"text": "RMS velocity (mm/s)"},
            number={"font": {"size": 44, "color": color}},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Advanced Rotating Machinery Vibration Trainer",
        template="plotly_white",
        height=850,
        showlegend=False,
    )

    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

    controls = f"""
    <form method="get">
      <label>Speed (RPM): <input type="number" step="10" name="speed_rpm" value="{rpm}"></label><br>
      <label>Unbalance: <input type="number" step="0.1" min="0" max="3" name="unbalance" value="{unbalance}"></label><br>
      <label>Misalignment: <input type="number" step="0.1" min="0" max="3" name="misalignment" value="{misalignment}"></label><br>
      <label>Looseness: <input type="number" step="0.1" min="0" max="3" name="looseness" value="{looseness}"></label><br>
      <label>Bearing defect: <input type="number" step="0.1" min="0" max="3" name="bearing_defect" value="{bearing_defect}"></label><br>
      <label>Bearing location:
        <select name="bearing_location">
          <option value="drive_end" {'selected' if bearing_location == 'drive_end' else ''}>Drive End (DE)</option>
          <option value="non_drive_end" {'selected' if bearing_location == 'non_drive_end' else ''}>Non-Drive End (NDE)</option>
          <option value="motor" {'selected' if bearing_location == 'motor' else ''}>Motor Bearing</option>
        </select>
      </label><br>
      <label>Bearing fault type:
        <select name="fault_type">
          <option value="none" {'selected' if fault_type == 'none' else ''}>No fault</option>
          <option value="outer_race" {'selected' if fault_type == 'outer_race' else ''}>Outer race (BPFO)</option>
          <option value="inner_race" {'selected' if fault_type == 'inner_race' else ''}>Inner race (BPFI)</option>
          <option value="ball" {'selected' if fault_type == 'ball' else ''}>Ball (BSF)</option>
          <option value="cage" {'selected' if fault_type == 'cage' else ''}>Cage (FTF)</option>
        </select>
      </label><br>
      <label>Include gearbox:
        <select name="gear_enabled">
          <option value="no" {'selected' if gear_enabled == 'no' else ''}>No</option>
          <option value="yes" {'selected' if gear_enabled == 'yes' else ''}>Yes</option>
        </select>
      </label><br>
      <label>Input gear teeth: <input type="number" step="1" min="1" name="input_teeth" value="{input_teeth}"></label><br>
      <label>Output gear teeth: <input type="number" step="1" min="1" name="output_teeth" value="{output_teeth}"></label><br>
      <label>Gear defect: <input type="number" step="0.1" min="0" max="3" name="gear_defect" value="{gear_defect}"></label><br>
      <label>Balls: <input type="number" step="1" min="1" name="balls" value="{balls}"></label><br>
      <label>Ball diameter (mm): <input type="number" step="0.01" name="ball_diameter" value="{ball_diameter}"></label><br>
      <label>Pitch diameter (mm): <input type="number" step="0.01" name="pitch_diameter" value="{pitch_diameter}"></label><br>
      <label>Contact angle (degrees): <input type="number" step="1" name="contact_angle" value="{contact_angle}"></label><br>
      <label>Training quiz:
        <select name="quiz_mode">
          <option value="off" {'selected' if quiz_mode == 'off' else ''}>Off</option>
          <option value="on" {'selected' if quiz_mode == 'on' else ''}>On</option>
        </select>
      </label><br>
      <label>Your answer:
        <select name="quiz_answer">
          <option value="">Choose an answer</option>
          <option value="Unbalance" {'selected' if quiz_answer == 'Unbalance' else ''}>Unbalance</option>
          <option value="Misalignment" {'selected' if quiz_answer == 'Misalignment' else ''}>Misalignment</option>
          <option value="Mechanical looseness" {'selected' if quiz_answer == 'Mechanical looseness' else ''}>Mechanical looseness</option>
          <option value="Bearing defect" {'selected' if quiz_answer == 'Bearing defect' else ''}>Bearing defect</option>
          <option value="Gear defect" {'selected' if quiz_answer == 'Gear defect' else ''}>Gear defect</option>
        </select>
      </label><br>
      <button type="submit">Run Machine</button>
      <button type="button" onclick="window.print()">Print / Save Report as PDF</button>
      <a href="/download-csv?speed_rpm={rpm}&unbalance={unbalance}&misalignment={misalignment}&looseness={looseness}&bearing_defect={bearing_defect}&fault_type={fault_type}&bearing_location={bearing_location}&gear_enabled={gear_enabled}&input_teeth={input_teeth}&output_teeth={output_teeth}&gear_defect={gear_defect}&balls={balls}&ball_diameter={ball_diameter}&pitch_diameter={pitch_diameter}&contact_angle={contact_angle}">
        <button type="button">Download CSV</button>
      </a>
    </form>
    """

    diagnosis_items = "".join(f"<li>{item}</li>" for item in diagnoses)

    return f"""
    <html>
      <head>
        <title>Advanced Rotating Machinery Vibration Trainer</title>
        <style>
          body {{ font-family: Arial; max-width: 1250px; margin: 25px auto; color:#263238; }}
          h1 {{ color:#0d47a1; }}
          .panel {{ background:#eef4fb; padding:16px; border-radius:10px; margin-bottom:18px; }}
          .zone {{ font-size:22px; font-weight:bold; color:{color}; }}
          input, select, button {{ padding:7px; margin:4px 0; }}
          button {{ background:#0d47a1; color:white; border:none; border-radius:5px; cursor:pointer; }}
          a button {{ background:#2e7d32; }}
        </style>
      </head>
      <body>
        <h1>Advanced Rotating Machinery Vibration Trainer</h1>
        <div class="panel">
          <b>Rotation speed:</b> {rpm:.0f} RPM ({rotation_hz:.2f} Hz)<br>
          <b>Vibration severity zone:</b> <span class="zone">{zone}</span><br>
          <b>Healthy RMS velocity:</b> {healthy_velocity_rms:.2f} mm/s<br>
          <b>Current RMS velocity:</b> {velocity_rms:.2f} mm/s<br>
          <b>Selected fault:</b> {selected_fault_name}<br>
          <b>Bearing location:</b> {bearing_location.replace('_', ' ').title()}<br>
          <b>Bearing fault frequencies:</b><br>
          BPFO: {bpfo:.2f} Hz &nbsp;|&nbsp;
          BPFI: {bpfi:.2f} Hz &nbsp;|&nbsp;
          BSF: {bsf:.2f} Hz &nbsp;|&nbsp;
          FTF: {ftf:.2f} Hz<br>
          <b>Gearbox:</b><br>
          Input gear-mesh frequency: {input_gear_mesh_frequency:.2f} Hz<br>
          Output shaft speed: {output_shaft_hz:.2f} Hz<br>
          Output gear-mesh frequency: {output_gear_mesh_frequency:.2f} Hz<br>
          <b>Assessment:</b> {advice}<br>
          <b>Maintenance advice:</b> {maintenance}
        </div>
        {controls}
        <div class="panel">
          <b>Training Quiz</b>
          <p>{selected_quiz['question']}</p>
          <p><b>Result:</b> {quiz_result if quiz_result else 'Select an answer and click Run Machine.'}</p>
        </div>
        <div class="panel">
          <b>Training diagnosis</b>
          <ul>{diagnosis_items if quiz_mode == "off" else "<li>Hidden during quiz mode.</li>"}</ul>
        </div>
        {chart}
        <div class="panel">
          <b>Severity Trend by Bearing Location</b>
          <div id="locationTrendChart"></div>
        </div>
        <div class="panel">
          <b>Training Quiz Score</b>
          <div id="scoreBox"></div>
          <form method="get">
            <input type="hidden" name="score_action" value="reset">
            <button type="submit">Reset Score</button>
          </form>
        </div>
        <script>
          const locationKey = "vibrationLocationHistory";
          const scoreKey = "vibrationQuizScore";
          const currentLocation = "{bearing_location}";
          const currentValue = {velocity_rms};
          const currentTime = new Date().toLocaleString();
          const currentCorrect = {str(quiz_is_correct).lower()};

          let locationHistory = JSON.parse(localStorage.getItem(locationKey) || "{{}}");

          if (!locationHistory[currentLocation]) {{
            locationHistory[currentLocation] = [];
          }}

          locationHistory[currentLocation].push({{
            time: currentTime,
            value: currentValue
          }});

          if (locationHistory[currentLocation].length > 20) {{
            locationHistory[currentLocation] = locationHistory[currentLocation].slice(-20);
          }}

          localStorage.setItem(locationKey, JSON.stringify(locationHistory));

          const locationNames = Object.keys(locationHistory);
          const locationTraces = [];

          const locationColors = {{
            drive_end: "#0d47a1",
            non_drive_end: "#ef6c00",
            motor: "#2e7d32"
          }};

          locationNames.forEach(location => {{
            const history = locationHistory[location];
            locationTraces.push({{
              x: history.map(item => item.time),
              y: history.map(item => item.value),
              mode: "lines+markers",
              name: location.replace("_", " ").toUpperCase(),
              line: {{
                color: locationColors[location] || "#6a1b9a",
                width: 3
              }},
              marker: {{ size: 7 }}
            }});
          }});

          Plotly.newPlot("locationTrendChart", locationTraces, {{
            title: "Vibration Severity History by Bearing Location",
            xaxis: {{ title: "Time" }},
            yaxis: {{ title: "RMS velocity (mm/s)" }},
            template: "plotly_white",
            height: 400,
            legend: {{ orientation: "h" }}
          }});

          let score = JSON.parse(localStorage.getItem(scoreKey) || '{{"correct": 0, "total": 0}}');

          if ("{quiz_mode}" === "on" && "{quiz_answer}" !== "") {{
            score.total = score.total + 1;

            if (currentCorrect === true) {{
              score.correct = score.correct + 1;
            }}

            localStorage.setItem(scoreKey, JSON.stringify(score));
          }}

          const percent = score.total > 0 ? Math.round((score.correct / score.total) * 100) : 0;

          document.getElementById("scoreBox").innerHTML =
            "<b>Correct:</b> " + score.correct + "<br>" +
            "<b>Total attempts:</b> " + score.total + "<br>" +
            "<b>Score:</b> " + percent + "%";

          if ("{score_action}" === "reset") {{
            score = {{ correct: 0, total: 0 }};
            localStorage.setItem(scoreKey, JSON.stringify(score));
            document.getElementById("scoreBox").innerHTML =
              "<b>Correct:</b> 0<br><b>Total attempts:</b> 0<br><b>Score:</b> 0%";
          }}
        </script>
      </body>
    </html>
    """


@app.get("/download-csv")
def download_csv(
    speed_rpm: float = 1800,
    unbalance: float = 0.5,
    misalignment: float = 0.0,
    looseness: float = 0.0,
    bearing_defect: float = 0.0,
    fault_type: str = "none",
    balls: int = 9,
    ball_diameter: float = 7.94,
    pitch_diameter: float = 39.04,
    contact_angle: float = 0.0,
    bearing_location: str = "drive_end",
    gear_enabled: str = "no",
    input_teeth: int = 20,
    output_teeth: int = 40,
    gear_defect: float = 0.0,
):
    rpm = max(speed_rpm, 1)
    rotation_hz = rpm / 60
    shaft_hz = rotation_hz
    contact_angle_rad = np.deg2rad(contact_angle)
    ratio = (ball_diameter / pitch_diameter) * np.cos(contact_angle_rad)

    bpfo = (balls / 2) * shaft_hz * (1 - ratio)
    bpfi = (balls / 2) * shaft_hz * (1 + ratio)
    bsf = (pitch_diameter / (2 * ball_diameter)) * shaft_hz * (1 - ratio**2)
    ftf = 0.5 * shaft_hz * (1 - ratio)

    input_gear_mesh_frequency = shaft_hz * input_teeth
    output_shaft_hz = shaft_hz * input_teeth / output_teeth
    output_gear_mesh_frequency = output_shaft_hz * output_teeth

    fault_frequencies = {
        "none": 0.0,
        "outer_race": bpfo,
        "inner_race": bpfi,
        "ball": bsf,
        "cage": ftf,
    }

    selected_fault_frequency = fault_frequencies.get(fault_type, 0.0)

    duration = 2.0
    samples = 4000
    t = np.linspace(0, duration, samples, endpoint=False)
    fs = samples / duration

    one_x = unbalance * np.sin(2 * np.pi * rotation_hz * t)
    two_x = misalignment * 0.5 * np.sin(2 * np.pi * 2 * rotation_hz * t)
    harmonics = looseness * 0.25 * (
        np.sin(2 * np.pi * 2 * rotation_hz * t)
        + 0.6 * np.sin(2 * np.pi * 3 * rotation_hz * t)
        + 0.4 * np.sin(2 * np.pi * 4 * rotation_hz * t)
    )

    location_factors = {
        "drive_end": 1.0,
        "non_drive_end": 0.75,
        "motor": 0.5,
    }

    location_factor = location_factors.get(bearing_location, 1.0)

    bearing = (
        bearing_defect
        * 0.3
        * location_factor
        * np.random.normal(0, 1, samples)
    )

    fault_wave = np.zeros(samples)

    if selected_fault_frequency > 0 and bearing_defect > 0:
        fault_omega = 2 * np.pi * selected_fault_frequency
        fault_wave = (
            bearing_defect
            * 0.4
            * location_factor
            * np.sin(fault_omega * t)
        )

        for sideband in [-1, 1]:
            sideband_frequency = selected_fault_frequency + sideband * shaft_hz
            if sideband_frequency > 0:
                fault_wave += (
                    bearing_defect
                    * 0.15
                    * location_factor
                    * np.sin(2 * np.pi * sideband_frequency * t)
                )

    gear_wave = np.zeros(samples)

    if gear_enabled == "yes" and gear_defect > 0:
        gear_wave = (
            gear_defect
            * 0.25
            * np.sin(2 * np.pi * input_gear_mesh_frequency * t)
        )

        gear_wave += (
            gear_defect
            * 0.15
            * np.sin(2 * np.pi * output_gear_mesh_frequency * t)
        )

        for sideband in [-1, 1]:
            sideband_frequency = input_gear_mesh_frequency + sideband * shaft_hz
            if sideband_frequency > 0:
                gear_wave += (
                    gear_defect
                    * 0.10
                    * np.sin(2 * np.pi * sideband_frequency * t)
                )

    waveform = one_x + two_x + harmonics + bearing + fault_wave + gear_wave
    spectrum = np.abs(np.fft.rfft(waveform))**2
    frequencies = np.fft.rfftfreq(samples, d=1 / fs)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time_s", "waveform", "frequency_hz", "spectrum_power"])

    max_length = max(len(t), len(frequencies))

    for i in range(max_length):
        time_value = t[i] if i < len(t) else ""
        wave_value = waveform[i] if i < len(waveform) else ""
        frequency_value = frequencies[i] if i < len(frequencies) else ""
        spectrum_value = spectrum[i] if i < len(spectrum) else ""
        writer.writerow([time_value, wave_value, frequency_value, spectrum_value])

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vibration_data.csv"},
    )