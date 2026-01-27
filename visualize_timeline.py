import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Define the project schedule data based on the plan
tasks = [
    # Phase 1: Research & Design
    ("Literature Review", datetime(2025, 6, 1), datetime(2025, 6, 20), "Phase 1"),
    ("System Architecture", datetime(2025, 6, 15), datetime(2025, 6, 30), "Phase 1"),
    ("Environment Setup", datetime(2025, 6, 25), datetime(2025, 7, 5), "Phase 1"),

    # Phase 2: Core Modules
    ("Data Utilities", datetime(2025, 7, 1), datetime(2025, 7, 20), "Phase 2"),
    ("Generator Model", datetime(2025, 7, 15), datetime(2025, 8, 10), "Phase 2"),
    ("Discrim. & Extractor", datetime(2025, 8, 1), datetime(2025, 8, 25), "Phase 2"),

    # Phase 3: Training
    ("Training Loop Impl.", datetime(2025, 8, 20), datetime(2025, 9, 10), "Phase 3"),
    ("Initial Training", datetime(2025, 9, 5), datetime(2025, 10, 25), "Phase 3"),

    # Phase 4: Security & App
    ("AES & Scrypt Integration", datetime(2025, 10, 25), datetime(2025, 11, 20), "Phase 4"),
    ("Reed-Solomon FEC", datetime(2025, 11, 15), datetime(2025, 12, 10), "Phase 4"),
    ("Gradio UI Dev", datetime(2025, 12, 5), datetime(2026, 1, 15), "Phase 4"),

    # Phase 5: Testing & Final
    ("Integration Testing", datetime(2026, 1, 10), datetime(2026, 2, 5), "Phase 5"),
    ("Metrics Eval (BER/SNR)", datetime(2026, 1, 25), datetime(2026, 2, 15), "Phase 5"),
    ("Final Report & Submission", datetime(2026, 2, 1), datetime(2026, 2, 28), "Phase 5"),
]

# Reverse order so the first task appears at the top of the chart
tasks.reverse()

# Extract data for plotting
names = [t[0] for t in tasks]
start_dates = [t[1] for t in tasks]
end_dates = [t[2] for t in tasks]
phases = [t[3] for t in tasks]

# Calculate durations in days
durations = [(end - start).days for start, end in zip(start_dates, end_dates)]
start_num = mdates.date2num(start_dates)

# Color mapping for phases
colors = []
color_map = {
    "Phase 1": "#3498db", # Blue
    "Phase 2": "#e67e22", # Orange
    "Phase 3": "#2ecc71", # Green
    "Phase 4": "#9b59b6", # Purple
    "Phase 5": "#e74c3c"  # Red
}
for p in phases:
    colors.append(color_map.get(p, "gray"))

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Create horizontal bars
ax.barh(names, durations, left=start_num, height=0.6, color=colors, alpha=0.8, edgecolor='black')

# Format x-axis to show months
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

# Add grid lines
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Add title and labels
plt.title('EchoCrypt Project Timeline (June 2025 - February 2026)', fontsize=16, weight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.tight_layout()

# Add legend manually
handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in sorted(color_map.keys())]
plt.legend(handles, sorted(color_map.keys()), loc='lower right', title="Project Phases")

# Save and show
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_timeline.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Timeline saved to {output_path}")
plt.show()
