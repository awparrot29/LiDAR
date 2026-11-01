import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import calculateangle
from calculateangle import csvout

# Get folder, and the MDS-UPDRS item the CSVs were written under (blank if the
# run did not name one) — csvout.stem needs it to rebuild the filename.
folder = input("Folder name: ")
test_id = input("Test id (e.g. 3.10, blank if none): ").strip() or None

for bp in ('left elbow', 'left shoulder', 'left wrist', 'left knee', 'left hip', 'left ankle', 'right elbow', 'right shoulder', 'right wrist', 'right knee', 'right hip', 'right ankle'):
    path = f"charts/{folder}/data/{csvout.stem(bp, test_id)}.csv"
    if not os.path.exists(path):
        calculateangle.main(folder, bp, test_id=test_id)

    # Read time and z straight from the file. Both are named columns now, so
    # there is no need to count positions or synthesise time from the row index.
    df = pd.read_csv(path, usecols=[csvout.TIME_HEADER, 'z_m'])
    df = df.rename(columns={csvout.TIME_HEADER: 'Time', 'z_m': bp})

    # Get arrays of time and distance data
    x = df['Time'].values
    y = df[bp].values

    # Create trendline
    degree = 2 
    coeffs = np.polyfit(x, y, degree)
    trend = np.polyval(coeffs, x)

    # Get detrended data by finding difference between actual and trend data
    df['Detrended'] = y - trend

    # Create plot with distance, trend, and detrended data over time
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Original')
    plt.plot(x, trend, label='Trend', linestyle='--')
    plt.plot(x, df['Detrended'], label='Detrended', linestyle=':')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(f"{bp.capitalize()} Depth")
    plt.title(f"Polynomial Detrending - {bp.capitalize()} (Degree = {degree})")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(f"charts/{folder}/detrended", exist_ok=True)
    plt.savefig(f"charts/{folder}/detrended/"
                f"{csvout.stem(bp + ' detrended', test_id)}.png")
