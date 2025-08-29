#%%
'''
This script processes Antarctic basin mass balance data to estimate precipitation 
using the Mass Budget (MB) approach. The input Excel file contains sub-annual 
basin-wide mass anomalies (in gigatonnes) derived from GRACE/altimetry inversions. 
The script converts fractional-year timestamps to calendar dates, aggregates the 
time series into annual storage change rates (ΔS, in Gt/yr) for each Rignot basin, 
and prepares these results for integration into the MB precipitation framework. 
Subsequent steps will map ΔS to basin areas (Gt/yr → mm/yr) and combine with ice 
discharge and sublimation estimates to derive annual precipitation accumulation 
rates over Antarctica (2002–2022).
'''

#%%

from program_utils import *
import numpy as np
#%%
# Load value and uncertainty sheets
file_path = "/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins/DataCombo_RignotBasins.xlsx"
df_val = pd.read_excel(file_path, sheet_name="Basin_Timeseries (Gt)")
df_err = pd.read_excel(file_path, sheet_name="1-sigma_Error(Gt)")

#%% # Convert decimal year to datetime (accounting for leap years)

# Align both sheets and add Date/Year
df = df_val.copy()
df["Date"] = df["Time"].apply(decimal_year_to_date).dt.strftime('%Y-%m-%d')
df['Date'] = pd.to_datetime(df['Date'])
df["Year"] = pd.to_datetime(df["Date"]).dt.year

err = df_err.copy()
err["Date"] = err["Time"].apply(decimal_year_to_date).dt.strftime('%Y-%m-%d')
err["Year"] = pd.to_datetime(err["Date"]).dt.year

# Sanity check: columns should match
basin_cols = [c for c in df.columns if c not in ["Time", "Date", "Year"]]

# plot the sub annual storage
plt.figure(figsize=(14, 8))

# Plot each basin's data
for basin in basin_cols:
    plt.plot(pd.to_datetime(df["Date"]), df[basin], label=basin, linewidth=2)

# Customize plot
plt.xlabel("Date", fontsize=20)
plt.ylabel("Storage (Gt)", fontsize=20)
plt.title("Sub-Annual Storage Changes by Basin", fontsize=22)
plt.legend(title="Basins", bbox_to_anchor=(1.05, 1), 
           loc="upper left", fontsize=12, title_fontsize=14)

# Format x-axis ticks for better readability
xticks = pd.date_range(start=pd.to_datetime(df["Date"]).min(), end=pd.to_datetime(df["Date"]).max(), periods=10)
plt.xticks(xticks, labels=xticks.strftime('%Y-%m-%d'), rotation=45, fontsize=16)
plt.yticks(np.arange(-2000, 501, 500), fontsize=18)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#%%
# Compute annual slopes per basin and per calendar year
annual_records = []
for year, group in df.groupby("Year"):
    # print(f"Processing year: {year}")
    # Require at least 4 epochs in a year to stabilize the fit
    if len(group) < 4:
        continue
    res = annual_slopes_with_se(group, basin_cols)
    for basin in basin_cols:
        annual_records.append({
            "Year": year,
            "Basin": basin,
            "dS_dt_Gt_per_yr": res[basin]["slope_Gt_per_yr"],
            "dS_dt_SE": res[basin]["slope_stderr"],
            "N_epochs": res[basin]["n"]
        })

annual_ds = pd.DataFrame(annual_records).sort_values(["Basin", "Year"]).reset_index(drop=True)

# Show head and also pivot (years as rows)
head = annual_ds.head(20)

# Plot each basins time series
# Create a side-by-side bar plot for better comparison

plt.figure(figsize=(12, 6))

# Get unique years and basins
years = sorted(annual_ds["Year"].unique())
num_years = len(years)
num_basins = len(basin_cols)

# Define bar width and positions
bar_width = 0.8 / num_basins  # Divide space for bars within each year
x_positions = np.arange(num_years)

# Plot each basin's data
for i, b in enumerate(basin_cols):
    basin_data = annual_ds[annual_ds["Basin"] == b]
    y_values = [basin_data[basin_data["Year"] == year]["dS_dt_Gt_per_yr"].values[0] if year in basin_data["Year"].values else 0 for year in years]
    # y_errors = [basin_data[basin_data["Year"] == year]["dS_dt_SE"].values[0] if year in basin_data["Year"].values else 0 for year in years]
    
    plt.bar(x_positions + i * bar_width, y_values, width=bar_width,  capsize=3, label=b) #, yerr=y_errors

# Customize plot
plt.xticks(x_positions + bar_width * (num_basins - 1) / 2, 
           years, rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Year", fontsize=18)
plt.ylabel("ΔS (Gt/yr)", fontsize=18)
plt.title("Annual Storage Change Rates by Basin", fontsize=18)
plt.legend(title="Basins", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# lets make another plot using line plot

plt.figure(figsize=(12, 6))

# Plot each basin's data
for i, b in enumerate(basin_cols):
    basin_data = annual_ds[annual_ds["Basin"] == b]
    plt.plot(basin_data["Year"], basin_data["dS_dt_Gt_per_yr"], marker='o', label=b)

# Customize plot
plt.xticks(years, rotation=45)
plt.xlabel("Year")
plt.ylabel("dS/dt (Gt/yr)")
plt.title("Annual Storage Change Rates by Basin")
plt.legend(title="Basins", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# plot the total number of epochs as well
plt.figure(figsize=(12, 6))
# Plot each basin's data
for i, b in enumerate(basin_cols):
    epoch_data = annual_ds[annual_ds["Basin"] == b]
    y_values = [epoch_data[epoch_data["Year"] == year]["N_epochs"].values[0] if year in epoch_data["Year"].values else 0 for year in years]
    # y_errors = [epoch_data[epoch_data["Year"] == year]["dS_dt_SE"].values[0] if year in epoch_data["Year"].values else 0 for year in years]

    plt.bar(x_positions + i * bar_width, y_values, width=bar_width,  capsize=3, label=b) #, yerr=y_errors

# Customize plot
plt.xticks(x_positions + bar_width * (num_basins - 1) / 2, 
           years, rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Number of Months", fontsize=18)
plt.title("Total number of Months by Basin", fontsize=18)
plt.legend(title="Basins", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

#%%
# Now we calculate delta S accross all years

# Initialize a list to store results
basin_slopes = []

time_values = df["Time"].values

for b in basin_cols:
    basin_data = df[b].values
    # Calculate the slope for each basin
    slope = np.polyfit(time_values, basin_data, 1)[0]
    basin_slopes.append({"Basin": b, "Slope_Gt_per_yr": slope})

# Convert results to a DataFrame
basin_slopes_df = pd.DataFrame(basin_slopes).sort_values("Basin").reset_index(drop=True)

# Plot bar chart for comparison
plt.figure(figsize=(10, 6))
# Use consistent colors for basins
colors = plt.cm.tab20(np.linspace(0, 1, len(basin_cols)))
plt.bar(basin_slopes_df["Basin"], basin_slopes_df["Slope_Gt_per_yr"], color=colors)

plt.xlabel("Basin", fontsize=18)
plt.ylabel("ΔS (Gt/yr)", fontsize=18)
plt.title("Comparison of ΔS (Gt/yr) per Basin \n Computed across All Years: 2002-2022", fontsize=18)

# Add legend mapping colors to basins
from matplotlib.patches import Patch
legend_patches = [Patch(color=colors[i], label=basin) for i, basin in enumerate(basin_slopes_df["Basin"])]
plt.legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.2), loc="upper center", 
           ncol=5, fontsize=12, title_fontsize=14, frameon=False)

# Adjust tick parameters based on the number of ticks
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)

# plt.tight_layout()
plt.show()