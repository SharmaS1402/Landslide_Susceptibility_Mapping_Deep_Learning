import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the CSV data
data = pd.read_csv("../data/raw/output.csv")  # Replace with your CSV file path

# Initialize the map centered around the average location
m = folium.Map(location=[data["Y"].mean(), data["X"].mean()], zoom_start=6)

# Create a list of points with weights for the heatmap
# Each point is represented as [latitude, longitude, weight], where 'weight' is the prediction value
heat_data = [[row["Y"], row["X"], row["Predicted_Value"]] for _, row in data.iterrows()]

# Add the heatmap layer
HeatMap(heat_data, min_opacity=0.2, max_opacity=0.8, radius=15, blur=25).add_to(m)

# Save and show the map
m.save("smooth_prediction_map.html")
print("Map saved as smooth_prediction_map.html")
