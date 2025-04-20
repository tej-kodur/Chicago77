# Combined Streamlit App: Multivariate Statistics + Folium Mapping

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium import Choropleth
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import json
import numpy as np

# ========== Load Data ==========
data = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/indexes_dataset.csv")
feature_metadata = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/metadata_with_index.csv")
index_metadata = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/indexes_metadata.csv")
manual_mapping = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/Completed_FineNeighborhood_to_CommunityAreaName.csv")

with open("E:/PythonCode/BADM550_NEW_PROJECT/data/chicago_neighborhoods.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ========== Metadata Mappings ==========
feature_desc = dict(zip(feature_metadata['column_name'], feature_metadata['description']))
index_desc = dict(zip(index_metadata['index_name'], index_metadata['description']))

INDEX_MAPPING = {
    "wellness_index": [col for col in data.columns if col not in ["CommunityAreaName", "EVI", "SSI", "EOI"]],
    "EVI": ["TRAFVOL", "AIRQUAL", "WATQUAL", "HEATVUL", "NOISEP"],
    "SSI": ["SCHLRAT", "HSDROP", "ASPROG", "SOCASSO", "VCRIME", "PCRIME", "ERTIME", "VSSERV"],
    "EOI": ["HOUSBURD", "MEDINC", "UNEMPLOY", "CHILCBD", "COLLACC", "BROADND"]
}

# ========== Streamlit UI ==========
st.set_page_config(layout="wide")
st.title("📊 Chicago Community Index Dashboard")

with st.sidebar:
    st.header("Configuration")
    selected_index = st.selectbox(
        "Select Index",
        options=["wellness_index", "EVI", "SSI", "EOI"],
        format_func=lambda x: f"{x} - {index_desc.get(x, '')}" if x != "wellness_index" else x
    )
    available_features = INDEX_MAPPING[selected_index]
    feature_options = [(f, feature_desc.get(f, f)) for f in available_features]

    col1_desc = st.selectbox("Select Primary Feature", options=feature_options, format_func=lambda x: x[1])
    col1 = col1_desc[0]
    remaining_features = [f for f in feature_options if f[0] != col1]
    col2_desc = st.selectbox("Select Secondary Feature", options=remaining_features, format_func=lambda x: x[1])
    col2 = col2_desc[0]

# ========== Geospatial Preparation ==========
gdf_raw = gpd.GeoDataFrame.from_features(geojson_data["features"])
gdf_raw = gdf_raw.rename(columns={"pri_neigh": "FineNeighborhood"})
gdf_raw.set_crs(epsg=4326, inplace=True)
gdf = gdf_raw.merge(manual_mapping, on="FineNeighborhood", how="left")
gdf = gdf.merge(data, on="CommunityAreaName", how="left")
gdf["CommunityAreaName"] = gdf["CommunityAreaName"].fillna("Unknown")

# ========== Stats Functions ==========
def get_stats(data):
    return {'Min': data.min(), 'Q1': data.quantile(0.25), 'Mean': data.mean(), 'Q3': data.quantile(0.75), 'Max': data.max()}

def plot_full_stats(ax, data, title, chart_type='bar'):
    stats = get_stats(data)
    metrics, values = list(stats.keys()), list(stats.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if chart_type == 'bar':
        bars = ax.bar(metrics, values, color=colors)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", ha='center', va='bottom')
    elif chart_type == 'box':
        ax.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='#2ca02c'))
        stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_dual_comparison(ax, feature_data, index_data, feature_name, index_name):
    feature_stats, index_stats = get_stats(feature_data), get_stats(index_data)
    metrics, colors = list(feature_stats.keys()), ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(metrics, feature_stats.values(), color=colors, alpha=0.7)
    ax2 = ax.twinx()
    line, = ax2.plot(metrics, index_stats.values(), color='black', marker='o', linewidth=2)
    ax.set_title(f"{feature_name} vs {index_name}", fontsize=12)
    ax.legend([bars[0], line], [feature_name, index_name], loc='upper left')

# ========== Map Function ==========
def render_map(gdf, column, title):
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10)
    gdf_json = json.loads(gdf.to_json())
    for idx, feature in enumerate(gdf_json["features"]):
        feature["properties"]["CommunityAreaName"] = gdf.iloc[idx]["CommunityAreaName"]

    Choropleth(
        geo_data=gdf_json,
        name="choropleth",
        data=gdf,
        columns=["CommunityAreaName", column],
        key_on="feature.properties.CommunityAreaName",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        highlight=True,
        nan_fill_color='gray'
    ).add_to(m)

    tooltip_data = []
    for _, row in gdf.iterrows():
        value_str = f"{row[column]:.2f}" if pd.notna(row[column]) else "No data"
        tooltip_data.append(f"{row['CommunityAreaName']} {value_str}")

    tooltip_gdf = gdf.copy()
    tooltip_gdf["tooltip"] = tooltip_data
    folium.GeoJson(
        tooltip_gdf,
        name="Neighborhood Borders",
        tooltip=folium.GeoJsonTooltip(fields=["tooltip"], aliases=["Community: "])
    ).add_to(m)
    return m

# ========== Main Dashboard ==========
st.header("Choropleth Map")
st.markdown(f"### {selected_index} Geographic Distribution")
main_map = render_map(gdf, selected_index, selected_index)
st_folium(main_map, use_container_width=True)

st.header("Factor-Level Maps")
col_map1, col_map2 = st.columns(2)
with col_map1:
    st.subheader(f"{col1} Map")
    st_folium(render_map(gdf, col1, col1), use_container_width=True)

with col_map2:
    st.subheader(f"{col2} Map")
    st_folium(render_map(gdf, col2, col2), use_container_width=True)

st.header(f"{selected_index} Index Summary")
fig1, ax1 = plt.subplots(figsize=(10, 4))
plot_full_stats(ax1, data[selected_index], f"{selected_index} Stats\n{index_desc.get(selected_index, '')}")
st.pyplot(fig1)

st.header("Feature Distributions")
col_box1, col_box2 = st.columns(2)
with col_box1:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    plot_full_stats(ax2, data[col1], f"{col1}\n{feature_desc.get(col1, col1)}", chart_type='box')
    st.pyplot(fig2)

with col_box2:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    plot_full_stats(ax3, data[col2], f"{col2}\n{feature_desc.get(col2, col2)}", chart_type='box')
    st.pyplot(fig3)

st.header("Feature vs Index Comparison")
col_comp1, col_comp2 = st.columns(2)
with col_comp1:
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    plot_dual_comparison(ax4, data[col1], data[selected_index], col1, selected_index)
    st.pyplot(fig4)

with col_comp2:
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    plot_dual_comparison(ax5, data[col2], data[selected_index], col2, selected_index)
    st.pyplot(fig5)

with st.expander("🗂️ Full Feature Metadata"):
    st.dataframe(feature_metadata[["column_name", "description"]])

with st.expander("🗂️ Index Metadata"):
    st.dataframe(index_metadata)
