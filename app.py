import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import io 
import numpy as np # For spacing calculation

# Page Setting
st.set_page_config(page_title="Myanmar Project Mapper", layout="wide")

st.title("📍 Myanmar Project Area Mapper")
st.markdown("Project Coverage Township တွေကို single colored map လေးနဲ့ ပြနိုင်ဖို့  High-Quality map လေးတစ်ခု အချိန်တိုအတွင်း ဖန်တီးနိုင်စေဖို့ ရည်ရွယ်တဲ့ Tool လေးဖြစ်ပါတယ်။")

# --- 1. Load Data Function ---
@st.cache_data
def load_geodata():
    current_dir = os.getcwd()
    shp_files = [f for f in os.listdir(current_dir) if f.endswith('.shp')]
    
    if shp_files:
        path = os.path.join(current_dir, shp_files[0])
        return gpd.read_file(path), "Shapefile"
    
    json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]
    if json_files:
        path = os.path.join(current_dir, json_files[0])
        return gpd.read_file(path), "GeoJSON"
        
    return None, None

@st.cache_data
def get_state_boundaries(_gdf, sr_col):
    return _gdf.dissolve(by=sr_col)

# Data Loading
try:
    gdf, file_type = load_geodata()
    if gdf is None:
        st.error("❌ Map Data မတွေ့ပါ! Folder ထဲတွင် .shp ဖိုင် ထည့်ထားပေးပါ။")
        st.stop()
    else:
        pass 
except Exception as e:
    st.error(f"❌ Error loading map data: {e}")
    st.stop()

# --- Auto-Detect Columns ---
cols = gdf.columns
possible_pcode = ['TS_PCODE', 'TS_PCODE_M', 'MMR_Township_Pcode', 'pcode']
pcode_col = next((c for c in possible_pcode if c in cols), None)

possible_name = ['TS_NAME_E', 'Township_Name_Eng', 'TS', 'township', 'TspName']
name_col = next((c for c in possible_name if c in cols), cols[0])

possible_sr = ['ST_NAME_E', 'SR_NAME_E', 'ST', 'SR', 'State_Region', 'Region']
sr_col = next((c for c in possible_sr if c in cols), None)

if not pcode_col:
    st.error("Data ထဲတွင် Township Pcode column ရှာမတွေ့ပါ။")
    st.stop()

# --- Session State ---
if 'selected_townships_set' not in st.session_state:
    st.session_state.selected_townships_set = set()

# --- 2. Sidebar Controls ---
st.sidebar.title("1. Data Input")
input_method = st.sidebar.radio("Input Method:", ["Manual Select", "Excel Upload"])

highlight_data = None

if input_method == "Manual Select":
    if sr_col:
        sr_list = sorted([str(x) for x in gdf[sr_col].unique() if x is not None])
        selected_sr = st.sidebar.selectbox("Filter by State/Region:", ["All (အားလုံး)"] + sr_list)
        
        if selected_sr != "All (အားလုံး)":
            filtered_gdf = gdf[gdf[sr_col] == selected_sr]
            available_townships = sorted(filtered_gdf[name_col].unique())
        else:
            available_townships = sorted(gdf[name_col].unique())
    else:
        available_townships = sorted(gdf[name_col].unique())

    current_default = list(st.session_state.selected_townships_set.intersection(available_townships))
    
    with st.sidebar.form(key="selection_form"):
        st.write("Township Selection:")
        user_selection = st.multiselect(
            label="Select Townships", 
            options=available_townships,
            default=current_default,
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button(label="Apply Selection (အတည်ပြုရန်)")

    if submit_button:
        removed_items = set(current_default) - set(user_selection)
        st.session_state.selected_townships_set.difference_update(removed_items)
        st.session_state.selected_townships_set.update(user_selection)
        st.rerun()

    if st.session_state.selected_townships_set:
        highlight_data = gdf[gdf[name_col].isin(st.session_state.selected_townships_set)]
        # --- Statistics Feature ---
        st.sidebar.markdown("---")
        st.sidebar.write("📊 **Summary:**")
        st.sidebar.write(f"- Total Townships: **{len(highlight_data)}**")
        if sr_col:
            unique_states = highlight_data[sr_col].nunique()
            st.sidebar.write(f"- States/Regions Covered: **{unique_states}**")
    else:
        st.sidebar.warning("No townships selected.")

elif input_method == "Excel Upload":
    st.sidebar.info("Excel must contain **TS_PCODE**.")
    uploaded_file = st.sidebar.file_uploader("Upload File:", type=["xlsx", "csv"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            excel_cols = [c.upper() for c in df.columns]
            if 'TS_PCODE' in excel_cols:
                target_col = df.columns[excel_cols.index('TS_PCODE')]
                valid_pcodes = df[target_col].astype(str).tolist()
                highlight_data = gdf[gdf[pcode_col].isin(valid_pcodes)]
                st.sidebar.success(f"Matched: {len(highlight_data)} townships")
            else:
                st.sidebar.error("Column 'TS_PCODE' not found.")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# --- 3. Plotting Logic ---
col1, col2 = st.columns([3, 1])

with col1:
    # Changed from st.subheader to st.header to match sidebar font size
    st.subheader("2. Map Customization")
    
    # Layout: Use 4 columns to put everything in one row
    # Ratios adjusted: Checkboxes need less space, Title needs more
    c1, c2, c3, c4 = st.columns([0.8, 0.8, 0.8, 1.6])
    
    with c1:
        map_color = st.color_picker("Area Color", "#FF4B4B")
        
    with c2:
        # Add spacing to align checkbox with input fields vertically
        st.write("") 
        st.write("")
        show_stats = st.checkbox("Show Summary Stats", value=True) 
        
    with c3:
        # Add spacing to align checkbox with input fields vertically
        st.write("")
        st.write("")
        show_labels = st.checkbox("Show Township Names", value=False)
        
    with c4:
        map_title = st.text_input("Map Title", value="Project Target Areas")
    
    generate_btn = st.button("Generate Map (မြေပုံထုတ်ရန်)", type="primary")

    if generate_btn:
        with st.spinner("Generating High-Quality Map..."):
            fig, ax = plt.subplots(figsize=(10, 12)) 
            
            # --- Get Bounds First ---
            min_x, min_y, max_x, max_y = gdf.total_bounds

            # 1. Base Map (White)
            gdf.plot(ax=ax, color='white', edgecolor='#e0e0e0', linewidth=0.3)
            
            # 2. Highlight Active State/Regions (Background Color)
            if highlight_data is not None and not highlight_data.empty and sr_col:
                active_states_list = highlight_data[sr_col].unique()
                active_states_subset = gdf[gdf[sr_col].isin(active_states_list)]
                active_states_subset.plot(ax=ax, facecolor='#f2f2f2', edgecolor='#e0e0e0', linewidth=0.3)

            # 3. State Boundaries (Overlay lines)
            if sr_col:
                state_gdf = get_state_boundaries(gdf, sr_col)
                state_gdf.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=1.2)
            
            # 4. Highlight Selected Townships
            if highlight_data is not None and not highlight_data.empty:
                highlight_data.plot(ax=ax, color=map_color, edgecolor='black', linewidth=0.8)
                
                # --- White Area Label Logic ---
                if show_labels:
                    mid_x = (min_x + max_x) / 2
                    left_margin_x = min_x - 0.5 
                    right_margin_x = max_x + 0.5 
                    
                    points = highlight_data.copy()
                    points['centroid'] = points.geometry.centroid
                    points['y_coord'] = points['centroid'].y
                    points['x_coord'] = points['centroid'].x
                    
                    left_group = points[points['x_coord'] < mid_x]
                    right_group = points[points['x_coord'] >= mid_x]
                    
                    def plot_grouped_stack(group, anchor_x, align):
                        if group.empty: return
                        
                        ordered_items = []
                        if sr_col and sr_col in group.columns:
                            state_order = group.groupby(sr_col)['y_coord'].mean().sort_values(ascending=True).index
                            for state in state_order:
                                state_towns = group[group[sr_col] == state].sort_values('y_coord', ascending=True)
                                for _, row in state_towns.iterrows():
                                    ordered_items.append((row, state))
                        else:
                            sorted_rows = group.sort_values('y_coord', ascending=True)
                            for _, row in sorted_rows.iterrows():
                                ordered_items.append((row, "Unknown"))

                        base_spacing = 0.22
                        state_gap = 0.5
                        y_offsets = []
                        current_y = 0
                        last_state = ordered_items[0][1] if ordered_items else None
                        
                        for i, (row, state) in enumerate(ordered_items):
                            if i > 0:
                                step = base_spacing
                                if state != last_state: step += state_gap
                                current_y += step
                            y_offsets.append(current_y)
                            last_state = state
                            
                        total_height = current_y
                        avg_map_y = group['y_coord'].mean()
                        start_y = avg_map_y - (total_height / 2)
                        
                        for i, (row, state) in enumerate(ordered_items):
                            pt_x, pt_y = row['x_coord'], row['y_coord']
                            label_text = row[name_col]
                            target_y = start_y + y_offsets[i]
                            
                            ax.annotate(
                                text=label_text,
                                xy=(pt_x, pt_y),
                                xytext=(anchor_x, target_y),
                                ha=align, va='center',
                                fontsize=8, fontweight='bold',
                                arrowprops=dict(arrowstyle="-", color='#666666', linewidth=0.5, relpos=(1, 0.5) if align == 'right' else (0, 0.5))
                            )

                    plot_grouped_stack(left_group, left_margin_x, 'right')
                    plot_grouped_stack(right_group, right_margin_x, 'left')

            # 5. Summary Stats
            if show_stats and highlight_data is not None and not highlight_data.empty:
                total_ts = len(highlight_data)
                total_sr = highlight_data[sr_col].nunique() if sr_col else 0
                stats_text = f"Total State/Regions: {total_sr}\nTotal Townships: {total_ts}"
                
                ax.text(0.2, 0.2, stats_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#aaaaaa'))

            # Title & Layout
            ax.set_title(map_title, fontsize=15, fontweight='bold', pad=5)
            ax.set_axis_off() 
            ax.set_xlim(min_x - 3.5, max_x + 3.5)
            
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="📥 Download High-Res Image",
                data=buf,
                file_name="project_map.png",
                mime="image/png"
            )

with col2:
    st.write("### Instructions")
    st.info("""
    1. **Data Input** မှာ နှစ်သက်ရာ နည်းလမ်းကို ရွေးပါ။ **Manual Select** နည်းလမ်းတွင် State/Region ရွေးပြီး ပါဝင်မည့် Township များ ဆက်တိုက်ရွေးကာ Apply ကိုနှိပ်ပါ။ နောက်ထပ် State/Region များအလိုက် Township များ ဆက်လက်ရွေးချယ်နိုင်ပါသည်။
    **Excel Upload** ပြုလုပ်လိုပါက "TS_PCODE" Title ဖြင့် MIMU township code column ပါဝင်သော file ကို upload လုပ်နိုင်ပါသည်။
    
    2. **Map Customization** တွင် နှစ်သက်ရာ အ‌ရောင်၊ Township နာမည် နှင့် Summary Stats ဖော်ပြ/မပြ ရွေးချယ်နိုင်ပါသည်။ Map Title တွင် ပေးလိုသော နာမည်ကို ရိုက်ထည့်နိုင်ပါသည်။
    
    3. အားလုံးပြီးပါက **Generate Map** ကို နှိပ်ကာ Mapထုတ်ယူနိုင်ပါပြီ။ ရရှိလာသော Mapကို Mouse မှ Right Click နှိပ်ကာ Copy ကူး၍ သုံးစွဲနိုင်သလို အောက်ခြေရှိ Download ကို နှိပ်၍လည်း Save ပြုလုပ်နိုင်ပါသည်။ Mapကို ပြင်ဆင်လိုပါက ပြန်လည်ရွေးချယ်လိုသည်များပြုလုပ်ကာ Generate Map ကို ထပ်မံနှိပ်ရပါမည်။
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: grey; font-size: 14px;'>
    All rights reserved 2025 | created by <a href='https://phonepyae.online/' target='_blank' style='text-decoration: none; color: #4A90E2;'>Phone Pyae</a>
    </p>
    """, 
    unsafe_allow_html=True
)