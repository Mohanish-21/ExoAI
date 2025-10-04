"""
EXO AI - Exoplanet Detection and Habitability Analyzer
Built by Team ExoAI Explorers for NASA Space Apps Challenge 2025
"""

import streamlit as st
from integration_pipeline import analyze_manual, analyze_csv
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="EXO AI - Exoplanet Analyzer",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🌍 EXO AI")
st.markdown("### Exoplanet Detection and Habitability Analyzer")
st.markdown("**AI-Powered Analysis Using NASA Mission Data**")
st.markdown("---")

# Description
st.markdown("""
Discover and analyze exoplanets using advanced machine learning. 
EXO AI processes data from NASA's Kepler, K2, and TESS missions 
to detect exoplanets with **78.5% accuracy** and evaluate their potential 
for habitability. Whether analyzing a single planet or processing entire 
datasets, our dual-AI system accelerates the search for habitable 
worlds beyond our solar system.
""")

st.markdown("---")

# Mode selection
st.subheader("Choose Your Analysis Method:")
col1, col2 = st.columns(2)

with col1:
    if st.button("📝 Manual Entry", use_container_width=True, type="primary"):
        st.session_state.mode = "manual"

with col2:
    if st.button("📁 CSV Upload", use_container_width=True, type="primary"):
        st.session_state.mode = "csv"

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = None

st.markdown("---")

# Manual Entry Mode
if st.session_state.mode == "manual":
    st.header("📝 Enter Planet Parameters")
    
    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.number_input("Orbital Period (days)", value=365.0, min_value=0.1, help="Time for one complete orbit around the star")
            radius = st.number_input("Planet Radius (Earth radii)", value=1.0, min_value=0.1, help="Planet radius relative to Earth")
            depth = st.number_input("Transit Depth (ppm)", value=84.0, min_value=0.0, help="Drop in star brightness during transit")
        
        with col2:
            duration = st.number_input("Transit Duration (hours)", value=13.0, min_value=0.1, help="Duration of the transit event")
            temp = st.number_input("Planet Temperature (Kelvin)", value=288.0, min_value=0.0, help="Equilibrium temperature of the planet")
            star_temp = st.number_input("Star Temperature (Kelvin)", value=5778.0, min_value=0.0, help="Temperature of the host star")
        
        snr = st.number_input("Signal-to-Noise Ratio (optional)", value=50.0, min_value=0.0, help="Quality of the detection signal")
        
        submitted = st.form_submit_button("🚀 Analyze Planet", use_container_width=True)
        
        if submitted:
            with st.spinner("🔍 Analyzing planet data..."):
                try:
                    user_input = {
                        'period': period,
                        'radius': radius,
                        'transit_depth': depth,
                        'transit_duration': duration,
                        'temperature': temp,
                        'stellar_temp': star_temp,
                        'signal_noise_ratio': snr
                    }
                    
                    result = analyze_manual(user_input)
                    
                    st.success("✅ Analysis Complete!")
                    st.markdown("---")
                    
                    # Results display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔍 Detection Result")
                        classification = result['detection']['classification']
                        confidence = result['detection']['confidence']
                        
                        if classification == "CONFIRMED":
                            st.success(f"### ✅ {classification}")
                        elif classification == "CANDIDATE":
                            st.warning(f"### ⚠️ {classification}")
                        else:
                            st.error(f"### ❌ {classification}")
                        
                        st.metric("Confidence", f"{confidence}%")
                        
                        st.write("**Probabilities:**")
                        for cls, prob in result['detection']['probabilities'].items():
                            st.progress(prob/100, text=f"{cls}: {prob:.1f}%")
                    
                    with col2:
                        st.subheader("🌟 Habitability Analysis")
                        if result['habitability']['classification'] != 'N/A':
                            hab_class = result['habitability']['classification']
                            hab_score = result['habitability']['percentage']
                            
                            if "Highly" in hab_class:
                                st.success(f"### ⭐ {hab_class}")
                            elif "Potentially" in hab_class:
                                st.warning(f"### 🌍 {hab_class}")
                            else:
                                st.error(f"### ❌ {hab_class}")
                            
                            st.metric("Habitability Score", f"{hab_score}%")
                            
                            st.write("**Classification Breakdown:**")
                            for cls, prob in result['habitability']['probabilities'].items():
                                st.progress(prob/100, text=f"{cls}: {prob:.1f}%")
                        else:
                            st.info("ℹ️ Habitability analysis only available for confirmed planets")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("💡 Please check your input values and try again")

# CSV Upload Mode
elif st.session_state.mode == "csv":
    st.header("📁 Upload CSV Dataset")
    
    st.info("💡 Upload a CSV file with NASA Exoplanet Archive format. Required columns: koi_period, koi_prad, koi_depth, koi_duration, koi_teq, koi_steff")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"**Preview:** {len(df)} planets found")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Analyze Dataset", use_container_width=True, type="primary"):
                with st.spinner(f"🔍 Processing {len(df)} planets..."):
                    try:
                        # Save temporarily
                        temp_path = "temp_upload.csv"
                        df.to_csv(temp_path, index=False)
                        
                        # Analyze
                        results = analyze_csv(temp_path)
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        st.success(f"✅ Analysis Complete! Processed {len(results)} planets")
                        st.markdown("---")
                        
                        # Summary stats
                        confirmed = sum(1 for r in results if r['detection']['classification'] == 'CONFIRMED')
                        candidates = sum(1 for r in results if r['detection']['classification'] == 'CANDIDATE')
                        false_pos = sum(1 for r in results if r['detection']['classification'] == 'FALSE_POSITIVE')
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("✅ Confirmed", confirmed, f"{confirmed/len(results)*100:.1f}%")
                        col2.metric("⚠️ Candidates", candidates, f"{candidates/len(results)*100:.1f}%")
                        col3.metric("❌ False Positives", false_pos, f"{false_pos/len(results)*100:.1f}%")
                        
                        st.markdown("---")
                        
                        # Results table
                        st.subheader("📊 Detailed Results")
                        table_data = []
                        for r in results:
                            table_data.append({
                                'Planet #': r['index'] + 1,
                                'Detection': r['detection']['classification'],
                                'Confidence (%)': round(r['detection']['confidence'], 1),
                                'Habitability': r['habitability']['classification'],
                                'Hab Score (%)': round(r['habitability'].get('percentage', 0), 1)
                            })
                        
                        results_df = pd.DataFrame(table_data)
                        st.dataframe(results_df, use_container_width=True, height=400)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="exoai_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Error analyzing dataset: {str(e)}")
                        st.info("💡 Please check your CSV format and column names")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Make sure your file is a valid CSV with the required columns")

# Add spacing before footer
st.markdown("<br><br><br>", unsafe_allow_html=True)

# FOOTER - Team names at the very bottom
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-top: 50px;'>
        <h2 style='color: white; margin-bottom: 15px;'>🚀 Team ExoAI Explorers</h2>
        <p style='color: white; font-size: 18px; margin: 10px 0;'>
            <b>Backend:</b> Mohanish R • Akarsh PT
        </p>
        <p style='color: white; font-size: 18px; margin: 10px 0;'>
            <b>Frontend:</b> Nithya • Abhinav UV
        </p>
        <p style='color: rgba(255,255,255,0.8); margin-top: 20px; font-size: 16px;'>
            NASA Space Apps Challenge 2025 | Mysore, India 🇮🇳
        </p>
        <p style='color: rgba(255,255,255,0.7); font-size: 14px; margin-top: 10px;'>
            Built  during a 48-hour hackathon
        </p>
    </div>
    """, unsafe_allow_html=True)
