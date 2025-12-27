import sys
import os
# Add root to sys.path hack for sidebar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data.loader import get_merged_data
from src.models import BaseRecommender, PopularityRecommender, SVDRecommender
from src.experiment.ab_test import Simulator, ExperimentEngine
from src.experiment.analysis import analyze_ab_test
from sklearn.model_selection import train_test_split
from surprise import accuracy

# Page Config
st.set_page_config(page_title="MovieCube A/B Test Platform", layout="wide")

# Title
st.title("🎬 MovieCube: Recommender System & A/B Testing")

# Session State for Models
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading Data & Training Models..."):
        df = get_merged_data()
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        control = PopularityRecommender()
        control.fit(train_df)
        
        treatment = SVDRecommender(n_components=20)
        treatment.fit(train_df)
        
        st.session_state['train_df'] = train_df
        st.session_state['test_df'] = test_df
        st.session_state['control_model'] = control
        st.session_state['treatment_model'] = treatment
        st.session_state['data_loaded'] = True
        st.success("Models Trained!")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigate", ["User View (Simulator)", "Admin Dashboard (A/B Results)"])

if page == "User View (Simulator)":
    st.header("👤 User View")
    st.markdown("Simulate a user logging in and see which experience they get assigned to.")
    
    user_id = st.number_input("Enter User ID", min_value=1, max_value=100000, value=1)
    
    # Assign Group
    engine = ExperimentEngine()
    group = engine.assign_bucket(user_id)
    
    st.info(f"User {user_id} is assigned to group: **{group.upper()}**")

    # User History Feature
    st.subheader("User History (Rated 4.0+)")
    user_history = st.session_state['train_df'][
        (st.session_state['train_df']['userId'] == user_id) & 
        (st.session_state['train_df']['rating'] >= 4.0)
    ].sort_values("rating", ascending=False).head(5)
    
    if not user_history.empty:
        st.table(user_history[['title', 'rating']])
    else:
        st.caption("No highly rated movies found in training history for this user.")
    
    if st.button("Get Recommendations"):
        if group == "control":
            recs = st.session_state['control_model'].recommend(user_id, n=10)
            model_name = "Global Popularity (Control)"
        else:
            recs = st.session_state['treatment_model'].recommend(user_id, n=10)
            model_name = "SVD Matrix Factorization (Treatment)"
            
        st.subheader(f"Recommended Movies ({model_name})")
        
        # Display as a dataframe for now, or posters if we had URLs
        # Get Titles
        # Ideally we have a mapping from ID to Title from our dataframe
        # Let's rebuild the map quickly
        movie_map = st.session_state['train_df'][['movieId', 'title']].drop_duplicates().set_index('movieId')['title'].to_dict()
        
        rec_titles = [movie_map.get(mid, f"Unknown ID {mid}") for mid in recs]
        st.table(pd.DataFrame({"Movie Title": rec_titles}))

elif page == "Admin Dashboard (A/B Results)":
    st.header("📊 Admin Dashboard: A/B Test Simulation")
    
    st.markdown("Run a full-scale simulation on the held-out Test Set to measure recommender performance.")
    
    start_sim = st.button("Run Experiment Simulation")
    
    # Configurable Sandbox
    sample_size = st.slider("Simulation Sample Size (Users from Test Set)", min_value=100, max_value=2000, value=500, step=100)
    
    if start_sim:
        with st.spinner("Simulating user visits..."):
            test_users = st.session_state['test_df']['userId'].unique()
            # Sample for speed if needed
            if len(test_users) > sample_size:
                test_users = test_users[:sample_size]
                st.warning(f"Simulating on a subset of {len(test_users)} users (Adjust slider to change).")
            
            sim = Simulator(st.session_state['test_df'])
            results = sim.run_simulation(
                test_users, 
                st.session_state['control_model'], 
                st.session_state['treatment_model']
            )
            
            st.session_state['sim_results'] = results
            
            # Calculate RMSE for SVD (Treatment)
            # We predict on the specific test set for these users to get a technical metric
            # Note: Simulator results track 'conversion', but RMSE tracks 'rating prediction accuracy'
            st.session_state['treatment_rmse'] = st.session_state['treatment_model'].evaluate(st.session_state['test_df'])['rmse']
            
            st.success("Simulation Complete!")
            
    if 'sim_results' in st.session_state:
        results = st.session_state['sim_results']
        
        # 1. High Level Metrics
        summary, pval = analyze_ab_test(results)
        
        col1, col2, col3 = st.columns(3)
        c_rate = summary.loc['control', 'conversion_rate']
        t_rate = summary.loc['treatment', 'conversion_rate']
        uplift = (t_rate - c_rate) / c_rate
        
        col1.metric("Control Conv. Rate", f"{c_rate:.2%}")
        col2.metric("Treatment Conv. Rate", f"{t_rate:.2%}")
        col3.metric("Uplift", f"{uplift:.2%}", delta_color="normal" if uplift > 0 else "inverse")
        
        # 2. Charts
        st.subheader("Conversion Rates by Group")
        fig = px.bar(
            x=["Control", "Treatment"], 
            y=[c_rate, t_rate], 
            color=["Control", "Treatment"],
            labels={'x': 'Group', 'y': "Conversion Rate"},
            title="A/B Test Outcome"
        )
        st.plotly_chart(fig)
        
        # 3. Stats Significance
        st.subheader("Statistical Significance")
        if pval < 0.05:
            st.success(f"Result is Significant! (p-value = {pval:.4f})")
        else:
            st.error(f"Result is NOT Significant. (p-value = {pval:.4f})")
            
        # 4. Model Accuracy Metrics
        st.divider()
        st.subheader("Model Diagnostic Metrics")
        if 'treatment_rmse' in st.session_state:
            st.metric("SVD Model RMSE (Test Set)", f"{st.session_state['treatment_rmse']:.4f}", help="Lower is better. Measures rating prediction error.")
        else:
            st.info("Run simulation to calculate RMSE.")
