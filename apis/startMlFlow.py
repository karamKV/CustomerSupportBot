import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go  # Add this import
import os
import requests
from datetime import datetime
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
os.environ['STREAMLIT_SERVER_PORT'] = '8000'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

def load_mlflow_metrics():
    """Load metrics from MLflow metrics.csv"""
    try:
        metrics_file = Path("metricsCSVPath")
        if metrics_file.exists():
            return pd.read_csv(metrics_file)
        return None
    except Exception as e:
        logger.error(f"Error loading MLflow metrics: {e}")
        return None

def query_rag_system(user_query: str) -> dict:
    """Send query to RAG API and get response."""
    try:
        response = requests.post(
            "QueryURL",
            headers={"Content-Type": "application/json"},
            json={"userQuery": user_query},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return {"error": str(e)}

def get_query_metrics(userQuery: str, context: str, answer: str) -> dict:
    """Get metrics for a specific query using the evaluation endpoint."""
    try:
        url = "QueryMetricesURL"  # Adjust the port number as needed
        
        payload = {
            "userQuery": userQuery,
            "context": context,
            "answer": answer
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get query metrics: {e}")
        return {"error": str(e)}

def format_metadata_markdown(metadata):
    """Format metadata as markdown."""
    markdown=""
    for j in metadata:
        for key, value in j.items():
            markdown += f"**{key}:** {value}  \n"
    print(markdown)
    return markdown

def main():
    st.set_page_config(
        page_title="RAG Experiment Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    tabs = st.tabs(["Experiment Metrics", "Query Testing", "Query Logs"])

    # Tab 1: Experiment Metrics
    with tabs[0]:
        st.title("RAG Experiment Metrics")
        
        # Load metrics from MLflow
        df = load_mlflow_metrics()
        print(df)
        if df is not None:
            # Display current metrics
            col1, col2, col3 = st.columns(3)
            numeric_columns = ['retriever_accuracy', 'avg_response_length', 'avg_query_length', 'relevance_score', 'groundedness_score']
    
            # Convert and clean numeric columns
            for col in numeric_columns:
                if col in df.columns:
                    # Remove % signs and convert to float if column contains percentages
                    df[col] = df[col].astype(str).str.rstrip('%').astype(float) / 100 if '%' in str(df[col].iloc[0]) else df[col].astype(float)
        
            # Calculate averages for numeric columns
            latest_metrics = df[numeric_columns].mean()
            
            with col1:
                st.metric(
                    "Retriever Accuracy",
                    f"{latest_metrics.get('retriever_accuracy', 0):.2%}"
                )
            
            
            with col2:
                st.metric(
                    "Relevance Score",
                    f"{latest_metrics.get('relevance_score', 0):.2f}"
                )
            
            with col3:
                st.metric(
                    "Groundedness Score",
                    f"{latest_metrics.get('groundedness_score', 0):.2f}"
                )
            
            # Historical metrics visualization
            st.subheader("Metrics Over Time")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add date range filter
            date_range = st.date_input(
                "Select Date Range",
                [df['timestamp'].min().date(), df['timestamp'].max().date()]
            )
            
            # Filter data based on date range
            mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
            filtered_df = df[mask]
            
            # Create tabs for different metrics
            metrics_tabs = st.tabs([
                "All Metrics",
                "Accuracy",
                "Response Length",
                "Query Length",
                "Relevance & Groundedness"
            ])
            
            # Plot All Metrics
            with metrics_tabs[0]:
                # Create figure with secondary y-axis
                fig = go.Figure()
                
                # Add accuracy line (percentage scale)
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['timestamp'],
                        y=filtered_df['retriever_accuracy'],
                        name='Accuracy',
                        line=dict(color='blue'),
                        yaxis='y'
                    )
                )
                
                # Add other metrics
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['timestamp'],
                        y=filtered_df['relevance_score'],
                        name='Relevance Score',
                        line=dict(color='green'),
                        yaxis='y2'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['timestamp'],
                        y=filtered_df['groundedness_score'],
                        name='Groundedness Score',
                        line=dict(color='red'),
                        yaxis='y2'
                    )
                )
                
                # Update layout with two y-axes
                fig.update_layout(
                    title='All Metrics Over Time',
                    yaxis=dict(
                        title='Accuracy',
                        tickformat='.2%',
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Scores',
                        overlaying='y',
                        side='right'
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot Accuracy
            with metrics_tabs[1]:
                fig_accuracy = px.line(
                    filtered_df,
                    x='timestamp',
                    y='retriever_accuracy',
                    title='Accuracy Over Time'
                )
                fig_accuracy.update_layout(
                    yaxis_tickformat='.2%',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_accuracy, use_container_width=True)
            
            # Plot Response Length
            with metrics_tabs[2]:
                fig_response = px.line(
                    filtered_df,
                    x='timestamp',
                    y='avg_response_length',
                    title='Average Response Length Over Time'
                )
                fig_response.update_layout(hovermode='x unified')
                st.plotly_chart(fig_response, use_container_width=True)
            
            # Plot Query Length
            with metrics_tabs[3]:
                fig_query = px.line(
                    filtered_df,
                    x='timestamp',
                    y='avg_query_length',
                    title='Average Query Length Over Time'
                )
                fig_query.update_layout(hovermode='x unified')
                st.plotly_chart(fig_query, use_container_width=True)
            
            # Plot Relevance and Groundedness
            with metrics_tabs[4]:
                fig_scores = go.Figure()
                
                # Add Relevance Score
                fig_scores.add_trace(
                    go.Scatter(
                        x=filtered_df['timestamp'],
                        y=filtered_df['relevance_score'],
                        name='Relevance Score',
                        line=dict(color='green')
                    )
                )
                
                # Add Groundedness Score
                fig_scores.add_trace(
                    go.Scatter(
                        x=filtered_df['timestamp'],
                        y=filtered_df['groundedness_score'],
                        name='Groundedness Score',
                        line=dict(color='red')
                    )
                )
                
                fig_scores.update_layout(
                    title='Relevance and Groundedness Scores Over Time',
                    yaxis_title='Score',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_scores, use_container_width=True)
            
            # Display raw data with download option
            st.subheader("Raw Metrics Data")
            st.dataframe(filtered_df)
            
            # Add download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"rag_metrics_{date_range[0]}_{date_range[1]}.csv",
                mime="text/csv"
            )

    # Tab 2: Query Testing
    with tabs[1]:
        st.header("Test RAG System")
        
        with st.form("query_form"):
            user_query = st.text_input(
                "Enter your query:",
                placeholder="What is the process for password reset?"
            )
            submit_button = st.form_submit_button("Submit Query")
        
        if submit_button and user_query:
            with st.spinner('Getting response...'):
                response = query_rag_system(user_query)
                
                if "error" not in response:
                    # Display query and response
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader("Query")
                        st.info(user_query)
                    
                    with col2:
                        st.subheader("Response")
                        if 'data' in response and 'answer' in response['data']:
                            st.success(response['data']['answer'])
                        else:
                            st.warning("No answer provided")
                    
                    with col3:
                        st.subheader("Metadata")
                        if 'data' in response and 'metadata' in response['data']:
                            # Format and display metadata
                            metadata_md = format_metadata_markdown(response['data']['metadata'])
                            st.markdown(metadata_md)
                            
                            # Display sources if available
                            if 'sources' in response['data']:
                                st.subheader("Sources")
                                for idx, source in enumerate(response['data']['sources'], 1):
                                    st.write(f"{idx}. {source}")
                        else:
                            st.warning("No metadata available")
                    
                    # Add to history with all information
                    query_metrics = get_query_metrics(user_query,response['data']['answer'],response['data']['context'])
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "query": user_query,
                        "response": response,
                        "metadata": response.get("metadata", {}),
                        "metrics": query_metrics
                    })
                else:
                    st.error(f"Error: {response['error']}")
    
    # Tab 3: Query Logs
    with tabs[2]:
        st.header("Query Logs")
        
        if st.session_state.query_history:
            for idx, item in enumerate(reversed(st.session_state.query_history), 1):
                with st.expander(f"Query {idx}: {item['query'][:50]}..."):
                    st.write(f"**Timestamp:** {item['timestamp']}")
                    
                    # Query, Response, and Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Query:**")
                        st.info(item['query'])
                    
                    with col2:
                        st.write("**Response:**")
                        st.success(item['response']['data'].get("answer", ""))
                    
                    with col3:
                        st.write("**Metadata:**")
                        if 'data' in item['response'] and 'metadata' in item['response']['data']:
                            st.markdown(format_metadata_markdown(item['response']['data']['metadata']))
                    
                    # Quality Metrics in a separate section
                    if "metrics" in item and "error" not in item["metrics"]:
                        st.write("**Quality Metrics:**")
                        metric_cols = st.columns(2)
                        
                        with metric_cols[0]:
                            st.metric("Relevance Score", 
                                    f"{item['metrics'].get('relevance', 0):.2f}")
                        
                        with metric_cols[1]:
                            st.metric("Groundedness Score", 
                                    f"{item['metrics'].get('groundedness', 0):.2f}")
                        
                    
                    # Sources
                    if 'data' in item['response'] and 'sources' in item['response']['data']:
                        st.write("**Sources:**")
                        for idx, source in enumerate(item['response']['data']['sources'], 1):
                            st.write(f"{idx}. {source}")
        else:
            st.info("No queries logged yet. Try testing some queries in the Query Testing tab.")

if __name__ == "__main__":
    main()