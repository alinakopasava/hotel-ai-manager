import streamlit as st
import pandas as pd
from generating_response import generate_polite_response_with_models
import os

# ustawienia strony i ikona hotelu
st.set_page_config(page_title="Hotel Review AI Manager", page_icon="🏨", layout="wide")

st.title("Hotel Review AI Manager")
st.markdown("""
This application analyzes guest reviews using local machine learning models (**BERT** for Sentiment Analysis, 
**XGBoost** for Category Classification) and generates professional, automated responses using the **Gemini API**.
""")

# panel boczny ze statystykami
st.sidebar.header("System Dashboard")
log_file = 'hotel_issues_log.csv'

if os.path.exists(log_file):
    try:
        df_log = pd.read_csv(log_file)
        # ile recenzji już sprawdziliśmy w sumie
        st.sidebar.metric("Total Processed", len(df_log))
        
        if not df_log.empty:
            last_sent = df_log['Sentiment'].iloc[-1]
            st.sidebar.write(f"Latest Sentiment: **{last_sent}**")
        
        st.sidebar.divider()
        # przycisk do czyszczenia historii logów
        if st.sidebar.button("Clear Log History"):
            os.remove(log_file)
            st.sidebar.success("Logs cleared!")
            st.rerun()
    except Exception as e:
        st.sidebar.error("Error loading log file.")
else:
    st.sidebar.info("No logs found yet.")

# miejsce do wpisania opinii gościa
review_input = st.text_area("Enter Guest Review:", placeholder="Guest review...", height=150)

if st.button("Analyze & Generate Response"):
    if not review_input.strip():
        st.warning("Please enter a guest review before proceeding")
    else:
        with st.spinner("Analyzing review and generating response..."):
            try:
                # wywołuje funkcję z pliku generating_response.py
                result = generate_polite_response_with_models(review_input)
                
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Analysis Results")
                    sentiment = result['sentiment']
                    # koloruje sentyment na zielono/czerwono
                    color = "green" if sentiment == "positive" else "red"
                    st.markdown(f"**Sentiment:** :{color}[{sentiment.upper()}]")
                    st.markdown(f"**Category:** `{result['category']}`")
                
                with col2:
                    st.subheader("Technical Details")
                    st.write("**Engine:** BERT + XGBoost")
                    st.write("**Agent:** Gemini 2.5 Flash")
                
                # gotowa odpowiedź 
                st.subheader("Generated Response")
                st.success(result['generated_response'])
                
                st.toast("Saved to logs", icon='💾')
                
            except Exception as e:
                st.error(f"Error: {e}")

# podgląd historii na samym dole strony
if os.path.exists(log_file):
    with st.expander("View Log History"):
        try:
            full_log = pd.read_csv(log_file)
            # pokazuje najnowsze recenzje na samej górze
            st.dataframe(full_log.sort_index(ascending=False), use_container_width=True)
        except:
            st.write("Log history is currently empty.")