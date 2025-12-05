import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

from database import init_db, save_entry, fetch_history
from agents.guardian import GuardianAgent
from agents.analyst import AnalystAgent
from agents.coach import CoachAgent

st.set_page_config(page_title="ReflectAI", layout="centered")

# Initialize Agents
if 'guardian' not in st.session_state:
    st.session_state.guardian = GuardianAgent()
if 'analyst' not in st.session_state:
    # Analyst uses the Mixed RoBERTa model + Simple GPT prompt
    st.session_state.analyst = AnalystAgent(use_local_model=True)
if 'coach' not in st.session_state:
    st.session_state.coach = CoachAgent()

init_db()

# SIDEBAR: Safety & Navigation
with st.sidebar:
    st.title("ReflectAI")
    st.markdown("---")
    st.warning("⚠️ **Not Medical Advice**\n\nIf you are in crisis, please call 988 or go to the nearest emergency room.")
    st.markdown("---")
    
    page = st.radio("Navigate", ["New Entry", "My Insights"])

# PAGE 1: NEW ENTRY (The Daily Loop)
if page == "New Entry":
    st.header("Daily Reflection")
    st.write("How are you feeling today?")

    journal_text = st.text_area("Journal Entry", height=150, placeholder="Write your thoughts here...")
    
    st.write("## Daily Context")
    st.caption("Let us obtain some general information for later insights. How was your baseline health today?")

    col1, col2 = st.columns(2)
    with col1:
        sleep = st.slider(
            "Hours Slept (Last Night)", 
            0, 12, 7,
            help="This helps the Coach find patterns between your emotions and sleep cycles (e.g., 'You feel more anxious when you sleep less than 6 hours')."
        )
    with col2:
        stress = st.slider(
            "Current Stress Level (Now)", 
            1, 10, 5,
            help="Rate your general stress for the current moment, as you are making this journal entry."
        )

    if st.button("Save Entry", type="primary"):
        if not journal_text.strip():
            st.error("Please write something before saving.")
        else:
            with st.spinner("Analyzing..."):
                # STEP A: Safety Check
                safety_result = st.session_state.guardian.analyze(journal_text)
                
                if safety_result["is_risk"]:
                    st.error("**You are not alone.**")
                    st.markdown("""
                        I hear how much difficulty you are in right now, and you should know that you're not alone in this. 
                        
                        You don't have to carry this weight by yourself. There are real people ready to listen and help you get through this moment, judgement-free.
                        
                        **Please reach out to them:**
                        * **988** (Suicide & Crisis Lifeline) - Call or Text anytime.
                        * **Text HOME to 741741** (Crisis Text Line).
                        * [Find a Therapist near you](https://www.psychologytoday.com/us)
                        
                        *The application has paused analysis to prioritize your safety.*
                    """)
                    save_entry(journal_text, sleep, stress, "High Risk", "Crisis", True)
                
                else:
                    # STEP B: Emotion Extraction
                    emotions = st.session_state.analyst.analyze_emotions(journal_text)
                    
                    # STEP C: Trigger Extraction 
                    triggers = st.session_state.analyst.extract_triggers(journal_text)
                    
                    # STEP D: Save to Database
                    save_entry(journal_text, sleep, stress, emotions, triggers, False)
                    
                    st.success("Entry saved successfully!")
                    
                    # Display Results
                    c1, c2 = st.columns(2)
                    c1.info(f"**Emotions:** {emotions}")
                    c2.info(f"**Topics:** {triggers}")

# PAGE 2: INSIGHTS (The Weekly Loop)
elif page == "My Insights":
    st.header("Your Wellbeing Dashboard")
    
    df = fetch_history()
    
    if df.empty:
        st.info("No entries yet. Go to 'New Entry' to start journaling!")
    else:
        # Recent History Table
        st.subheader("Recent Entries")
        st.dataframe(df[['date', 'emotions', 'triggers', 'sleep_hours', 'stress_level']].head(5), use_container_width=True)
        
        # Stress Chart
        st.subheader("Stress Trends")
        chart = alt.Chart(df).mark_line(point=True).encode(
            x='date:T',
            y='stress_level:Q',
            tooltip=['date', 'emotions', 'stress_level']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        
        # Trigger Frequency Chart 
        st.subheader("What affects you the most?")
        
        if 'triggers' in df.columns and not df['triggers'].dropna().empty:
            all_triggers = [
                t.strip() 
                for sublist in df['triggers'].dropna().str.split(',') 
                for t in sublist 
                if t.strip()
            ]
            
            if all_triggers:
                trigger_counts = pd.Series(all_triggers).value_counts().reset_index()
                trigger_counts.columns = ['Trigger', 'Count']
                
                bar_chart = alt.Chart(trigger_counts).mark_bar().encode(
                    x=alt.X('Count', title='Frequency', axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y('Trigger', sort='-x', title='Topic'),
                    color=alt.Color('Count', legend=None),
                    tooltip=['Trigger', 'Count']
                ).properties(height=300)
                
                st.altair_chart(bar_chart, use_container_width=True)
            else:
                st.info("Not enough trigger data to show trends yet.")
        else:
            st.info("Start journaling to see what triggers your emotions.")
        
        # Weekly Coach Section
        st.markdown("---")
        st.subheader("Weekly Coach")
        
        # Allow user to pick the timeframe
        days_back = st.slider(
            "How many days of history should the Coach analyze?",
            min_value=1,
            max_value=30,
            value=7,
            help="Select 7 for a weekly review, or 14/30 for longer-term trends."
        )
        
        if st.button(f"Generate Insight for Last {days_back} Days", key="btn_weekly_review"):
            with st.spinner("The Coach is reading your journal..."):
                
                df['date'] = pd.to_datetime(df['date'])
                cutoff_date = datetime.now() - timedelta(days=days_back)
                analysis_df = df[df['date'] >= cutoff_date]
                
                if analysis_df.empty:
                    st.warning(f"No entries found in the last {days_back} days. Try writing a new entry first!")
                else:
                    summary = st.session_state.coach.generate_weekly_report(analysis_df)
                    
                    st.markdown("### Your Personal Insights")
                    st.markdown(f"**Analyzing {len(analysis_df)} entries from the past {days_back} days...**")
                    st.markdown(summary)