
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from utils.data_processing import (
    load_and_clean_data,
    apply_filters,
    make_recommendation
)

# Page configuration
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Dashboard title
st.title("🎓 Student Performance Dashboard")
st.write("Monitor student academic performance and identify students who need support.")

# Load dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"

try:
    df = load_and_clean_data(DATA_PATH)

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())

selected_gender = st.sidebar.selectbox(
    "Select Gender",
    gender_options
)

performance_options = [
    "All",
    "Excellent",
    "Good",
    "Needs Improvement"
]

selected_performance = st.sidebar.selectbox(
    "Select Performance",
    performance_options
)

# Apply filters
filtered_df = apply_filters(
    df,
    gender=selected_gender,
    performance=selected_performance
)

# Dashboard metrics
st.subheader("📊 Overall Student Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Students",
    len(filtered_df)
)

col2.metric(
    "Average Grade",
    f"{filtered_df['Final_Grade'].mean():.2f}" if not filtered_df.empty else "N/A"
)

col3.metric(
    "Average Attendance",
    f"{filtered_df['Attendance'].mean():.1f}%" if not filtered_df.empty else "N/A"
)

col4.metric(
    "Passing Students",
    len(filtered_df[filtered_df["Final_Grade"] >= 75])
)

st.divider()

if filtered_df.empty:
    st.warning("No student records match your selected filters.")
    st.stop()

# Student performance table
st.subheader("📋 Student Records")

display_df = filtered_df.copy()

display_df["Recommendation"] = display_df.apply(
    make_recommendation,
    axis=1
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# Charts
st.subheader("📈 Performance Analytics")

col1, col2 = st.columns(2)

with col1:
    grade_chart = px.bar(
        filtered_df,
        x="Student_Name",
        y="Final_Grade",
        color="Final_Grade",
        title="Student Final Grades",
        labels={
            "Student_Name": "Student",
            "Final_Grade": "Final Grade"
        }
    )

    grade_chart.add_hline(
        y=75,
        line_dash="dash",
        annotation_text="Passing Grade"
    )

    st.plotly_chart(grade_chart, use_container_width=True)

with col2:
    performance_counts = (
        filtered_df["Performance"]
        .value_counts()
        .reset_index()
    )

    performance_counts.columns = [
        "Performance",
        "Number of Students"
    ]

    performance_chart = px.pie(
        performance_counts,
        names="Performance",
        values="Number of Students",
        title="Student Performance Distribution",
        hole=0.4
    )

    st.plotly_chart(
        performance_chart,
        use_container_width=True
    )

# Attendance analysis
st.subheader("📚 Attendance Analysis")

attendance_chart = px.scatter(
    filtered_df,
    x="Attendance",
    y="Final_Grade",
    color="Performance",
    hover_name="Student_Name",
    title="Relationship Between Attendance and Final Grades",
    labels={
        "Attendance": "Attendance (%)",
        "Final_Grade": "Final Grade"
    }
)

st.plotly_chart(
    attendance_chart,
    use_container_width=True
)

# Recommendations
st.divider()
st.subheader("💡 Student Recommendations")

for _, row in filtered_df.iterrows():
    recommendation = make_recommendation(row)

    if recommendation == "Needs Improvement":
        st.warning(
            f"**{row['Student_Name']}**: {recommendation}"
        )

    elif recommendation == "Good Performance":
        st.info(
            f"**{row['Student_Name']}**: {recommendation}"
        )

    else:
        st.success(
            f"**{row['Student_Name']}**: {recommendation}"
        )

# Download filtered records
csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Student Records",
    data=csv_data,
    file_name="student_performance_report.csv",
    mime="text/csv"
)

st.divider()

st.caption("Student Performance Dashboard | Developed using Python and Streamlit")
