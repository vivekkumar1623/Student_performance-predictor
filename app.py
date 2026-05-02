import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.markdown("Predict whether a student will **Pass or Fail** based on study habits and background.")

# ─── Generate Synthetic Dataset ─────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500

    study_hours     = np.random.randint(1, 10, n)
    attendance      = np.random.randint(50, 100, n)
    prev_score      = np.random.randint(30, 100, n)
    sleep_hours     = np.random.randint(4, 10, n)
    parental_edu    = np.random.choice(['None', 'High School', 'Bachelor', 'Master'], n)
    internet_access = np.random.choice(['Yes', 'No'], n)

    # Rule-based label with noise for realism
    score = (study_hours * 4 + attendance * 0.3 + prev_score * 0.5 + sleep_hours * 2)
    label = (score > np.percentile(score, 40)).astype(int)  # ~60% pass rate

    df = pd.DataFrame({
        'Study_Hours_Per_Day': study_hours,
        'Attendance_Percent':  attendance,
        'Previous_Score':      prev_score,
        'Sleep_Hours':         sleep_hours,
        'Parental_Education':  parental_edu,
        'Internet_Access':     internet_access,
        'Result':              label   # 1 = Pass, 0 = Fail
    })
    return df

@st.cache_resource
def train_model(df):
    le_edu      = LabelEncoder()
    le_internet = LabelEncoder()
    df = df.copy()
    df['Parental_Education'] = le_edu.fit_transform(df['Parental_Education'])
    df['Internet_Access']    = le_internet.fit_transform(df['Internet_Access'])

    X = df.drop('Result', axis=1)
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, le_edu, le_internet, X.columns.tolist()

df                               = load_data()
model, accuracy, le_edu, le_net, feature_names = train_model(df)

# ─── Sidebar: Prediction Inputs ─────────────────────────────────────────────
st.sidebar.header("📋 Enter Student Details")

study_hours     = st.sidebar.slider("Study Hours Per Day",  1, 12, 5)
attendance      = st.sidebar.slider("Attendance (%)",       50, 100, 75)
prev_score      = st.sidebar.slider("Previous Score",       30, 100, 65)
sleep_hours     = st.sidebar.slider("Sleep Hours",          4, 10, 7)
parental_edu    = st.sidebar.selectbox("Parental Education", ['None', 'High School', 'Bachelor', 'Master'])
internet_access = st.sidebar.selectbox("Internet Access",   ['Yes', 'No'])

if st.sidebar.button("🔍 Predict"):
    edu_enc = le_edu.transform([parental_edu])[0]
    net_enc = le_net.transform([internet_access])[0]

    input_data = pd.DataFrame([[
        study_hours, attendance, prev_score,
        sleep_hours, edu_enc, net_enc
    ]], columns=feature_names)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.sidebar.markdown("---")
    if prediction == 1:
        st.sidebar.success(f"✅ **PASS**  ({probability[1]*100:.1f}% confidence)")
    else:
        st.sidebar.error(f"❌ **FAIL**  ({probability[0]*100:.1f}% confidence)")

# ─── Main Dashboard ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("📊 Model Accuracy", f"{accuracy*100:.2f}%")
col2.metric("📁 Dataset Size",   f"{len(df)} students")
col3.metric("🌳 Algorithm",      "Random Forest")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Dataset Preview", "📈 Visualizations", "🔬 Feature Importance"])

with tab1:
    st.subheader("Sample Dataset")
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Pass vs Fail Distribution")
        fig1, ax1 = plt.subplots()
        counts = df['Result'].value_counts()
        ax1.pie(counts, labels=['Pass', 'Fail'], autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'], startangle=90)
        st.pyplot(fig1)

    with col_b:
        st.subheader("Study Hours vs Previous Score")
        fig2, ax2 = plt.subplots()
        colors = df['Result'].map({1: '#4CAF50', 0: '#F44336'})
        ax2.scatter(df['Study_Hours_Per_Day'], df['Previous_Score'],
                    c=colors, alpha=0.5)
        ax2.set_xlabel("Study Hours Per Day")
        ax2.set_ylabel("Previous Score")
        st.pyplot(fig2)

with tab3:
    st.subheader("Feature Importance (Random Forest)")
    importance_df = pd.DataFrame({
        'Feature':   feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=importance_df, x='Importance', y='Feature',
                palette='viridis', ax=ax3)
    ax3.set_title("Which features matter most?")
    st.pyplot(fig3)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit · Random Forest Classifier · scikit-learn")
