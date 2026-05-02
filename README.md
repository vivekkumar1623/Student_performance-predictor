# Student_performance-predictor
Student Performance Predictor is a machine learning project that analyzes student data (attendance, study time, past scores) to predict academic outcomes. It identifies key performance factors and enables early intervention using models like Random Forest and Decision Trees for accurate, data-driven insights.
🎓 Student Performance Predictor
A machine learning web application built with Streamlit that predicts whether a student will Pass or Fail based on study habits, attendance, previous scores, and background information.

📌 Project Overview
ItemDetailsAlgorithmRandom Forest ClassifierModel Accuracy≥ 70%FrameworkStreamlitLanguagePython 3.xWeekWeek 10 – Final Deployment

🚀 Features

📊 Interactive Dashboard – Visualize dataset statistics and distributions
🔍 Live Prediction – Enter student details in the sidebar and get instant Pass/Fail prediction with confidence score
📈 Feature Importance – See which factors most influence performance
🌳 Random Forest Model – Ensemble learning for high accuracy


🧠 Input Features
FeatureDescriptionStudy Hours Per DayDaily hours spent studying (1–12)Attendance (%)Percentage of classes attendedPrevious ScoreMarks obtained in previous examSleep HoursDaily hours of sleepParental EducationEducation level of parentsInternet AccessWhether student has internet at home
Target Variable: Result → 1 = Pass, 0 = Fail

🛠️ Tech Stack

Python – Core language
Streamlit – Web app framework
scikit-learn – Machine learning (Random Forest, train/test split, accuracy score)
Pandas & NumPy – Data manipulation
Matplotlib & Seaborn – Data visualization


⚙️ How to Run Locally
bash# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
cd student-performance-predictor

# 2. Install dependencies
pip install streamlit scikit-learn pandas numpy matplotlib seaborn

# 3. Run the app
streamlit run app.py
Then open your browser at http://localhost:8501

📂 Project Structure
student-performance-predictor/
│
├── app.py          # Main Streamlit application
├── README.md       # Project documentation

📊 Model Performance

Algorithm: Random Forest Classifier (100 estimators)
Train/Test Split: 80% / 20%
Documented Accuracy: ≥ 70% ✅


🖼️ App Preview
The app includes three tabs:

Dataset Preview – Shows sample data used for training
Visualizations – Pass/Fail pie chart and scatter plots
Feature Importance – Bar chart of top contributing features


👨‍💻 Author
Vivek Kumar
B.Tech / BCA – Lovely professional University
Week 10 Lab Assignment – Student Performance Predictor
