# 💼 AI-Powered Employee Salary Prediction

This AI web application predicts employee salaries based on key professional attributes. Developed during a 6-week virtual internship with **Edunet Foundation** through **IBM SkillsBuild** (AICTE internship portal).

## ✨ Features
- **Real-time Salary Estimation**: Get instant predictions based on user inputs
- **High-Accuracy ML Model**: 
  - R² Score: 0.91 
  - RMSE: ₹15,616.67
- **Intuitive Interface**: Clean Streamlit UI for seamless experience
- **Data-Driven Insights**: Trained on real-world Kaggle dataset

## ⚙️ How It Works
1. **Input Features**:
   - Age, Gender, Education Level
   - Job Title, Years of Experience
2. **ML Processing**:
   - Inputs processed through trained pipeline (`salary_prediction_pipeline.pkl`)
3. **Output**:
   - Estimated salary in INR (₹)

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 243,880,416.72 |
| Root Mean Squared Error (RMSE) | ₹15,616.67 |
| R-squared (R² Score) | 0.91 |

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python 3.10+
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib
- **Model Deployment**: Joblib

## 🚀 Installation
```bash
git clone https://github.com/<your-github-username>/Employee-Salary-Prediction-Rachel-Arora.git
cd Employee-Salary-Prediction-Rachel-Arora
pip install -r requirements.txt
streamlit run app.py
```
## 🔮 Future Enhancements
- **🌐 Multilingual interface support**: Expand accessibility with language options
- **📊 Interactive salary visualizations**: Dynamic charts showing prediction factors
- **🤖 SHAP/LIME explainable AI components**: Transparent model decision explanations
- **🏢 Industry/location-based features**: Enhanced accuracy with geographic/job sector data
- **💰 Salary range prediction module**: Transition from point estimates to range forecasts

## 🌐 Live Deployment
[![Live App](https://img.shields.io/badge/🚀_Live_App-Click_Here-FF4B4B?style=for-the-badge)](https://smartsalarypredictor.streamlit.app/))

## 🙏 Acknowledgments
- Dataset source: [Kaggle](https://www.kaggle.com/)
- IBM SkillsBuild & Edunet Foundation
- AICTE Internship Portal
🧑‍💻 **Created by**:Aman Prajapati
💌 **Connect**:  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-amanprajapati-%230A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/amanprajapati004/)
