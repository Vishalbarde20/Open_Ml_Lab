# 🤖 Open_ML_Lab
An interactive web-based **Machine Learning Training Platform** that allows users to train, visualize, and evaluate machine learning models in real-time.  
Built using **Python**, **Streamlit**, and **scikit-learn**, this project helps users understand ML workflows through an intuitive dashboard.

---

## 1. Project Title
**Open ML Lab**

---

## 2. Members Information

List of all group members and their respective contributions are mentioned below:

| Name of Student     | Registration Number | Role / Work Done                                                                                           |
|----------------------|---------------------|-------------------------------------------------------------------------------------------------------------|
| **Om Rakhade**     | 2024BIT503             | Developed backend logic, implemented model training and dataset integration, and managed overall structure. |
| **Vishal Barde**       | 2024BIT508             | Implemented frontend in Streamlit, created real-time visualization modules, and designed interactive UI.    |
| **Ramprasad Londhe** | 2024BIT509              | Worked on UI/UX design, testing, documentation, and visual metrics module integration.                      |
| **All Members**      |                     | Testing, debugging, optimization, documentation, and final presentation.                                    |

---

## 3. Project Description

The **Interactive ML Training Platform** is designed to make machine learning model training accessible and understandable for all users.  
It allows users to choose a dataset, select an ML model (e.g., Decision Tree, Random Forest, SVM), tune hyperparameters, and instantly visualize performance metrics.

Users can:
- Upload custom datasets or use built-in ones.
- Select from multiple ML algorithms.
- Tune parameters in real time.
- View evaluation metrics such as **Accuracy, Precision, Recall, and F1 Score**.
- Visualize results through charts like **Confusion Matrix**, **Feature Importance**, and **Metric Comparisons**.

This platform provides a **hands-on experience** with ML workflows, offering both technical depth and visual clarity for educational and experimental purposes.  

---

## 4. Technologies Used
- **Language:** Python  
- **Framework:** Streamlit  
- **Libraries:** scikit-learn, NumPy, Pandas, Plotly, Matplotlib, Seaborn  
- **Deployment:** Streamlit Cloud / Heroku  
- **Version Control:** Git & GitHub  

---

## 5. Key Features
- 📊 Interactive ML model training and evaluation  
- ⚙️ Real-time hyperparameter tuning  
- 📈 Dynamic visualizations (Confusion Matrix, Feature Importance)  
- 📁 Built-in and custom dataset support  
- 💾 Export trained model feature  
- 🧠 Supports multiple ML algorithms  
- 🌐 Easy cloud deployment  

---

## 6. How to Run the Project
```bash
# Step 1: Create a virtual environment
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the application
streamlit run app.py
