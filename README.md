# 🔐 AI-Based Network Intrusion Detection System

## 📌 Project Description
This project implements an **AI-Based Network Intrusion Detection System (NIDS)** using **Machine Learning (Random Forest Algorithm)**.  
The system analyzes network traffic data and classifies it as **Normal** or **Intrusion (Attack)**.  
A **Streamlit-based web interface** is used to provide an interactive dashboard.

This project is developed as a **VOIS AICTE Intership Project**.

---

## 🎯 Objectives
- To detect network intrusions automatically
- To improve cybersecurity using machine learning
- To reduce manual monitoring of network traffic
- To provide real-time intrusion detection

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Machine Learning Algorithm:** Random Forest  
- **Web Framework:** Streamlit  
- **Libraries Used:**  
  - pandas  
  - numpy  
  - scikit-learn  
  - streamlit  

---

## 📂 Project Structure

AI_NIDS_Project/

├── nids_main.py # Main Streamlit application

├── Train_data.csv # Training dataset with labels

├── Test_data.csv # Test dataset (optional for prediction)

├── requirements.txt # Required Python libraries

└── README.md # Project documentation


---

## 📊 Dataset Information
- **Source:** Kaggle – Network Intrusion Detection Dataset  
- **Format:** CSV  
- **Type:** Network traffic data  
- **Labels:** Normal / Attack  

> The training dataset contains labeled data for supervised learning.  
> Test dataset may be unlabeled and is used for live predictions.


---

## ⚙️ Installation & Setup

### 1️⃣ Install Python
Download and install **Python 3.8 or above**:  
https://www.python.org/downloads/

> Ensure **“Add Python to PATH”** is selected during installation.


---

### 2️⃣ Install Required Libraries
Open terminal/command prompt in the project folder and run:


pip install pandas numpy scikit-learn streamlit


---

3️⃣ Run the Project

Execute the following command in terminal:

python -m streamlit run nids_main.py


---


4️⃣ Open in Browser

The web application will open automatically, or visit:

http://localhost:8501


---



🖥️ How the System Works

1)Load and preprocess the training dataset (Train_data.csv).

2)Convert labels into numeric format (Normal = 0, Attack = 1).

3)Train a Random Forest classifier on the dataset.

4)Predict network traffic as Normal or Intrusion.

5)Display results interactively on the Streamlit dashboard.


---



🚦 Live Intrusion Detection

1)Users can input network traffic values manually via the dashboard.

2)The model predicts traffic status (Normal / Intrusion) in real-time.

3)Visual feedback is given instantly: ✅ Normal Traffic, 🚨 Intrusion Detected.



---



📈 Features

1)Real-time network traffic analysis

2)Automated detection of intrusions using ML

3)Interactive Streamlit dashboard

4)Easy-to-use interface for manual testing

5)Lightweight and fast for laptop execution


---


🚀 Deployment Options

1)Local Deployment: Run on your PC using Streamlit (localhost)

2)Online Deployment (Optional): Deploy on Streamlit Cloud using GitHub repository

3)Executable (Optional): Convert to .exe for standalone desktop usage (advanced)


---


🧪 Future Enhancements

1)Implement Deep Learning models (LSTM, CNN) for improved accuracy.

2)Add real-time packet capture using libraries like scapy.

3)Multi-class classification for different types of attacks.

4)Email/SMS alert notifications for detected intrusions.

5)Cloud-based deployment for enterprise networks.


---


👨‍🎓 Author

Name: Roshan Patil

Project Type: VOIS AICTE Intership Project

Domain: Cyber Security & Machine Learning


---

📜 License

This project is created for educational purposes only and is not intended for commercial use.


---


✅ References

1)Kaggle Network Intrusion Detection Dataset: https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection

2)Scikit-Learn Documentation: https://scikit-learn.org/

3)Streamlit Documentation: https://docs.streamlit.io/


---
