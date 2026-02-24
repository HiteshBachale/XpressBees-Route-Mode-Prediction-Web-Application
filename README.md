# XpressBees Route Mode Prediction Web Application

## 📌 Project Overview

This project is a **Flask-based web application** designed to predict the **optimal shipment route mode (Air or Surface)** for logistics operations. The application uses shipment attributes loaded dynamically from an **Excel dataset** and applies a **Machine Learning–ready architecture** (currently rule-based, easily extendable to ML) to provide routing recommendations.

The system integrates:

* **Python (Flask, Pandas)** for backend processing
* **HTML, CSS, Jinja2** for frontend rendering
* **Excel (.xlsx)** as the data source

This repository is suitable for **academic projects, internships, and GitHub portfolios**.

---

## 🎯 Objectives

* Automate shipment route mode selection
* Reduce manual and heuristic-based decisions
* Demonstrate backend–frontend integration
* Provide a scalable foundation for Machine Learning deployment

---

## 🧩 Features

* Dynamic dropdowns populated from Excel data
* Date, numeric, and categorical input handling
* Route mode prediction with explanation
* Responsive and clean UI
* Easily extensible to Machine Learning models

---

## 🗂️ Project Structure

```
XpressBees-Route-Prediction/
│
├── app.py                     # Flask backend application
├── templates/
│   └── index.html             # Frontend UI (HTML + CSS + Jinja2)
├── XpressBees.xlsx            # Shipment dataset (Excel file)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Technology Stack

| Layer            | Technology                  |
| ---------------- | --------------------------- |
| Backend          | Python, Flask               |
| Data Handling    | Pandas, OpenPyXL            |
| Frontend         | HTML, CSS, Jinja2           |
| Deployment Ready | Gunicorn, Docker (optional) |

---

## 📊 Dataset Description

The application reads shipment data from an Excel file. The following attributes are used to populate the UI and assist prediction:

* Ship Pin Code
* Inscan and Bag Scan Dates
* Origin and Destination Hub Details
* Shipment Status
* Physical Weight
* Volumetric Weight
* Lane Information

> ⚠️ The Excel file path must be correctly configured in `app.py`.

---

## 🧠 Prediction Logic

Currently, the application uses a **rule-based decision engine** to determine the route mode:

### Rule Highlights

* **Urgent deliveries (≤ 2 days)** → Air
* **High volumetric vs physical weight** → Air
* **Very heavy shipments (≥ 50 kg)** → Surface
* **Metro / ROI lanes** → Air
* **Default case** → Surface

Each prediction is accompanied by a **human-readable explanation**.

> 🔁 This logic can be replaced with a trained **Machine Learning model** without changing the UI.

---

## 🖥️ Web Application Workflow

```
User Input → HTML Form → Flask Backend → Prediction Logic → Result Display
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone <your-github-repo-url>
cd XpressBees-Route-Prediction
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install flask pandas openpyxl
```

### 4️⃣ Configure Excel Path

Edit `app.py` and update:

```python
EXCEL_PATH = "path/to/XpressBees.xlsx"
```

### 5️⃣ Run the Application

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

---

## 🌐 Frontend Details (`index.html`)

* Responsive grid layout (3 → 2 → 1 columns)
* Dynamic form generation using Jinja2
* Date picker for date fields
* Number input for weight fields
* Dropdowns for categorical data
* Styled using pure CSS (no external libraries)

---

## ☁️ Deployment (Production-Ready)

### Using Gunicorn

```bash
gunicorn app:app
```

### Render Deployment

1. Push code to GitHub
2. Create a new Web Service on Render
3. Build Command:

```bash
pip install -r requirements.txt
```

4. Start Command:

```bash
gunicorn app:app
```

### AWS EC2 (Optional)

* Launch EC2 instance
* Install Python & dependencies
* Run using Gunicorn or Docker

---

## 🔮 Future Enhancements

* Replace rule-based logic with Machine Learning model
* Add model evaluation metrics
* Store prediction history
* Add authentication and role management
* Deploy with Docker and CI/CD

---

## 👨‍💻 Author

**Hitesh Bachale**
Project created for academic and learning purposes.

---

## 📄 License

This project is intended for **educational and demonstration use**. You are free to modify and extend it.
