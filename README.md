# Numera App

## Overview
Numera App is a Django-based application designed for statistical data analysis and visualization. Users can upload data files, perform various statistical operations, and view interactive visualizations. The application includes session-based authentication for user management without requiring a database.

---

## Features

- **Session-based Authentication**: Secure login and session management.
- **Data Upload**: Support for uploading Excel and CSV files.
- **Statistical Operations**: Perform operations such as mean, median, mode, variance, and standard deviation.
- **Data Visualization**: Generate interactive charts like bar plots, pie charts, and line graphs.

---

## Installation

### Prerequisites

1. Python 3.9+
2. Django 4.0+
3. pip (Python package installer)
4. Virtual environment tool (e.g., `venv` or `virtualenv`)

### Steps

1. Clone the Repository:
   ```bash
   git clone https://github.com/chaaamah/Numera-Data-Analysis-App.git
   cd Numera-Data-Analysis-App
   ```

2. Set up a Virtual Environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Development Server:
   ```bash
   python manage.py runserver
   ```
   Access the application at `http://127.0.0.1:8000`.

---

## Usage

1. Register or log in to your account.
2. Upload a data file (Excel or CSV).
3. Select a statistical operation to perform.
4. View the results and generated visualizations.

---

## Dependencies

- Django
- Pandas (for data processing)
- Matplotlib / Plotly (for visualization)
- OpenPyXL (for Excel file support)
- Scipy

---

## Contribution

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes and push:
   ```bash
   git commit -m "Add your message"
   git push origin feature-name
   ```
4. Create a pull request.

---
