
# Analyza: Data Analysis & Visualization Tool📊🔎

A powerful web-based platform that simplifies complex data analysis for everyone - from students to professionals


Experience the Live Demo Now - 
https://analyza-dashboard.onrender.com/


## About the Project⭐ 

Analyza is an intuitive data analysis and visualization tool designed to bridge the gap between raw data and actionable insights. Our platform combines:

- LLM-powered descriptive analysis (using Google's Gemini API)

- Machine learning-based predictive analysis (9+ algorithms)

- Comprehensive diagnostic tools (correlation, anomaly detection, Root cause analysis)

- Customizable visualizations (7 chart types)

### Key Features:

#### Interactive Dashboards ✨
Dynamic, customizable visualizations including bar charts, histograms, scatter plots, and heatmaps. Users can filter, drill down, and compare different data points interactively.

#### Machine Learning Integration 🤖 
Incorporates descriptive, predictive, and diagnostic analytics using machine learning models:

#### Descriptive Analysis 📈
LLM-powered summaries and insights
#### Predictive Analysis 📊
Linear Regression, Logistic Regression, Random Forest, Polynomial Regression & Boosting Models
#### Diagnostic Analysis 📉
Correlation analysis, anomaly detection, root cause analysis
#### Custom Visualization 💹
Generate tailored visualizations based on your data with options for: Bar Charts, Scatter Plots, Line Graphs, Histograms, Pie Charts, Box Plots, Violin Plots, Data Pre-processing

#### Automated handling of:

- Missing values 
- Outlier detection and correction
- Data standardization
- Categorical variable encoding

#### User-Friendly Interface
Intuitive, guided workflow with interactive tooltips, predefined templates, and AI-powered recommendations to assist users at every stage of data exploration.

### 🛠️ Tech Stack
#### Frontend:

- React.js - For dynamic, responsive UI
- Vite - Next generation frontend tooling
- Tailwind CSS - Utility-first CSS framework

#### Backend:

- FastAPI - High-performance Python backend
- Uvicorn - ASGI server implementation

#### Machine Learning
- Scikit-learn - Machine learning models 
- Pandas/Numpy - Data processing

#### Visualization
- Matplotlib - Comprehensive library for creating static, animated, and interactive visualizations
- Seaborn- Statistical data visualization library

#### AI Integration:

- Google Gemini 2.0 Flash API - Large Language Model for descriptive analysis and natural language insights

### Dataset Requirements
For optimal performance with Analyza, datasets should meet the following criteria:

- Structured data in CSV format (comma-separated values)
- Clear column headers in the first row
- Consistent data types within each column
- File size under 100MB (for web browser processing)
- For ML analysis: At least 50-100 records recommended for meaningful results

### Target Users
#### Businesses
 Businesses can leverage these tools to gain critical insights into market trends, un
derstand customer behavior more deeply, and significantly improve operational efficiency. By analyzing data patterns, companies can make informed decisions that
 drive growth and improve their competitive edge.
#### Researchers
 Researchers benefit from the ability to analyze large datasets, identify complex
 patterns, and validate hypotheses more effectively. These capabilities accelerate the
 pace of discovery and enable groundbreaking research across various fields.
#### Financial Experts
 Financial experts can visualize financial data, assess risks, and predict market trends
 with greater accuracy. These tools provide the insights needed to make informed
 investment decisions and manage financial resources effectively.
#### Students and Educators
 Students and educators find vaulue in facilitating academic research, tracking stu
dent performance, and enhancing learning outcomes. These resources support a
 more engaging and effective educational environment.
#### Healthcare Professionals
 Healthcare professionals can monitor patient data, predict health trends, and opti
mize hospital resources, leading to improved patient care and more efficient health
care delivery.
#### Government Agencies
 Government agencies can analyze public data, improve policy decisions, track eco
nomic trends, and enhance governance. By leveraging data-driven insights, agencies
 can make informed decisions that benefit society as a whole.
## 🚀 To Run The Application Locally

### Prerequisites
- Node.js (v16+)
- Python (3.8+)
- Google Gemini API key (for LLM features)

#### Clone the repository

```bash
git clone https://github.com/shru2811/Analyza
cd Analyza
```
#### Setup Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
#### Important Links replacement
- Replace all instances of https://analyza-dashboard.onrender.com with http://localhost:5173 (or any port where your frontend works)

- Replace all instances of https://analyza-server.onrender.com with http://localhost:8080 

#### Setup Frontend
```bash
cd ../Frontend/Dashboard
npm install
```
#### Configure environment
Create .env files in both frontend directory with your API key of Gemini 2.0 flash.

### Running the Application

#### Start backend server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

#### Start running Frontend
```bash
npm run dev
```
#### Access the application
Open http://localhost:5173 in your browser (port 5173 is the default port of Vite)
## Documentation
Refer "Documentation" Tab of the Website for detailed Documentation
[Link](https://analyza-dashboard.onrender.com/)


## 🔗 Project Team
- [Dhuruv Kumar](https://github.com/dhuruv3421)
- [Khushi Chauhan](https://github.com/Khushi20Chauhan)
- [Shruti Srivastava (me)](https://github.com/shru2811)

