# 🛡️ Insurance Claim Fraud Analysis System (UC044)

An enterprise-grade AI-powered insurance claims investigation solution that combines **Machine Learning, Computer Vision, and Intelligent Workflow Orchestration** to assess claim legitimacy, detect potential fraud, analyze vehicle damage, and generate investigation recommendations. The system helps insurance investigators make faster, data-driven decisions through an interactive Streamlit-based interface.

---

# 📑 Table of Contents

- Overview
- Business Problem
- Architecture Overview
- Solution Workflow
- Key Features
- Technology Stack
- Project Structure
- Getting Started
- Datasets
- Security & Compliance
- Future Enhancements
- Expected Business Outcomes

---

# 📖 Overview

The Insurance Claim Fraud Analysis System automates insurance claim investigation by combining structured claim analysis, vehicle damage assessment, fraud prediction, and risk evaluation into a unified workflow.

Using a LangGraph-powered multi-agent architecture, the system processes claim information, evaluates uploaded vehicle images, predicts fraud risk, and generates investigation recommendations to support claim reviewers.

---

# 💼 Business Problem

Insurance companies process thousands of claims every day. Manual claim assessment is time-consuming, expensive, and prone to inconsistencies.

Fraudulent, exaggerated, or suspicious claims can result in significant financial losses and increased operational costs.

This solution aims to:

- Reduce manual claim review effort
- Flag potentially fraudulent claims early
- Automate damage assessment from vehicle images
- Improve investigation efficiency
- Provide evidence-backed recommendations to claim handlers

---

# 🏗️ Architecture Overview

The system uses a multi-agent workflow orchestrated through **LangGraph**, with all major processing performed locally.

## Core Agents

### 1️⃣ Claim Intake Agent

Responsible for:

- Collecting claim information
- Parsing structured inputs
- Validating mandatory fields
- Preparing claim data for analysis

### 2️⃣ Fraud Prediction Agent

Uses a trained **Random Forest Machine Learning Model** to:

- Analyze structured claim attributes
- Predict fraud probability
- Generate fraud risk scores
- Identify suspicious claim patterns

### 3️⃣ Damage Assessment Agent

Powered by **YOLOv8** and **OpenCV**.

Responsibilities:

- Vehicle image validation
- Damage detection
- Damage localization
- Severity estimation
- Visual evidence generation

### 4️⃣ Risk Assessment Agent

Combines outputs from multiple agents and generates:

- Overall risk score
- Investigation priority
- Claim risk classification
- Decision-support metrics

### 5️⃣ Investigation Recommendation Agent

Responsible for:

- Summarizing findings
- Consolidating evidence
- Generating investigation recommendations
- Supporting manual claim review

---

# ⚙️ Solution Workflow

```text
                Claim Submission
                         │
                         ▼
                 Claim Intake Agent
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
Fraud Prediction Agent       Damage Assessment Agent
 (Random Forest)                   (YOLOv8 + OpenCV)
          │                             │
          └──────────────┬──────────────┘
                         ▼
              Risk Assessment Agent
                         │
                         ▼
       Investigation Recommendation Agent
                         │
                         ▼
          Final Investigation Report
                         │
                         ▼
            Manual Review / Decision
```

---

# ✨ Key Features

### 🔍 Fraud Risk Prediction

- Random Forest-based fraud classification
- Historical claim pattern analysis
- Fraud probability scoring
- High-risk claim identification

### 🚗 Vehicle Damage Assessment

- Vehicle image preprocessing and validation using OpenCV
- Damage detection using YOLOv8
- Damage localization and bounding box identification
- Severity assessment based on detected damage regions
- Visual evidence generation for claim investigation
- Evidence-based claim validation

### 🔄 Multi-Agent Workflow

- LangGraph-powered orchestration
- Modular processing architecture
- Distributed claim analysis workflow
- Automated decision support

### 📊 Interactive Dashboard

- Streamlit-powered interface
- Claim submission forms
- Real-time analysis results
- Fraud risk visualization
- Investigation review screens

### 📄 Investigation Reports

- Automated report generation
- Consolidated fraud assessment
- Damage assessment summaries
- Investigation recommendations
- Downloadable reports

### 👨‍💼 Human-in-the-Loop Review

- Escalation support for suspicious claims
- Evidence-backed decision support
- Manual review recommendations
- Risk-based prioritization

---

# 🛠️ Technology Stack

### Programming Language

- Python 3.12

### Workflow Orchestration

- LangGraph

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy

### Computer Vision

- YOLOv8 (Ultralytics)
- OpenCV

### User Interface

- Streamlit

### Reporting

- ReportLab

---

# 📂 Project Structure

```text
InsuranceClaimFraudAnalysis/
│
├── agents/
│   ├── claim_intake_agent.py
│   ├── damage_assessment_agent.py
│   ├── fraud_prediction_agent.py
│   ├── risk_assessment_agent.py
│   └── investigation_agent.py
│
├── graph/
│   ├── nodes.py
│   ├── state.py
│   └── workflow.py
│
├── models/
│   ├── fraud_model.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── feature_names.json
│   └── model_metrics.json
│
├── pages/
├── utils/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <repository_url>
cd InsuranceClaimFraudAnalysis
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

### Linux / Mac

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Launch Application

```bash
streamlit run app.py
```

---

# 📊 Datasets

This project uses publicly available datasets for model training and evaluation.

### Fraud Dataset

```text
data/raw/fraud_dataset/fraud.csv
```

### Vehicle Damage Dataset

```text
data/raw/vehide/
```

> Public datasets are used for model development and evaluation. Large datasets may be excluded from the repository to reduce repository size.

---
### Model Download

The application uses the pretrained YOLOv8n model.

If `yolov8n.pt` is not present locally, Ultralytics will automatically download it during the first execution.
---
# 🔒 Security & Compliance

### Data Privacy

- No production customer data is stored in the repository.
- Public datasets are used for training and testing.

### Credential Protection

- Environment variables should be managed through a local `.env` file.
- Sensitive credentials must never be committed to source control.

### Recommended Exclusions

```text
.env
.venv/
__pycache__/
logs/
uploads/
```

---

# 🚀 Future Enhancements

- Explainable AI (XAI) visualizations
- Advanced fraud pattern analysis
- Damage cost estimation
- Multi-image claim assessment
- Cloud deployment support
- Integration with insurer claim management systems
- Real-time claim triage

---

# 📈 Expected Business Outcomes

✅ Faster claim investigation

✅ Reduced manual review effort

✅ Improved fraud detection accuracy

✅ Consistent claim assessment

✅ Enhanced investigator productivity

✅ Better operational efficiency

✅ Reduced financial losses from fraudulent claims

---

### Developed for UC044 – Insurance Claim Fraud Analysis System

*Leveraging Machine Learning, Computer Vision, and workflow orchestration to modernize insurance claims investigation and fraud detection.*
