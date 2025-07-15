# MLOps Random Forest Breast Cancer Classifier

A machine learning project for breast cancer classification using the Random Forest algorithm. This project demonstrates data preprocessing, model training, evaluation, and deployment best practices for a binary classification problem in healthcare.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Docker Usage](#docker-usage)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model](#model)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This repository contains code and documentation for building a breast cancer classifier using the Random Forest algorithm. The goal is to predict whether a tumor is malignant or benign based on clinical features. The project follows best practices in data science and DevOps for reproducibility and maintainability.

## Features

- Data preprocessing and cleaning
- Exploratory data analysis (EDA)
- Random Forest model training and hyperparameter tuning
- Model evaluation with metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Model persistence (saving/loading)
- Example API for model inference
- Automated tests and CI/CD integration
- Dockerized application for easy deployment
- Docker Compose support for multi-service orchestration
- Deployment instructions for AWS EC2

## Installation

1. **Clone the repository:**

```
git clone https://github.com/yourusername/ops-random-forest-breast.git
cd ops-random-forest-breast
```

2. Create and activate a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```
pip install -r requirements.txt
```

## Docker Usage

1. **Build the Docker image:**

```
docker build -t breast-cancer-classifier .
```

2. **Run the application using Docker:**

```
docker run -p 8000:8000 breast-cancer-classifier
```

3. **Using Docker Compose (recommended for multi-service setup):**

If you have a `docker-compose.yml` file, start all services with:

```
docker compose up --build
```

This will build and run the API and any other defined services (e.g., database, monitoring).

## AWS EC2 Deployment

- The application can be deployed on AWS EC2 instances for scalable cloud hosting.
- Typical steps:
  1. Launch an EC2 instance (Ubuntu recommended).
  2. Install Docker and Docker Compose on the instance.
  3. Clone this repository to the EC2 instance.
  4. Build and run the application using Docker or Docker Compose as described above.
- Ensure the appropriate security group rules are set to allow inbound traffic on the required ports (e.g., 8000 for the API).

## Usage

1. **Prepare the data:**

- Place your dataset (e.g., `data.csv`) in the `data/` directory.

2. **Train the model:**

```
python src/train.py --data data/data.csv
```

3. **Evaluate the model:**

```
python src/evaluate.py --model models/random_forest.pkl --data data/test.csv
```

4. **Run the API (optional):**

```
uvicorn src.api:app --reload
```

Or, if using Docker:

```
docker run -p 8000:8000 breast-cancer-classifier
```

Or with Docker Compose:

```
docker compose up
```

## Contact

- **LinkedIn:** [https://www.linkedin.com/in/mohammad-rammal](https://www.linkedin.com/in/mohammad-rammal)
- **Email:** mohammad.rammal@hotmail.com
