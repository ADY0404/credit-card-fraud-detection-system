 # credit-card-fraud-detection-system

This project is a Credit Card Fraud Detection System targeted at banks, e-commerce sites, and any firm that accepts payment through credit cards. The system utilizes a pre-trained RandomForestClassifier model to detect fraudulent transactions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Installation

### Step 1: Clone the repository

git clone https://github.com/ADY0404/credit-card-fraud-detection-system.git
cd credit-card-fraud-detection-system

### step 2: Create a virtual environment 
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`

### Step 3: Install dependencies
pip install -r requirements.txt

## Usage
streamlit run app.py

## Instructions
1. Upload your CSV file using the sidebar.
2. Adjust the sliders to filter transactions based on amount.
3. Click on the Predict button to see the fraud prediction results.
4. Download the detailed results using the provided buttons.

## Features
1. Upload CSV files with transaction data.
2 Filter transactions based on the amount.
3. Predict fraudulent transactions using a pre-trained RandomForestClassifier.
4. Visualize and download prediction results.

## Contributing
1. Fork the repository.
2. Create a new branch (git checkout -b feature/feature-name).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/feature-name).
5. Open a pull request.

## Contact
- For any questions or support, please contact dicksonadddai@gmail.com.

## Acknowledgements
-Special thanks to all contributors and open-source projects that made this project possible.