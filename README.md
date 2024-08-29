# Car Price Prediction

## Project Overview

This project is a machine learning application designed to predict the price of used cars based on various features such as vehicle age, mileage, engine specifications, and more. The project leverages data preprocessing, feature engineering, and model training to provide accurate price predictions.

## Features

- **Data Ingestion**: Loads raw data from CSV files and splits it into training and test datasets.
- **Data Transformation**: Handles preprocessing of numerical and categorical features, including label encoding for high-cardinality features and one-hot encoding for other categorical features.
- **Model Training**: Trains a machine learning model to predict car prices based on preprocessed features.
- **Prediction Pipeline**: Provides a prediction service that accepts user input, preprocesses it, and outputs the predicted car price.
- **Streamlit Web Application**: A user-friendly web interface to interact with the model and view predictions.

## Project Structure
CarPricePrediction/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── prediction.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── artifact/
│   ├── model.pkl
│   ├── preprocessing.pkl
│   └── label_encoders.pkl
│
├── Notebook/
│   └── cars_data.csv
│
├── app.py
├── README.md
└── requirements.txt


## Installation

### Prerequisites

Ensure you have Python 3.7+ installed. You can check your Python version by running:

```bash
python --version
##Setup
git clone https://github.com/yourusername/CarPricePrediction.git
cd CarPricePrediction
##create venv
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
##Install requirements
pip install -r requirements.txt
```


### Features

- **Input Form**: Enter details about the car such as vehicle age, mileage, engine capacity, etc.
- **Prediction**: View the predicted price for the entered car details.
- **Customization**: Provide your OpenAI API key for additional insights or responses.

### Example Usage

1. Enter the car details into the form on the Streamlit app.
2. Click the "Predict Price" button.
3. View the predicted price highlighted with the rupee sign.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. **Create a new branch**: `git checkout -b feature-branch`.
3. **Make your changes**.
4. **Commit your changes**: `git commit -am 'Add new feature'`.
5. **Push to the branch**: `git push origin feature-branch`.
6. **Create a new Pull Request**.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact:

- **Name**: Your Name
- **Email**: your.email@example.com
