# Stock Price Prediction using LSTM and RNN

This project aims to predict the future stock prices using deep learning techniques such as *Long Short-Term Memory (LSTM)* and *Recurrent Neural Networks (RNN)*. The model is trained on historical stock data, and it predicts future trends based on past price movements.

## *Table of Contents*
- [Introduction](#introduction)
- [Objective](#objective)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Evaluation](#results-and-evaluation)
- [Challenges](#challenges)
- [Future Enhancements](#future-enhancements)

## *Introduction*
The project uses historical stock market data to train a machine learning model capable of predicting future stock prices. The project demonstrates the application of LSTM and RNN, which are commonly used for time-series forecasting. The model predicts stock price trends, which can assist investors in making more informed decisions.

## *Objective*
- Build a predictive model to forecast stock prices using historical data.
- Showcase how deep learning techniques can be applied to financial data.
- Provide a tool for stock market enthusiasts, analysts, and investors to predict stock trends.

## *Technologies Used*
- *Python* (Programming Language)
- *Keras* (For building deep learning models)
- *TensorFlow* (For model training)
- *Pandas* (For data manipulation)
- *NumPy* (For numerical operations)
- *Matplotlib* (For data visualization)
- *Scikit-learn* (For data preprocessing)

## *Dataset*
The dataset used in this project contains historical stock data, including features like:
- *Open Price*
- *Close Price*
- *High Price*
- *Low Price*
- *Volume*

The data can be downloaded from *Yahoo Finance, **Alpha Vantage*, or any other stock market API.

## *Model Architecture*
The model is based on two key deep learning architectures:
1. *Recurrent Neural Network (RNN):* A simple RNN used for time-series prediction.
2. *Long Short-Term Memory (LSTM):* A more complex version of RNN designed to overcome the vanishing gradient problem and learn long-term dependencies in sequential data.

### *Steps:*
1. *Data Preprocessing:* Cleaning the data, removing missing values, and normalizing the dataset.
2. *Model Building:* Defining the LSTM and RNN models using Keras.
3. *Training the Model:* Splitting the data into training and test sets, and training the models.
4. *Model Evaluation:* Evaluating model performance using RMSE (Root Mean Squared Error).
5. *Prediction:* Making predictions on future stock prices and visualizing the results.

## *Installation*

To run the project on your local machine, follow these steps:

### Clone the repository:
```bash
git clone https://github.com/your-username/stock-price-prediction.git
```
###  Install dependencies:
```bash
pip install -r requirements.txt
```
### Requirements.txt:
```makefile
tensorflow==2.x
keras==2.x
pandas==1.3.x
numpy==1.19.x
scikit-learn==0.24.x
matplotlib==3.x
```
## *Usage*
1. Download the historical stock data and place it in the data/ folder.
2. Run the script to train the model:
```bash
python train_model.py
```
3. After training, the model will generate predictions and display the results using Matplotlib.
You can also adjust the train_model.py script to work with your own dataset or modify the model’s hyperparameters to improve accuracy.

## *Results and Evaluation*
The model's performance is evaluated using the Root Mean Squared Error (RMSE) metric. The predictions are visualized by comparing the predicted stock prices to the actual stock prices in a plot.

Example:
- RMSE: 0.25 (indicative of model performance)

## *Challenges*
- Stock market data is highly volatile and affected by many external factors (e.g., economic events, news, etc.), which makes it difficult to predict accurately.
- Fine-tuning the model to reduce overfitting and improving its ability to generalize to new data.
## *Future Enhancements*
- Integrating additional features such as news sentiment analysis or technical indicators (e.g., moving averages) to improve the model’s predictive power.
- Expanding the model to predict stock prices for multiple companies simultaneously.
- Adding an interactive dashboard to visualize predictions in real-time.
