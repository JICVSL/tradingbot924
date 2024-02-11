import tkinter as tk
import yfinance as yf
import pandas as pd
from tkinter import ttk
from tkcalendar import Calendar
import mplfinance as mpf
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import timedelta, datetime

# Constants
SYMBOL = "BTC-USD"
START_DATE_DEFAULT = datetime.now() - timedelta(days=1095)
END_DATE_DEFAULT = datetime.now()
CRYPTO_MAP = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Chia (XCH)": "XCH-USD",
    "Filecoin (FIL)": "FIL-USD",
    "BitTorrent (BTT)": "BTT-USD"
}

class Utility:
    @staticmethod
    def prepare_data(data):
        """
        Preprocesses data for training the model.

        :param data: DataFrame containing cryptocurrency data.
        :return: Scaled data and scaler object.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.filter(['Close']).values)
        return scaled_data, scaler

    @staticmethod
    def create_sequences(data, sequence_length=60):
        """
        Creates input sequences and target values for training the model.

        :param data: Scaled cryptocurrency data.
        :param sequence_length: Length of input sequence.
        :return: Input sequences and target values.
        """
        x_data, y_data = [], []
        for i in range(sequence_length, len(data)):
            x_data.append(data[i - sequence_length:i, 0])
            y_data.append(data[i, 0])
        return np.array(x_data), np.array(y_data)

    @staticmethod
    def build_model(input_shape):
        """
        Builds and compiles the LSTM model using Keras.

        :param input_shape: Shape of input data.
        :return: Compiled LSTM model.
        """
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    @staticmethod
    def train_model(model, x_train, y_train, epochs, batch_size=32):
        """
        Trains the LSTM model.

        :param model: Compiled LSTM model.
        :param x_train: Input data for training.
        :param y_train: Target values for training.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :return: Model training history.
        """
        return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    @staticmethod
    def generate_predictions(model, data, training, scaler):
        """
        Generates predictions using the trained model.

        :param model: Trained LSTM model.
        :param data: Scaled cryptocurrency data.
        :param training: Index of training data.
        :param scaler: Scaler object used for preprocessing.
        :return: Predicted prices.
        """
        test_data = data[training - 60:, :]
        x_test, y_test = Utility.create_sequences(test_data)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predictions = model.predict(x_test)
        return scaler.inverse_transform(predictions)

    @staticmethod
    def plot_results(train, test, predictions):
        """
        Plots the training data, test data, and predictions.

        :param train: Training data.
        :param test: Test data.
        :param predictions: Predicted prices.
        """
        plt.figure(figsize=(10, 8))
        plt.xticks(rotation=45)
        plt.plot(train.index, train['Close'])
        plt.plot(test.index, test[['Close', 'Predictions']])
        plt.title('BTC-USD')
        plt.xlabel('Date')
        plt.ylabel("Close")
        plt.legend(['Train', 'Test', 'Predictions'])
        plt.show()


class CryptocurrencyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cryptocurrency Info")
        self.selected_crypto = tk.StringVar(value="Bitcoin (BTC)")
        self.start_selected = tk.StringVar(value=START_DATE_DEFAULT.strftime("%Y-%m-%d"))
        self.end_selected = tk.StringVar(value=END_DATE_DEFAULT.strftime("%Y-%m-%d"))
        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the user interface.
        """
        self.create_crypto_dropdown()
        self.create_date_selection_widgets()
        self.create_action_buttons()
        self.create_status_labels()

    def create_crypto_dropdown(self):
        """
        Creates the dropdown for selecting cryptocurrencies.
        """
        crypto_dropdown = ttk.Combobox(self, textvariable=self.selected_crypto, values=list(CRYPTO_MAP.keys()))
        crypto_dropdown.pack()

    def create_date_selection_widgets(self):
        """
        Creates widgets for selecting start and end dates.
        """
        self.create_date_selection_label("Select Start Date:")
        self.create_date_button(self.start_selected)

        self.create_date_selection_label("Select End Date:")
        self.create_date_button(self.end_selected)

    def create_date_selection_label(self, text):
        """
        Creates label for date selection.

        :param text: Text for the label.
        """
        start_date_label = tk.Label(self, text=text)
        start_date_label.pack()

    def create_date_button(self, date_variable):
        """
        Creates button for selecting a date.

        :param date_variable: Variable to hold the selected date.
        """
        date_button = tk.Button(self, textvariable=date_variable, command=lambda: self.display_calendar(date_variable))
        date_button.pack()

    def create_action_buttons(self):
        """
        Creates action buttons for various functionalities.
        """
        action_buttons = [
            ("Get Data and Plot", self.get_data_and_plot),
            ("Show Trend", self.show_trend),
            ("Predict Prices (Polynomial)", self.predict_prices_polynomial),
            ("RNN - LSTM network using TensorFlow", self.show_close_price_prediction)
        ]
        for text, command in action_buttons:
            button = tk.Button(self, text=text, command=command)
            button.pack()

        # Add entry field for epochs
        self.epochs_entry_label = tk.Label(self, text="Epochs:")
        self.epochs_entry_label.pack()

        self.epochs_entry = tk.Entry(self)
        self.epochs_entry.pack()

    def create_status_labels(self):
        """
        Creates labels for displaying status messages.
        """
        self.status_label = tk.Label(self, text="")
        self.status_label.pack()

        self.prediction_label = tk.Label(self, text="")
        self.prediction_label.pack()

    def display_calendar(self, selected_date):
        """
        Displays a calendar for selecting dates.

        :param selected_date: Variable to hold the selected date.
        """
        top = tk.Toplevel(self)
        cal = Calendar(top, selectmode='day', date_pattern='Y-m-d')
        cal.pack()

        def on_date_select():
            selected_date.set(cal.get_date())
            top.destroy()

        select_button = tk.Button(top, text="Select", command=on_date_select)
        select_button.pack()
        top.grab_set()

    def get_data_and_plot(self):
        """
        Downloads cryptocurrency data, saves it to a CSV file, and plots candlestick data.
        """
        try:
            selected_crypto = self.selected_crypto.get()
            symbol = CRYPTO_MAP[selected_crypto]
            start_date = self.start_selected.get()
            end_date = self.end_selected.get()
            crypto_data = yf.download(symbol, start=start_date, end=end_date)
            crypto_data.to_csv(f"{symbol}_data.csv")
            self.status_label.config(text=f"CSV data for {symbol} downloaded from {start_date} to {end_date}")
            self.plot_candlestick_data(crypto_data)
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

    def plot_candlestick_data(self, data):
        """
        Plots candlestick chart using mplfinance.

        :param data: Cryptocurrency data.
        """
        try:
            mpf.plot(data, type='candle', title='Cryptocurrency Candlestick Chart', ylabel='Price', style='charles',
                     volume=True)
            plt.show()
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

    def show_trend(self):
        """
        Plots a linear trend line for the cryptocurrency.
        """
        try:
            selected_crypto = self.selected_crypto.get()
            symbol = CRYPTO_MAP[selected_crypto]
            file_path = f"{symbol}_data.csv"
            crypto_data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            coefficients = np.polyfit(np.arange(len(crypto_data)), crypto_data['Close'].values, 1)
            trend_line = np.polyval(coefficients, np.arange(len(crypto_data)))
            plt.plot(crypto_data.index, trend_line, color='orange', linestyle='--', label='Trend Line')
            plt.legend()
            plt.show()
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

    def predict_prices_polynomial(self):
        """
        Predicts prices using a polynomial regression model.
        """
        try:
            selected_crypto = self.selected_crypto.get()
            symbol = CRYPTO_MAP[selected_crypto]
            file_path = f"{symbol}_data.csv"
            crypto_data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
            dates = crypto_data.index.map(datetime.toordinal).values.reshape(-1, 1)
            prices = crypto_data['Close'].values
            model.fit(dates, prices)
            future_date = crypto_data.index[-1] + timedelta(days=7)
            prediction = model.predict([[future_date.toordinal()]])[0]
            last_price = crypto_data['Close'].iloc[-1]
            percent_change = ((prediction - last_price) / last_price) * 100
            self.prediction_label.config(text=f"Predicted price for next week: {prediction:.2f} USD\n"
                                              f"Percentage change from last price: {percent_change:.2f}%")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

    def show_close_price_prediction(self):
        """
        Trains and predicts cryptocurrency prices using an LSTM model.
        """
        try:
            epochs = int(self.epochs_entry.get())
            selected_crypto = self.selected_crypto.get()
            symbol = CRYPTO_MAP[selected_crypto]
            file_path = f"{symbol}_data.csv"
            data = pd.read_csv(file_path, parse_dates=True)
            scaled_data, scaler = Utility.prepare_data(data)
            training = int(np.ceil(len(scaled_data) * .95))
            x_train, y_train = Utility.create_sequences(scaled_data)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            model = Utility.build_model(input_shape=(x_train.shape[1], 1))
            history = Utility.train_model(model, x_train, y_train, epochs=epochs)
            predictions = Utility.generate_predictions(model, scaled_data, training, scaler)
            train = data[:training].copy()
            test = data[training:].copy()
            test['Predictions'] = predictions
            Utility.plot_results(train, test, predictions)
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")


if __name__ == "__main__":
    app = CryptocurrencyApp()
    app.mainloop()