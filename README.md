# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 02/05/2026

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
df = pd.read_csv("/content/archive (2).zip")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
# Use Adjusted Close as the time series, enforce business-day frequency
X = df["Close"].asfreq("B")
X_log = np.log(X)
X_log_diff = X_log.diff().dropna()
plt.figure(figsize=(12,6))
plt.plot(X_log_diff, label="Log Differenced Adj Close")
plt.title("Final Goldrate Time Series")
plt.xlabel("Date")
plt.ylabel("Log Diff Price")
plt.legend()
plt.show()
result = adfuller(X_log_diff)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
# Plot ACF and PACF
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plot_acf(X_log_diff, lags=40, ax=plt.gca())
plt.title("Final Goldrate Data ACF")
plt.subplot(2,1,2)
plot_pacf(X_log_diff, lags=40, ax=plt.gca())
plt.title("Final Goldrate Data PACF")
plt.tight_layout()
plt.show()
arma11_model = ARIMA(X_log_diff, order=(1, 0, 1)).fit()
print("ARMA(1,1) Summary:\n", arma11_model.summary())
phi1 = arma11_model.params.get('ar.L1', 0)
theta1 = arma11_model.params.get('ma.L1', 0)
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])
arma11_sim = ArmaProcess(ar1, ma1).generate_sample(nsample=200)
plt.plot(arma11_sim)
plt.title("Simulated ARMA(1,1) Process")
plt.show()
plot_acf(arma11_sim)
plt.show()
plot_pacf(arma11_sim)
plt.show()
arma22_model = ARIMA(X_log_diff, order=(2, 0, 2)).fit()
print("ARMA(2,2) Summary:\n", arma22_model.summary())
phi1 = arma22_model.params.get('ar.L1', 0)
phi2 = arma22_model.params.get('ar.L2', 0)
theta1 = arma22_model.params.get('ma.L1', 0)
theta2 = arma22_model.params.get('ma.L2', 0)
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])
arma22_sim = ArmaProcess(ar2, ma2).generate_sample(nsample=200)
plt.plot(arma22_sim)
plt.title("Simulated ARMA(2,2) Process")
plt.show()
plot_acf(arma22_sim)
plt.show()
plot_pacf(arma22_sim)
plt.show()
```

## OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="915" height="507" alt="image" src="https://github.com/user-attachments/assets/0cb54887-ca4b-437f-94eb-1783c5b483ac" />

<img width="682" height="547" alt="image" src="https://github.com/user-attachments/assets/3f69849d-67c7-4171-af65-d42e4f6a2ba6" />

Partial Autocorrelation
<img width="707" height="551" alt="image" src="https://github.com/user-attachments/assets/a8fd3219-f02c-4d56-87d8-4d9df859eec2" />

Autocorrelation
<img width="702" height="540" alt="image" src="https://github.com/user-attachments/assets/867c2ba8-9343-4d55-8fc6-ebbece7d963e" />


SIMULATED ARMA(2,2) PROCESS:

<img width="920" height="571" alt="image" src="https://github.com/user-attachments/assets/699fa724-9ddf-4ed2-b074-eaa710975f47" />
<img width="677" height="538" alt="image" src="https://github.com/user-attachments/assets/d46e4bb3-dd7a-43ba-b3c4-53ed3903173e" />

Partial Autocorrelation

<img width="695" height="540" alt="image" src="https://github.com/user-attachments/assets/09d861ce-8369-464c-84fb-8bdcce66c276" />

Autocorrelation

<img width="692" height="536" alt="image" src="https://github.com/user-attachments/assets/78e139d7-24b5-4df2-a001-7816dee13e25" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
