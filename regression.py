import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("aggregated_per_day.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

#daily sentiment plot
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Score"], linewidth=0.8, alpha=0.7)
plt.title("Daily Aggregated News Sentiment")
plt.xlabel("Date")
plt.ylabel("Sentiment score")
plt.axhline(0, linestyle=":", alpha=0.4)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ts_daily_sentiment.png", dpi=300, bbox_inches="tight")
#plt.show()


#daily returns plot
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Percentage price difference"], linewidth=0.8, alpha=0.7)
plt.title("Daily S&P 500 Percentage Returns")
plt.xlabel("Date")
plt.ylabel("Return (%)")
plt.axhline(0, linestyle=":", alpha=0.4)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ts_daily_returns.png", dpi=300, bbox_inches="tight")
#plt.show()

#combined plot
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df["Date"], df["Percentage price difference"], color="blue", alpha=0.6)
ax1.set_ylabel("Return (%)", color="blue")
ax1.axhline(0, linestyle=":", alpha=0.3)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df["Date"], df["Score"], color="green", alpha=0.6)
ax2.set_ylabel("Sentiment score", color="green")
ax2.axhline(0, linestyle=":", alpha=0.3)

plt.title("Daily Returns and News Sentiment")
plt.tight_layout()
plt.savefig("ts_daily_returns_and_news_sentiment.png", dpi=300, bbox_inches="tight")
plt.show()

#day shift
df["Return_Next"] = df["Percentage price difference"].shift(-1)
df = df.dropna()

#sanity check
print(df.columns)
df[["Date", "Score", "Percentage price difference", "Return_Next"]].head(5)
df.iloc[0]["Return_Next"] == df.iloc[1]["Percentage price difference"]

#sanity check - we verify alignment by checking that Return_Next at row t equals the percentage 
# return at row t+1, confirming that each day’s sentiment is matched with the following day’s market return (we should get the same number)
print("Return_Next (t):", df.iloc[0]["Return_Next"])
print("Return (t+1):", df.iloc[1]["Percentage price difference"])


#Simple regression model setup
from sklearn.model_selection import train_test_split

X_sent = df[["Score"]]
y = df["Return_Next"]

# time-based split (80% train, 20% test)
split_idx = int(len(df) * 0.8)

X_train_sent = X_sent.iloc[:split_idx]
X_test_sent  = X_sent.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

#Baseline model: predict the mean of the training returns
import numpy as np

baseline_pred = np.full_like(y_test, y_train.mean())

#Linear Regression model
from sklearn.linear_model import LinearRegression

sent_model = LinearRegression()
sent_model.fit(X_train_sent, y_train)

sent_pred = sent_model.predict(X_test_sent)

#Evaluation
from sklearn.metrics import root_mean_squared_error

rmse_baseline = root_mean_squared_error(y_test, baseline_pred)
rmse_sent     = root_mean_squared_error(y_test, sent_pred)

print("Baseline RMSE:", rmse_baseline)
print("Sentiment RMSE:", rmse_sent)


#Direction accuracy (hit rate)
import numpy as np

baseline_dir = np.sign(baseline_pred)
sent_dir     = np.sign(sent_pred)
true_dir     = np.sign(y_test)

baseline_hit = np.mean(baseline_dir == true_dir)
sent_hit     = np.mean(sent_dir == true_dir)

print("Baseline hit rate:", baseline_hit)
print("Sentiment hit rate:", sent_hit)


# Correlation between predictions and true returns
from scipy.stats import pearsonr, spearmanr

pearson = pearsonr(df["Score"], df["Return_Next"])
spearman = spearmanr(df["Score"], df["Return_Next"])

print("Pearson:", pearson)
print("Spearman:", spearman)


for lag in range(0, 6):
    corr = pearsonr(df["Score"].shift(lag).dropna(),
                    df["Return_Next"][lag:])[0]
    print(f"Lag {lag}: corr = {corr:.3f}")
