# price

# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ================================
# 2. LOAD PRICE DATASET
# ================================
price_url = "https://raw.githubusercontent.com/kartikpardeshi32-cloud/price/main/Price_Agriculture_commodities_Week.csv"
price_df = pd.read_csv(price_url)

price_df.columns = price_df.columns.str.strip()
price_df['Arrival_Date'] = pd.to_datetime(price_df['Arrival_Date'], format="%d-%m-%Y")

# ================================
# 3. LOAD CROP DATASET
# ================================
crop_url = "https://raw.githubusercontent.com/AyushAgnihotri2025/SmartCrop/master/SmartCrop-Dataset.csv"
data = pd.read_csv(crop_url)

data = data.dropna()
data.columns = [col.strip().lower() for col in data.columns]

crop_stats = data.groupby("label").mean()

X = data.drop("label", axis=1)
y = data["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# ================================
# 4. USER INPUT
# ================================
try:
    print("\nEnter Soil & Weather Values:")

    N = float(input("Nitrogen: "))
    P = float(input("Phosphorus: "))
    K = float(input("Potassium: "))
    temp = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    ph = float(input("pH: "))
    rainfall = float(input("Rainfall: "))

    city = input("\nEnter City/Market: ").lower()

except:
    print("❌ Invalid input")
    exit()

# ================================
# 5. CROP PREDICTION
# ================================
input_df = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]],
                        columns=X.columns)

input_scaled = scaler.transform(input_df)

probs = model.predict_proba(input_scaled)[0]
top3_idx = np.argsort(probs)[-3:][::-1]
top3_crops = encoder.inverse_transform(top3_idx)
top3_probs = probs[top3_idx]

# ================================
# 6. YIELD DATA
# ================================
yield_data = {
    "rice": 25,
    "wheat": 20,
    "maize": 18,
    "cotton": 10,
    "sugarcane": 300
}


# ================================
# 7. FUNCTION: GET MARKET PRICE
# ================================
def get_market_price(crop, city):
    filtered = price_df[
        (price_df['Market'].str.lower() == city) &
        (price_df['Commodity'].str.lower() == crop.lower())
        ]

    if not filtered.empty:
        latest = filtered.sort_values(by="Arrival_Date", ascending=False).iloc[0]
        return latest['Modal Price']

    # fallback: average across all cities
    alt = price_df[price_df['Commodity'].str.lower() == crop.lower()]

    if not alt.empty:
        return round(alt['Modal Price'].mean(), 2)

    return None


# ================================
# 8. LOGIC FUNCTION
# ================================
def get_logic(crop, user_input):
    stats = crop_stats.loc[crop]
    reasons = []

    features = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

    for i, feature in enumerate(features):
        diff = abs(user_input[i] - stats[feature])

        if diff < 10:
            reasons.append(f"{feature} ✔")
        else:
            reasons.append(f"{feature} ⚠ ({diff:.1f})")

    return reasons


# ================================
# 9. OUTPUT
# ================================
print("\n==============================")
print("🌾 FarmX Intelligent System")
print("==============================")

user_input = [N, P, K, temp, humidity, ph, rainfall]

for i, (crop, prob) in enumerate(zip(top3_crops, top3_probs), 1):

    price = get_market_price(crop, city)
    yield_est = yield_data.get(crop, 20)

    if price:
        profit = price * yield_est
    else:
        profit = 0

    print(f"\n{i}. {crop} ({prob * 100:.2f}% confidence)")

    if price:
        print(f"💰 Market Price: ₹{price}")
        print(f"📈 Estimated Profit: ₹{profit}/acre")
    else:
        print("❌ Price data not available")

    logic = get_logic(crop, user_input)

    print("📊 Reasoning:")
    for l in logic:
        print("  -", l)

# ================================
# 10. FINAL RECOMMENDATION
# ================================
best_crop = top3_crops[0]
best_price = get_market_price(best_crop, city)
best_profit = (best_price * yield_data.get(best_crop, 20)) if best_price else 0

print("\n📌 Final Recommendation:")

if best_price:
    print(
        f"👉 Under these conditions, {best_crop} is the most profitable choice in {city} with approx profit ₹{best_profit}/acre.")
else:
    print(f"👉 {best_crop} is suitable, but market price data is unavailable.")

print("\n==============================")
