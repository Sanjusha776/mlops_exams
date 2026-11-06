import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Example data: square feet, bedrooms, age (years), and price (USD).
X = np.array([
	[1200, 2, 20], [1500, 3, 15], [1800, 3, 10], [2200, 4, 8],
	[900, 2, 30], [2000, 3, 5], [1600, 3, 25], [2500, 4, 12],
	[1100, 2, 5], [1900, 4, 18], [2300, 4, 3], [1400, 3, 22],
], dtype=float)
y = np.array([
	240_000, 310_000, 380_000, 470_000, 190_000, 420_000,
	300_000, 510_000, 260_000, 350_000, 530_000, 280_000,
], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.25, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Test mean absolute error: ${mean_absolute_error(y_test, predictions):,.0f}")

# Predict for a 1,700 sq ft, 3-bedroom, 10-year-old home.
home = np.array([[1700, 3, 10]], dtype=float)
print(f"Estimated price: ${model.predict(home)[0]:,.0f}")
