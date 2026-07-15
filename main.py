import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generar datos sintéticos
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X + np.random.randn(100, 1)

# Convertir a DataFrame para mejor manejo
df = pd.DataFrame(np.hstack([X, y]), columns=['Feature', 'Target'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[['Feature']], df['Target'], test_size=0.2, random_state=42)

# Entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir y evaluar
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
