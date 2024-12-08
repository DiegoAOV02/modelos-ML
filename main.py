import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset en un dataframe
try:
    df = pd.read_csv('respuestas.csv')
    print("Dataset cargado correctamente")
except FileNotFoundError:
    print("No se ha encontrado el archivo 'respuestas.csv'")
    exit()

# Eliminar las columnas no necesarias
columns_to_drop = ["Marca temporal", "Nombre de usuario"]
df = df.drop(columns=columns_to_drop, errors='ignore')

# Dividir las columnas en datos numéricos y categóricas
numeric_data = df.select_dtypes(include=['number'])
categorical_columns = df.select_dtypes(include=['object']).columns

# Asegurar que hay una columna categórica para la etiqueta
if categorical_columns.empty:
    print("No se encontraron columnas categóricas para usar como etiquetas.")
    exit()

# Seleccionar la primera columna categórica como etiqueta
target_column = categorical_columns[0]
y = df[target_column]
X = numeric_data

# Verificar si hay datos numéricos y categóricos suficientes
if X.empty or y.empty:
    print("Faltan datos numéricos o etiquetas categóricas.")
    exit()

# Codificar etiquetas categóricas en valores numéricos
y_encoded = y.astype('category').cat.codes

# Estandarizar los datos numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Entrenar el modelo SVM con kernel radial
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = svm_model.predict(X_test)

# Evaluar el modelo
print("\nMatriz de confusión:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.cat.categories, yticklabels=y.cat.categories)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Verdad")
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=y.cat.categories))

