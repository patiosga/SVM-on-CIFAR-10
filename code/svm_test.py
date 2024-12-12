from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM with a polynomial kernel
poly_model = SVC(kernel='poly', degree=4, coef0=1, C=1.0)
poly_model.fit(X_train, y_train)

# Test the model
y_pred = poly_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Polynomial Kernel (degree=3): {accuracy:.2f}")



import matplotlib.pyplot as plt
import numpy as np

# Plot decision boundary
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title(title)
    plt.show()

# Test different degrees
for degree in [2, 3, 4]:
    poly_model = SVC(kernel='linear', degree=degree, coef0=1, C=1.0)
    poly_model.fit(X_train, y_train)
    plot_decision_boundary(poly_model, X, y, f"Polynomial Kernel (degree={degree})")
