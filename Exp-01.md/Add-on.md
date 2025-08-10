Code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))

for i in range(len(X)):
 if y[i] == 0:
   plt.scatter(X[i][0], X[i][1], color='red')
 else:
   plt.scatter(X[i][0], X[i][1], color='blue')

x_values = [0, 1]
y_values = -(clf.coef_[0][0]*np.array(x_values) + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_values, y_values)
plt.title('Perceptron Decision Boundary for XOR')
plt.show()

Output:

Predictions: [0 0 0 0]
/tmp/ipython-input-801145304.py:17: RuntimeWarning: invalid value encountered in divide
  y_values = -(clf.coef_[0][0]*np.array(x_values) + clf.intercept_)/clf.coef_[0][1]
<img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/e39c4dea-19f2-4bc4-af1e-137fa54f51c6" />
