import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data = pd.read_csv("cervicalCancerFix.csv")
Y = data.iloc[0:668, 28]
data = data.iloc[0:668, 0:28]
from sklearn.model_selection import train_test_split
X = data.iloc[0:668, 0:28]

X_train, X_Test, y_train, y_test = train_test_split(X,Y,
                                                    test_size = 0.2,
                                                    random_state = 0)
model = RandomForestClassifier(max_depth = None, random_state = None)
model.fit(X_train, y_train)
predic = model.predict(X_Test)
cm = confusion_matrix(predic, y_test)
important = model.feature_importances_

plt.figure(1)
indices = np.argsort(important)
plt.title('Important Features')
plt.xlabel('Relative Importance')
features = list(data)
plt.barh(range(len(indices)), important[indices], color = 'b', align = 'center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.show()
