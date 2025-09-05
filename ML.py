from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier instance and fit it to some sample data
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)
RandomForestClassifier(random_state=0)

clf.predict(X)  # predict classes of the training data
# array([0, 1])
clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data
# array([0, 1])