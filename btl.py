from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

# 1. Get data from csv
data = pd.read_csv('heart_failure_clinical_records_dataset.csv',
                      usecols=['age', 'anaemia', 'creatinine_phosphokinase',
                               'diabetes', 'ejection_fraction', 'high_blood_pressure',
                               'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
# 2. Preprocessing data
le = preprocessing.LabelEncoder()
for column_name in list(data.columns):
    data[column_name] = le.fit_transform(data[column_name])

# 3. Splitting the dataset into train and test
X = data.values[:, :-1]
Y = data.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=100)

clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 6)

clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 4. Tính toán độ chính xác của mô hình
print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)

print("Report : ",
    classification_report(y_test, y_pred))

# 5. Thực nghiệm
# Kfold checking
n_splits = 5
kf = KFold(n_splits=n_splits)

# Raw data
X = data.values[:, :-1]
Y = data.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=100)

for i in [1, 2, 3, 4, 5, 6]:
    sum_train_err = 0
    sum_test_err = 0
    for train_index, test_index in kf.split(X_train):
        KX_train, KX_test = X_train[train_index], X_train[test_index]
        ky_train, ky_test = y_train[train_index], y_train[test_index]

        # Kiểm nghiệm mô hình
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=i)
        clf = clf.fit(KX_train, ky_train)
        y_train_pred = clf.predict(KX_train)
        y_test_pred = clf.predict(KX_test)

        sum_train_err += (100 - (accuracy_score(ky_train,y_train_pred)*100))
        sum_test_err += (100 -(accuracy_score(ky_test,y_test_pred)*100))
    sum_err = sum_train_err/n_splits + sum_test_err/n_splits
    print("Trung binh loi train lan lap", i , ":", sum_train_err / n_splits)
    print("Trung binh loi test lan lap", i , ":", sum_test_err / n_splits)
    print("Tong trung binh loi lan", i, ":", sum_err, "\n")
