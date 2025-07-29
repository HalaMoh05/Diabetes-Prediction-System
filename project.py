# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:30:00 2025

@author: Hala
"""

import numpy as np #للعمليات العددية
import pandas as pd #لقراءة البيانات

# Step 1: استيراد البيانات
data = pd.read_csv('diabetes.csv')
print(data.info())
print(data.head()) #طباعة اول 5 صفوف من البيانات لمعاينتها
print(data.isna().sum()) #بحسب عدد القيم الناقصة (NaN) في كل عامود
print(data['Age'].head()) 
print(data.loc[2])

# Step 2: فصل X و y
X = data.iloc[:, 0:-1].values  # كل الأعمدة ما عدا Outcome
y = data.iloc[:, -1].values    # عمود Outcome فقط

# Step 3: تعويض القيم الناقصة بالقيمة المتوسطة للأعمدة
from sklearn.impute import SimpleImputer
Simpleimputer = SimpleImputer(missing_values=0, strategy='mean')

# الأعمدة التي فيها أصفار تعتبر غير منطقية ونعوضها
columns_with_zeros = [1,2,3,4,5]  # Glucose, BloodPressure, SkinThickness, Insulin, BMI
Simpleimputer.fit(X[:, columns_with_zeros])
X[:, columns_with_zeros] = Simpleimputer.transform(X[:, columns_with_zeros])

# Step 4: تقسيم البيانات إلى تدريب واختبار
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: تدريب موديل Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: توقع القيم للبيانات الاختبارية
y_pred = model.predict(X_test)

# Step 7: تقييم الموديل
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))


# Step 8: حفظ الموديل
from joblib import dump
dump(model, 'diabetes_model.pkl')
print("Model saved as diabetes_model.pkl")
