
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report



column_names = [
    'word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our',
    'word_freq_over','word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail',
    'word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses',
    'word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit',
    'word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp',
    'word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs',
    'word_freq_telnet','word_freq_857','word_freq_data','word_freq_415','word_freq_85',
    'word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct',
    'word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re',
    'word_freq_edu','word_freq_table','word_freq_conference',
    'char_freq_;','char_freq_(','char_freq_[','char_freq_!','char_freq_$','char_freq_#',
    'capital_run_length_average','capital_run_length_longest','capital_run_length_total',
    'class'
]

df = pd.read_csv("spambase.csv", header=None, names=column_names)



print("\nFirst 10 Records:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())



print("\nClass Distribution:")
print(df['class'].value_counts())

plt.figure()
sns.countplot(x='class', data=df)
plt.title("Spam vs Not Spam Distribution")
plt.xticks([0,1], ['Not Spam', 'Spam'])
plt.show()

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)



model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)



cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam','Spam'],
            yticklabels=['Not Spam','Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()



FP = cm[0][1]
FN = cm[1][0]

print("False Positives:", FP)
print("False Negatives:", FN)


spam_count = df['class'].value_counts()[1]
not_spam_count = df['class'].value_counts()[0]

print("\nSpam Count:", spam_count)
print("Not Spam Count:", not_spam_count)

if abs(spam_count - not_spam_count) > 500:
    print("Dataset is moderately imbalanced.")
else:
    print("Dataset is fairly balanced.")

print("\nExperiment Completed Successfully.")
