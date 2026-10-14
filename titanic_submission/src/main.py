import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
SEED = 33
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)




data = pd.read_csv(r'C:\Users\ss\Desktop\titanic_submission\src\train.csv')

print(data.info())      
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
print("数据处理完毕！现在的表格长这样：")
print(data.head())
print("\n再检查一下还有没有缺失值和文字：")
print(data.info())

y = data['Survived'].values.astype(np.float32)
X = data.drop('Survived', axis=1).values.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


train_age_mean = X_train[:, 2].mean()
train_embarked_mode = 'S'  

print("原始数据总数:", len(X))
print("训练集数量:", len(X_train))
print("测试集数量:", len(X_test))

class TitanicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_dataset = TitanicDataset(X_train, y_train)
test_dataset = TitanicDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TitanicModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

input_size = X_train.shape[1]
model = TitanicModel(input_size)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
loss_history = []
train_acc_history = []
test_acc_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for features, labels in train_loader:
        y_pred = model(features)
        loss = criterion(y_pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (y_pred >= 0.5).float()
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)
        
    avg_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train
    loss_history.append(avg_loss)
    train_acc_history.append(train_acc)
    
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad(): 
        for features, labels in test_loader:
            y_pred = model(features)
            predictions = (y_pred >= 0.5).float()
            correct_test += (predictions == labels).sum().item()
            total_test += labels.size(0)
            
    test_acc = correct_test / total_test
    test_acc_history.append(test_acc)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | "
              f"训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

torch.save(model.state_dict(), 'titanic_model.pth')
print("\n训练完成！模型已保存为 titanic_model.pth")






import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, label='训练损失 (Loss)', color='red')
plt.xlabel('Epoch（训练轮次）')
plt.ylabel('Loss（损失值）')
plt.title('模型损失趋势图')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_acc_history, label='训练集准确率', color='blue')
plt.plot(range(1, epochs + 1), test_acc_history, label='测试集准确率', color='orange')
plt.xlabel('Epoch（训练轮次）')
plt.ylabel('Accuracy（准确率）')
plt.title('训练集与测试集准确率对比图')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png')
plt.show()

print("\n画图完成！")


import json

preprocess_info = {
    'age_mean': float(train_age_mean),
    'embarked_mode': str(train_embarked_mode),
    'feature_columns': list(data.drop('Survived', axis=1).columns)
}

with open('preprocess_info.json', 'w', encoding='utf-8') as f:
    json.dump(preprocess_info, f, ensure_ascii=False, indent=4)

print("预处理信息已保存为 preprocess_info.json")

def predict_new_passenger(passenger_info):
    df = pd.DataFrame([passenger_info])
    
    df['Age'] = df['Age'].fillna(preprocess_info['age_mean'])
    df['Embarked'] = df['Embarked'].fillna(preprocess_info['embarked_mode'])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    for col in preprocess_info['feature_columns']:
        if col not in df.columns:
            df[col] = 0
    df = df[preprocess_info['feature_columns']]
    
    features_tensor = torch.tensor(df.values, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        probability = model(features_tensor).item()
    
    result = 1 if probability >= 0.5 else 0
    return result, probability

print("\n测试新样本预测：")
print("请输入乘客信息（直接按回车使用默认值）：")

pclass = int(input("请输入舱位等级 (1/2/3，默认3): ") or 3)
sex = input("请输入性别 (male/female，默认male): ") or "male"
age = float(input("请输入年龄 (如 25.5，默认 30): ") or 30)
sibsp = int(input("请输入兄弟姐妹/配偶数量 (默认 0): ") or 0)
parch = int(input("请输入父母/子女数量 (默认 0): ") or 0)
fare = float(input("请输入票价 (默认 8.05): ") or 8.05)
embarked = input("请输入登船港口 (C/Q/S，默认 S): ") or "S"


new_passenger = {
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}

# 调用预测函数
prediction, prob = predict_new_passenger(new_passenger)

print("\n" + "="*30)
print(f"您输入的乘客信息: {new_passenger}")
print(f"模型预测的生存概率: {prob:.4f}")
print(f"最终预测结果 (0=遇难, 1=幸存): {prediction}")
print("="*30)


prediction, prob = predict_new_passenger(new_passenger)
print(f"预测乘客信息: {new_passenger}")
print(f"模型预测的生存概率: {prob:.4f}")
print(f"最终预测结果 (0=遇难, 1=幸存): {prediction}")
