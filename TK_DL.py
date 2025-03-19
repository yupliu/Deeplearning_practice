from rdkit.Chem import PandasTools
file_path =  f'/home/am7574/Run_dir/Data/TK_ALL.sdf'
df_mol = PandasTools.LoadSDF(file_path)
data_mol = df_mol[['ROMol', 'IC50 uM']]
data_mol = data_mol.dropna()
data_mol = data_mol.reset_index(drop=True)
data_mol['activity'] = data_mol['IC50 uM'].apply(lambda x: 0 if x == 'NI' or float(x) > 20 else 1)
print(data_mol.head())

from rdkit.Chem import rdFingerprintGenerator
fp_gen = rdFingerprintGenerator.GetMorganGenerator(2)
data_mol['fp'] = data_mol['ROMol'].apply(lambda x: fp_gen.GetFingerprintAsNumPy(x))
X= data_mol['fp'].to_list()
y = data_mol['activity'].to_list()

import numpy as np
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim

class TK_Model(nn.Module):
    def __init__(self):
        super(TK_Model, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
  
batch_size = 128
epochs = 15
model = TK_Model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train,y_train, batch_size=batch_size, epochs= epochs, validation_split=0.1)

