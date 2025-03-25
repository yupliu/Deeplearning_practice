from rdkit.Chem import PandasTools
file_path =  f'/home/am7574/Run_dir/Data/TK_ALL.sdf'
df_mol = PandasTools.LoadSDF(file_path)
data_mol = df_mol[['ROMol', 'IC50 uM']]
data_mol = data_mol.dropna()
data_mol = data_mol.reset_index(drop=True)
data_mol['activity'] = data_mol['IC50 uM'].apply(lambda x: 0 if x == 'NI' or float(x) > 20 else 1)
print(data_mol.head())

def calcFingerprint(mols):
    from rdkit.Chem import rdFingerprintGenerator
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(2)
    mols["MolFP"] = mols["ROMol"].apply(lambda x: fp_gen.GetCountFingerprintAsNumPy(x))
    return mols
X_fp = calcFingerprint(data_mol)
X= X_fp["MolFP"].to_list()
y = data_mol['activity'].to_list()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim

class TK_Model(nn.Module):
    def __init__(self):
        super(TK_Model, self).__init__()
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 32)
        #self.fc3 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        #x = self.relu(x)
        #x = self.fc4(x)
        x = self.sigmoid(x)
        return x
  
batch_size = 128
epochs = 15
model = TK_Model()
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#model.fit(X_train,y_train, batch_size=batch_size, epochs= epochs, validation_split=0.1)

criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(torch.tensor(X_train).float())
    loss = criteria(y_pred, torch.tensor(y_train).float().view(-1,1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
y_pred = model(torch.tensor(X_test).float())
y_pred = y_pred.detach().numpy()
y_pred = [1 if i > 0.5 else 0 for i in y_pred]
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
accuracy = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')