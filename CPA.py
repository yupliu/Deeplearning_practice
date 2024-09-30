import rdkit
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
import mols2grid
import requests

from rdkit.Chem.Draw import IPythonConsole #RDKit drawing

IPythonConsole.ipython_useSVG = True

rdDepictor.SetPreferCoordGen(True)

sdTrainPath = '/home/am7574/Project_backup/CPA/CPA_train_TRIPNI.sdf'
sdTestPath = '/home/am7574/Project_backup/CPA/CPA_test_TRIPNI.sdf'


trainData = PandasTools.LoadSDF(sdTrainPath)
testData = PandasTools.LoadSDF(sdTestPath)

#mols2grid.display(trainData,mol_col="ROMol")
#Draw.MolsToGridImage(trainData["ROMol"], molsPerRow= 4, useSVG=True)

def CalcProp(moldf:pd.DataFrame, molCol: str):
    from rdkit.Chem.Descriptors import CalcMolDescriptors
    moldf["Pprops"] = [CalcMolDescriptors(x) for x in moldf[molCol]]

CalcProp(trainData,'ROMol')


