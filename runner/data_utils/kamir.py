import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .base_datamodule import DataModule
from tqdm import tqdm
import pickle
from types import SimpleNamespace
from typing import Tuple, List, Dict

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
import pandas as pd
from typing import List

class KamirDataModule(DataModule):
    
    def __init__(self,
        task: str,
        config: SimpleNamespace
        ) -> None:
        super().__init__(task)
        self.config = config

    def load_data(self, 
                sheet_name: str ='KAMIR-V 1년 F-U DATA', 
                skiprows: List[int] = [0]
        ) -> pd.DataFrame:
        data_path = self.config.file_path.split('/')
        dir_path = '/'.join(data_path[:-1])
        file_name = data_path[-1]

        if os.path.isfile(f'{dir_path}/data.pickle'):
            data = pickle.load(open(f'{dir_path}/data.pickle', 'rb'))
        else:
            data = pd.read_excel(f'{dir_path}/{file_name}', sheet_name=sheet_name, skiprows=skiprows, engine='openpyxl')
            pickle.dump(data, open(f'{dir_path}/data.pickle', 'wb'))
        
        return data

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:

        if os.path.exists(self.config.dataset_path):
            with open(self.config.dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            return dataset['data'], dataset['label']
        
        data = self.load_data()

        categorical_cols = data.columns[self.config.categorical_cols_idx].to_list()
        binary_cols = data.columns[self.config.binary_cols_idx].to_list()
        continuous_cols = data.columns[self.config.continuous_cols_idx].to_list()

        data = self.str_to_float(data, continuous_cols)
        data = self.abnormal2nan(data)
        label = self.get_label(data)

        data = self.organize_initial_diagnosis(data)

        data = self.organize_sex(data)

        data, medicine_cols = self.get_medicine(data)
        
        continuous_cols.extend(medicine_cols)
        
        data = data[categorical_cols + binary_cols + continuous_cols]

        data, drop_cols = self.drop_over_missing(data) 
        
        binary_cols, continuous_cols, categorical_cols = self.get_remained_cols(binary_cols, continuous_cols, categorical_cols, drop_cols)
        
        imputer = Imputer()
        imputed = imputer.impute(data, binary_cols, continuous_cols, categorical_cols)

        data, categorical_cols = self.to_onehot(imputed, categorical_cols)


        le = LabelEncoder()

        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])

        
        # if hasattr(self.config, 'rename_cols'):
        data.rename(columns=self.config.rename_cols, inplace=True)
        for idx, c in enumerate(continuous_cols):
            if c in self.config.rename_cols.keys():
                continuous_cols[idx] = self.config.rename_cols[c]
        
        for idx, c in enumerate(categorical_cols):
            if c in self.config.rename_cols.keys():
                categorical_cols[idx] = self.config.rename_cols[c]

        # if self.config.runner_option.save_data:
        # self.save_data(data, label)
            
        return data, label.values, continuous_cols, categorical_cols

    def get_6M(self, 
                data: pd.DataFrame
        ) -> pd.Series:
        label_6M = data.iloc[:, 320]
        label_6M = label_6M.apply(lambda x : 1 if x == 'Death' else 0)
        return label_6M
    
    def get_12M(self, 
                data: pd.DataFrame
        ) -> pd.Series:
        label_12M_idx = data['1 Year Follow-up'].apply(bool)
        label_12M = (data.loc[label_12M_idx, '12M_Cardiac death'] == 1) | (data.loc[label_12M_idx, '12M_Non-cardiac death'] == 1)
        label_12M = label_12M.apply(lambda x : 1 if x == True else 0)
        return label_12M
    
    def get_reverse_remodeling(self, 
                                data: pd.DataFrame
        ) -> pd.Series:
        LVEF = self.percentage_to_float(data, 'LVEF')
        LVEDV = self.percentage_to_float(data, 'LVEDV')
        LVESV = self.percentage_to_float(data, 'LVESV')

        LVEF_12M = self.percentage_to_float(data, '12M_LVEF')
        LVEDV_12M = self.percentage_to_float(data, '12M_LVEDV')
        LVESV_12M = self.percentage_to_float(data, '12M_LVESV')

        LVEF_gap = (LVEF_12M - LVEF)[(~(LVEF - LVEF_12M).isna() | ~(LVEDV - LVEDV_12M).isna() | ~(LVESV - LVESV_12M).isna())].apply(lambda x : 1 if (x >= 10) & (not np.isnan(x)) else 0)
        LVEDV_gap = (LVEDV_12M - LVEDV)[(~(LVEF - LVEF_12M).isna() | ~(LVEDV - LVEDV_12M).isna() | ~(LVESV - LVESV_12M).isna())].apply(lambda x : 1 if (x <= 10) & (not np.isnan(x)) else 0)
        LVESV_gap = (LVESV_12M - LVESV)[(~(LVEF - LVEF_12M).isna() | ~(LVEDV - LVEDV_12M).isna() | ~(LVESV - LVESV_12M).isna())].apply(lambda x : 1 if (x <= 10) & (not np.isnan(x)) else 0)

        reverse_remodeling = (LVEF_gap + LVEDV_gap + LVESV_gap).apply(lambda x : 1 if x > 0 else x)

        return reverse_remodeling

    def get_label(self, 
                    data: pd.DataFrame
        ) -> pd.Series:
        if self.config.target == '6M':
            return self.get_6M(data)
        elif self.config.target == '12M':
            return self.get_12M(data)
        elif self.config.target == 'reverse_remodeling':
            return self.get_reverse_remodeling(data)

    def str_to_float(self, 
                        data: pd.DataFrame, 
                        target_cols: List[str]
        ) -> pd.DataFrame:

        for col in target_cols:
            data[col] =  data[col].apply(lambda x : x.split('|') if type(x) == str else x)
            data[col] = data[col].apply(lambda x : float('.'.join([_.strip().strip('.') for _ in x])) if type(x) == list else x)
        return data

    def percentage_to_float(self, 
                            data: pd.DataFrame, 
                            target_col: str
        ) -> pd.Series:
        LVEs =  data[target_col].apply(lambda x : x.split('|') if type(x) == str else x)
        return LVEs.apply(lambda x : float('.'.join([_.strip().strip('.') for _ in x])) if type(x) == list else x)
    
    def get_bmi(self, 
                data: pd.DataFrame
        ) -> pd.Series:
        bmi = data['WT'] / (data['HT'] / 100) ** 2
        return bmi

    def abnormal2nan(self, 
                    data: pd.DataFrame
        ) -> pd.DataFrame:
        for key in self.config.abnormal_numerics.keys():
            limits = self.config.abnormal_numerics[key]
            data[key][(data[key] < limits[0]) | (data[key] > limits[1])] = np.nan
        
        bmi = self.get_bmi(data)
        bmi_idx = ((bmi < 10) | (bmi > 50))
        data['WT'][bmi_idx] = np.nan
        data['HT'][bmi_idx] = np.nan
        
        wrong_idx = (data['LVEDD'] <= data['LVESD']) | (data['LVEDD'] > 100) | (data['LVEDD'] < 0) | (data['LVESD'] > 100) | (data['LVESD'] < 0)
        data['LVEDD'][wrong_idx] = np.nan
        data['LVESD'][wrong_idx] = np.nan

        wrong_idx = (data['LVEDV'] <= data['LVESV']) | (data['LVEDV'] > 500) | (data['LVEDV'] < 0) | (data['LVESV'] > 500) | (data['LVESV'] < 0)
        data['LVEDD'][wrong_idx] = np.nan
        data['LVESD'][wrong_idx] = np.nan
        
        return data

    def organize_initial_diagnosis(self, 
                                    data: pd.DataFrame
        ) -> pd.DataFrame:
        for i in range(len(data)):
            if data['Initial diagnosis'][i] == "NSTEMI" and np.isnan(data['STEMI'][i]):
                data['STEMI'][i] = 'N'
            elif data['Initial diagnosis'][i] == "STEMI" and np.isnan(data['NSTEMI'][i]):
                data['NSTEMI'][i] = 'N'
            
        return data
    
    def organize_sex(self, 
                    data: pd.DataFrame
        ) -> pd.DataFrame:
        data['Sex'] = data['Sex'].apply(lambda x : 0 if x == '여' else 1)
        return data
    
    def get_missing_ratio(self, 
                        data: pd.DataFrame
        ) -> pd.Series:
        res = {}
        for col in data.columns:
            res[col] = data[col].isnull().sum() / len(data)
        return res

    def drop_over_missing(self, 
                            data: pd.DataFrame, 
                            col_dict: Dict[str, str] = None
        ) -> Tuple[pd.DataFrame, List[str]]:
        missing_ratio = self.get_missing_ratio(data)

        drop_cols = []
        for k in missing_ratio:
            if missing_ratio[k] > self.config.allowed_missing:
                drop_cols.append(k)
                if col_dict != None:
                    print(k, col_dict[k], "| %d%% |" % (round(missing_ratio[k] * 100)))
        
        data = data.drop(drop_cols, axis=1)

        return data, drop_cols
    
    def get_remained_cols(self, 
                        binary_cols: List[str], 
                        continuous_cols: List[str], 
                        categorical_cols: List[str], 
                        drop_cols: List[str]
        ) -> Tuple[List[str], List[str], List[str]]:
        binary = []
        for col in binary_cols:
            if not col in drop_cols:
                binary.append(col)
        
        numeric = []
        for col in continuous_cols:
            if not col in drop_cols:
                numeric.append(col)
        
        category = []
        for col in categorical_cols:
            if not col in drop_cols:
                category.append(col)
        
        return binary, numeric, category
    
    def to_onehot(self, 
                    data: pd.DataFrame, 
                    categorical_cols: List[str]
        ) -> Tuple[pd.DataFrame, List[str]]:
    
        onehot_cols = self.config.onehot_cols
        
        for k in onehot_cols.keys():
            for i in range(len(onehot_cols[k])):
                onehot_cols[k][i] = k + '|-|' + onehot_cols[k][i]

        final_onehot_cols = []
        for k in onehot_cols.keys():
            final_onehot_cols += onehot_cols[k]
        
        temp = pd.DataFrame(np.zeros((len(data), len(final_onehot_cols))), columns=final_onehot_cols, dtype=np.int8)
        
        for i in tqdm(range(len(data))):
            for col in temp.columns:
                for k in onehot_cols.keys():
                    if col.split('|-|')[1] in data[k][i]:
                        temp[col][i] = 1
                        break
        
        data = data.merge(temp, left_index=True, right_index=True)
        data.drop(onehot_cols.keys(), inplace=True, axis=1)

        for k in onehot_cols.keys():
            categorical_cols.remove(k)
        
        data[categorical_cols] = data[categorical_cols].astype("str")

        return data, categorical_cols
    
    def get_medicine(self, data):

        medicine_cols = []
        
        temp = {
            'Bisoprolol' : ['Concor', '콩코르정', 'bisoprolol', 'cobis', 'conbloc',
                            'conocor', 'Bisoprolo', 'Combloc', 'Conbroc', 'conbolc'], 
            'Carvedilol' : ['Dilatrend', 'dilatrend SR', '딜라트렌(정)', '딜라트렌 에스알', 'carvedilol', 'carvelol', 'vasotrol',
                            '딜라트렌', 'Dillatrend', 'diltrend', 'Dilatrned', 'Dilatend', 'dialtrend', 'Dilantrend', 'dilatred', 'DIILATREND', 'DILATRNED', '딜리트렌', 'dialtrend', 'dilarend', 'dilatredn', 'Dilatren', 'dilatren', 'dilattrend', 'Dilarend', 'Dilatren', 'dilatrencd', 'Diltatrend', 'diliatrend', 'DIALTEND', 'Diatrend', 'dilatend', 'dilatrens', 'Dilatresnd', 'dilatrrend', 'Diltrend', 'dilatend', 'dilatred', 'dilatrnd', 'dilatrned' 
                            'dolatrend', 'cilatrend'], 
            'Nebivolol' : ['Nebistol', 'nebilet', 'nebiret', 'nebivolol',
                            'Nebiret', 'nebisol', 'nebistal', 'nebisto', 'nebistrol', 'nebitol', 'nebstol', 'nevisol', 'Nrbistol', 'Vebistol', 'nebistol', 'NEVISTOL', 'Neblstol', 'Nebistiol', 'neebistol', 'nebistlol', 'nedistol', 'neistol', 'Nevistol', 'nebiistol'], #오타
            'Atenolol' : ['Atenolol', 'tenormin',],
            'Betaxolol' : ['kerlone',],
            'Metoprolol' : ['Metoprolol', 'Betaloc'],
            'Celiprolol' : ['selectol']
        }

        for key in temp.keys():
            data[key] = 0
            medicine_cols.append(key)
            for v in temp[key]:
                idxs = data.iloc[:, 370].str.upper().str.strip().str.contains(v.upper()).fillna(False)
                data.loc[idxs, key] = self.percentage_to_float(data, 'Using Dose.6')[idxs]
        
        temp = {
            'Perindopril' : ['Acertil', '아서틸', 'Perindopril',
                                'aceril', 'acerrtil', 'acertik', 'acetil', 'acetril', 'acsrtil', 'aertil', 'acerpril'],
            'Captopril' : ['Capril', 'A-rin', 'Carfril', 'Captopril'],
            'Enalapril' : ['Enaprin', 'Lenipril'],
            'Ramilpril' : ['Tritace', 'Heartpril', 'Ramipril',
                            'TRIATACE', 'Tritacr', 'tritice', 'ritace tab'],
            'Cilazapril' : ['Cilazapril', 'Inhibace'],
            'Imidapril' : ['Tanatril'],
            'Zofenopril' : ['Zofenil']
        }

        for key in temp.keys():
            data[key] = 0
            medicine_cols.append(key)
            for v in temp[key]:
                idxs = data.iloc[:, 373].str.upper().str.strip().str.contains(v.upper()).fillna(False)
                data.loc[idxs, key] = self.percentage_to_float(data, 'Using Dose.7')[idxs]

        temp = {
            'Atorvastatin' : ['Atozet', 'Lipitor', '리피토정', 'arovan', 'atorva', 'atorvastatin', 'atorvin', 'atova', 'caduet', 'lipilou', 'lipinon', 'lipiwon', 'neustatin-A', 'newvast', '아토젯', 'Ataozet', 'aticzet', 'atoxet', 'Liipirotr', 'lipirou', 
                                'kipitor', 'atrozet', 'Liipitor', 'lipicon', 'Lipidil Supra', 'Nerstatin-R', 'Neu-statin-R', 'Lipitoer Tab', 'Neustatin R', 'liipitor', 'Lipitol', 'lipitopr', 'neustatin A', 'Atorcastatin', 'Atrova', 'Lipitou', 'nesustatin-a', 'lipiotr', 'lipitoe', 'atrovan statin'],
            'Rosuvastatin' : ['Crestor', 'rosuzet', 'vivacor', '비바코정', 'allstatin', 'creazin', 'credouble', 'cresant', 'cresnon', 'crezet', 'duonon', 'duowell', 'esuba', 'megarozet', 'neustatin-R', 'olosta', 'rosulord', 'rosuvamibe', 'rotacand', 'rovatitan', 'rovaid', 'rovasta', 'rovazet', 'rovelito', 'suvast', '로바스타정', '로바젯', '로벨리토정', '콜레스논정', '크레스토', '올로스타정', 'Crestorr', 'crestpr',
                                'Cresto', 'Rosuvamide', 'cresrtor', 'Vivarcor', 'Rovatitian', 'Rouzet', 'Vivcor', '콜레스논 정', 'crestro', '20mg 비비코정', '로바스타 정', 'rosuvatine', 'rosubamibe',
                                'Cholesnone', 'Cholesnon', 'cholesnone'] ,
            #'Lovastatin' : [],
            'Simvastatin' : ['Vytorin', 'simvalord', 'simvarol', 'simvastar', 'simvastatin', 'sistar', 'Zocor',
                                'vytrorin', 'simvasta'],
            'Pravastatin' : ['Mevalotin', 'pravafenix', 'prastan',
                                'Mevalothin', 'Mevalitin Tab'],
            'Fluvastatin(XL)' : ['lescol XL', 
                                    'lesxol xl'],
            'Fluvastatin' : ['Lescol'],
            'Pitavastatin' : ['Livalo', 'livalo V', 'pitaduce', 'pitavastatin', 'Liavalo',
                                'Lavalo', 'Livaro V', 'livaro', 'Livallo V']
        }

        for key in temp.keys():
            data[key] = 0
            medicine_cols.append(key)
            for v in temp[key]:
                idxs = data.iloc[:, 380].str.upper().str.strip().str.contains(v.upper()).fillna(False)
                data.loc[idxs, key] = self.percentage_to_float(data, 'Using Dose.9')[idxs]
        
        data['omega3'] = 0
        idxs = data.iloc[:, 395].apply(lambda x : True if x == 1.0 else False)
        data.loc[idxs,'omega3'] = self.percentage_to_float(data, 'Using Dose.14')[idxs]
        medicine_cols.append('omega3')

        return data, medicine_cols

class Converter(object):
    def __init__(self):
        pass

    def encode(self, 
                data: pd.DataFrame
        ) -> pd.DataFrame:
        self.encoder = {}
        for col in tqdm(data.columns):
            self.encoder[col] = LabelEncoder()
            self.encoder[col].fit(data[col])
            nan_idx = data[col].isna()
            data[col] = self.encoder[col].transform(data[col])
            data[col][nan_idx] = np.nan
        return data
    
    def decode(self, 
                data: pd.DataFrame
        ) -> pd.DataFrame:
        for col in tqdm(data.columns):
            data[col] = np.round(data[col])
            data[col] = self.encoder[col].inverse_transform(data[col])
        return data

class Imputer(object):
    def __init__(self, 
                estimator: str = "BayesianRidge", 
                n_estimators: int = None, 
                random_state: int = 0, 
        ) -> None:

        self.converter = Converter()
        self.random_state = random_state
        if estimator == "ExtraTree":
            self.estimator = ExtraTreesRegressor(random_state=self.random_state, n_estimators=n_estimators)
        elif estimator == "BayesianRidge":
            self.estimator = BayesianRidge()
    

    def category_impute(self, 
                        data: pd.DataFrame
        ) -> pd.DataFrame:
        """
        data : pandas DataFrame
        """
        encoded = self.converter.encode(data)
        max_value = [len(self.converter.encoder[c].classes_) - 2 for c in data.columns]
        min_value = 0
        imputer = IterativeImputer(min_value=min_value, max_value=max_value, random_state = self.random_state, estimator=self.estimator).fit(encoded)
        imputed = imputer.transform(encoded)

        for i in range(len(data.columns)):
            data.iloc[:, i] = imputed[:, i].astype(np.uint32)
        converted = self.converter.decode(data)

        return converted

    def numeric_impute(self, 
                        data: pd.DataFrame
        ) -> pd.DataFrame:
        max_value = [data[col].max() for col in data.columns]
        min_value = [data[col].min() for col in data.columns]
        imputer = imputer = IterativeImputer(min_value=min_value, max_value=max_value, random_state = self.random_state, estimator=self.estimator).fit(data.to_numpy())
        imputed = imputer.transform(data.to_numpy())

        for i in range(len(data.columns)):
            data.iloc[:, i] = imputed[:, i].astype(np.float32)

        return data
    
    def binary_impute(self, 
                        data: pd.DataFrame
        ) -> pd.DataFrame:
        max_value = 1
        min_value = 0
        imputer = imputer = IterativeImputer(min_value=min_value, max_value=max_value, random_state = self.random_state, estimator=self.estimator).fit(data.to_numpy())
        imputed = imputer.transform(data.to_numpy())

        for i in range(len(data.columns)):
            data.iloc[:, i] = imputed[:, i].astype(np.uint32)

        for col in data.columns:
            data[col] = data[col].apply(lambda x: 0 if x < 0.5 else 1)
        
        return data

    def impute(self, 
                data: pd.DataFrame, 
                binary_cols: List[str], 
                numeric_cols: List[str], 
                categorical_cols: List[str]
        ) -> pd.DataFrame:
        binary_data = self.binary_impute(data[binary_cols])
        numeric_data = self.numeric_impute(data[numeric_cols])
        category_data = self.category_impute(data[categorical_cols])

        for col in binary_cols:
            data[col] = binary_data[col]
        
        for col in numeric_cols:
            data[col] = numeric_data[col]

        for col in categorical_cols:
            data[col] = category_data[col]
        
        return data