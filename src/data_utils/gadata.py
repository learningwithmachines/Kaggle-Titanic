import numpy as np
import pickle
import dill
from typing import Tuple, List, ClassVar

# datafiles
featuresfile: str = './../../saves/features_data_dict.bin'
augdatafile: str = './../../saves/augmented_data_dict.bin'

# custom dataclass for convenience,
# loads data from a dictionary of numpy arrays and column labels
class GAData:
    '''
    Class object to store data, properties and methods needed for GA.
    '''
    def __init__(self, filepath: str, do_norm: bool=False) -> None:
        '''
        initialize GAData by loading, perform optionally normalize and cast data to float
        :param filepath: str, location of pre-pickled & compatible data dictionary
        :param do_norm: Bool, flag to perform np.linalg norm on train and test features.
        '''
        self.X_train, self.Y_train, self.X_test, self.features = self.load_dict(filepath)
        renorm: np.float64 = np.linalg.norm(np.vstack([self.X_train, self.X_test]))

        if do_norm:
            self.X_train = self.X_train.astype(np.float64) / renorm
            self.X_test = self.X_test.astype(np.float64) / renorm

        self.X_train: ClassVar = self.X_train.astype(np.float32)
        self.X_test: ClassVar = self.X_test.astype(np.float32)
        self.Y_train: ClassVar = self.Y_train.reshape(-1)
        self.features: ClassVar = tuple(self.features)

    @staticmethod
    def load_dict(filepath:str)->Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        '''
        method to load pickled data and unpack, and return values to class
        :param filepath: str, location of pre-pickled & compatible data dictionary
        :return: tuple[ndarray, ndarray, ndarray, list[str,]]
        '''
        with open(filepath, 'rb') as pfile:
            datafile = pickle.load(file=pfile)
            pfile.close()

        return datafile['X_train'], datafile['Y_train'], datafile['X_test'], datafile['features']

    @property
    def training_size(self) -> int:
        '''
        shape of training data
        :return: int
        '''
        return self.X_train.shape[0]

    @property
    def testing_size(self) -> int:
        '''
        shape of testing data
        :return: int
        '''
        return self.X_test.shape[0]

    @property
    def numfeatures(self) -> int:
        '''
        number of feature columns in training data
        :return: int
        '''
        return self.X_train.shape[1]

    def c_args(self) -> List[dict]:
        '''
        make **kwarg pairs from featurenames for pset renaming
        :return: list[dict]
        '''
        return [{f'ARG{ix}': name} for ix, name in enumerate(self.features)]

# dataclass instance, for testing
# data = GAData(filepath=featuresfile)