from ast import Raise
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import column
from data_extraction import EEGProcess
from CWT import MRA
from SSA import SSA
import scipy as sc
from tqdm import tqdm

class TransformationTransformer( BaseEstimator, TransformerMixin):
    def __init__(self,transformation_type,fit_params):
        self.transformation_type = transformation_type
        self.params = {'cwt__f_int': 0.5, "cwt__n_c": 6, 'ssa_L': 71}
        for key,value in fit_params.items():
            self.params[key] = value
        print(f'\n {transformation_type} trasnsformation initialized....')

    def CWT(self,X_,f_int,n_c,fi=1,ff=20):
        k = int((ff-fi)/f_int)
        n = X_.shape[2]
        X_fit = np.zeros([X_.shape[0],X_.shape[1],k,n],dtype=np.float16)
        n_i = n_c-2
        n_f = n_c+2
        for i in tqdm(range(X_.shape[0])):
            for j in range(X_.shape[1]):
                channel_signal = X_[i,j]
                spect = MRA(channel_signal, fi,ff,f_int, n_i,n_f)
                for k_ in range(k):
                    for n_ in range(n):
                        X_fit[i,j,k_,n_] = spect[k_,n_]
        return X_fit

    def RAW(self,X):
        return X

    def SSA(self,X, L):
        '''Definimos la función que realizará la remoción de artefactos dado el umbral y agrupará las componentes'''
        def A_EXTR(signal,glist,l_list,v0=200):
            s_max = signal.max()
            s_list = np.sqrt(l_list)
            # Remoción de artefactos
            if s_max < v0:
                g_cut = np.delete(glist,0,axis=0)
                s_cut = np.delete(s_list,0)
            else: 
                g_cut = np.delete(glist,[0,1],axis=0)
                s_cut = np.delete(s_list,[0,1])
            #Agrupamiento de las componentes
            PC = []
            NP = []
            for i in range(len(s_cut)):
                for k in range(len(s_cut)-1):
                    aux_list = np.delete(s_cut,i)
                    ratio = abs(1- (aux_list[k]/s_cut[i]))
                if ratio < 0.05:
                    PC.append(g_cut[i])
                else:
                    NP.append(g_cut[i])
            
            return np.array(PC),np.array(NP)
        '''Definimos la función que encontrará la frecuencia máxima para cada grupo de componetes'''
        def Fmax(PC,NP):
            PC_fft = sc.fft.fft(PC)
            NP_fft = sc.fft.fft(NP)
            
            PC_fmax = []
            NP_fmax = []
            for i in range(len(PC)):
                N = len(PC_fft[i])
                T = 1/500
                freq = sc.fft.fftfreq(N,T)[1:N//2]
                f_max = np.round(freq[np.argmax(abs(PC_fft[i][1:N//2]))],2)
                PC_fmax.append(f_max)
            for i in range(len(NP)):
                N = len(NP_fft[i])
                freq = sc.fft.fftfreq(N,1/500)[1:N//2]
                f_max = np.round(freq[np.argmax(abs(NP_fft[i][1:N//2]))],2)
                NP_fmax.append(f_max)
            PC_fmax_s, NP_fmax_s = np.array(sorted(PC_fmax)),np.array(sorted(NP_fmax))

            return np.array(PC_fmax_s),np.array(NP_fmax_s)

        def GRP (PC,NP):
            cp_max,cnp_max = Fmax(PC,NP)
            '''Componentes Theta'''
            theta = []
            for i in range(4,8):
                for k in range(len(cp_max)):
                    if int(cp_max[k]) == i:
                        theta.append(PC[k])
                for j in range(len(cnp_max)):
                    if int(cnp_max[j]) == i:
                        theta.append(NP[j])
            '''Componentes alfa'''
            alfa = []
            for i in range(9,13):
                for k in range(len(cp_max)):
                    if int(cp_max[k]) == i:
                        alfa.append(PC[k])
                for j in range(len(cnp_max)):
                    if int(cnp_max[j]) == i:
                        alfa.append(NP[j])
            '''Componentes beta'''
            beta = []
            for i in range(14,31):
                for k in range(len(cp_max)):
                    if int(cp_max[k]) == i:
                        beta.append(PC[k])
                for j in range(len(cnp_max)):
                    if int(cnp_max[j]) == i:
                        beta.append(NP[j])
            '''Componentes gama'''
            gama = []
            for i in range(32,51):
                for k in range(len(cp_max)):
                    if int(cp_max[k]) == i:
                        gama.append(PC[k])
                for j in range(len(cnp_max)):
                    if int(cnp_max[j]) == i:
                        gama.append(NP[j])
            '''Componentes baja frecuencia'''
            bass=[]
            for i in range(0,3):
                for k in range(len(cp_max)):
                    if int(cp_max[k]) == i:
                        bass.append(PC[k])
                for j in range(len(cnp_max)):
                    if int(cnp_max[j]) == i:
                        bass.append(NP[j])
            '''Componentes de alta frecuencia'''
            treble = []
            limit = int(max(cp_max.max(),cnp_max.max()))+1
            for i in range(52,limit):
                for k in range(len(cp_max)):
                    if int(cp_max[k]) == i:
                        treble.append(PC[k])
                for j in range(len(cnp_max)):
                    if int(cnp_max[j]) == i:
                        treble.append(NP[j])
            return theta,alfa,beta,gama,bass,treble


        X_fit = np.zeros([X.shape[0],X.shape[1],3,X.shape[2]])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                channel_signal = X[i,j]
                lam, gklist, wMatrix = SSA(channel_signal,L)
                theta,alfa,beta,gama,low_f,high_f = GRP(A_EXTR(channel_signal,gklist,lam))
                components = [alfa,beta,gama]
                for k_ in range(3):
                    aggreg_comp = components[k_][0]
                    for n_ in range(X.shape[2]):
                        X_fit[i,j,k_,n_] = aggreg_comp[k_,n_]
        return X_fit

    def fit(self,X=None, y=None):
        """ 
        Given a transformation type, (RAW, CWT, SSA) a fit transform function is generated over the data

        """
        y  = X[1]
        X = X[0]
        self.y =y       
        if self.transformation_type == 'CWT':
            self.X = self.CWT(X, self.params['cwt__f_int'],self.params['cwt__n_c'] )
            print('CWT finished')
            return self
        elif self.transformation_type == 'SSA':
            self.X = self.SSA(X, self.params['ssa_L'])
            return self
        elif self.transformation_type == 'RAW':
            self.X = self.RAW(X)
            return self
    def transform(self, X=None,y=None):
        print('Transformation transform invoqued')
        return self.X, self.y

class ExtractionTransformer( BaseEstimator, TransformerMixin):
        def __init__(self,t0,t1,filter_channels):
            print('\n initialize extraction')
            self.t0 = t0
            self.t1 = t1
            self.filter_channels = filter_channels
            pass
        
        def fit(self, X_=None, y_=None):
            print("Importing data...\n")
            data = EEGProcess().extraction(s1=(2,22),s2=(20,37),s3=(38,60),t0_s=self.t0, t1_s=self.t1,states=[1,2,3])
            X = data.drop(columns = ['Control','X5'])
            y = data['Control']
            X_copy = X.copy()
            for key in X.columns:
                X_copy[key] = X[key].apply(lambda x: len(x))
            est_len = int((self.t1- self.t0)*200)
            X.drop(X_copy[X_copy['O1']!=est_len].index, inplace=True)
            y.drop(X_copy[X_copy['O1']!=est_len].index, inplace=True)
            print('Converting to matrix... \n')
            if len(self.filter_channels)>=1:
                df_mat = np.zeros([X.values.shape[0],len(self.filter_channels),est_len])
                X = X[self.filter_channels]
                for i in tqdm(range(X.values.shape[0])):
                    for j in range(len(self.filter_channels)):
                        for k in range(est_len):
                            df_mat[i][j][k] =  X.values[i][j][k]
            else:
                df_mat = np.zeros([X.values.shape[0],X.values.shape[1],est_len])
                for i in tqdm(range(X.values.shape[0])):
                    for j in range(X.values.shape[1]):
                        for k in range(est_len):
                            df_mat[i][j][k] =  X.values[i][j][k]
            print(f'Resulting X matrix shape {df_mat.shape}\n')
            self.X = df_mat
            self.y = y.values
            return self

        def transform(self, X=None,y=None):
            print('Extract Transform invoqued ')
            return self.X,self.y

class preprocessing_pipeline():

    def __init__(self, t0,t1, transformation_type,transformation_parameters, filter_channels):
        #Store parameters used
        self.t0 = t0
        self.t1 = t1
        self.transformation_type = transformation_type
        self.transformation_parameters = transformation_parameters
        self.filter_channels = filter_channels
        pass

    def pipeline(self):
        scaler = StandardScaler()
        extraction = ExtractionTransformer(t0=self.t0,t1=self.t1,filter_channels=self.filter_channels)
        transformation = TransformationTransformer(transformation_type=self.transformation_type,fit_params=self.transformation_parameters)
        print('Trying to build pipeline')
        stage_1 = Pipeline([('extraction',extraction),('transformation',transformation)])
        return stage_1



if __name__ =='__main__':
    dummy_x = 1
    dummy_y = 1
    stage = preprocessing_pipeline( t0=-0.5, t1=1.0, transformation_type='CWT', transformation_parameters={'cwt__n_c': 5, 'cwt__f_int':0.5}).pipeline()
    X, y= stage.fit_transform(dummy_x,dummy_y)
    print(X)
    print(y)