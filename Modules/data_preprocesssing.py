from data_extraction import *
from datap_pipe import * 
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from CWT import MRA
from tqdm import tqdm
import tensorflow as tf
import multiprocessing
import os 
class EEG_preparation:
    def __init__(self) -> None:
        pass
    
    def SSA_fit(self,X, L,comp_sel_m):
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
            if len(PC) == 0:
                PC_ = np.array([0])
            else:
                PC_fft = np.fft.fft(PC)
                PC_fmax = []
                for i in range(len(PC)):
                    N = len(PC_fft[i])
                    T = 1/500
                    freq = np.fft.fftfreq(N,T)[1:N//2]
                    f_max = np.round(freq[np.argmax(abs(PC_fft[i][1:N//2]))],2)
                    PC_fmax.append(f_max)
                PC_fmax_s = np.array(sorted(PC_fmax))
                PC_ = np.array(PC_fmax_s)
            if len(NP) == 0:
                NPC_ = np.array(0)
            else:
                NP_fft = np.fft.fft(NP)
                NP_fmax = []
                for i in range(len(NP)):
                    N = len(NP_fft[i])
                    freq = np.fft.fftfreq(N,1/500)[1:N//2]
                    f_max = np.round(freq[np.argmax(abs(NP_fft[i][1:N//2]))],2)
                    NP_fmax.append(f_max)
                NP_fmax_s = np.array(sorted(NP_fmax))
                NPC_ = np.array(NP_fmax_s)
            return PC_,NPC_

        def GRP (PC,NP,comp_sel=comp_sel_m):
            cp_max,cnp_max = Fmax(PC,NP)
            '''Componentes Theta'''
            theta = []
            if 'theta' in comp_sel:
                for i in range(4,8):
                    for k in range(len(cp_max)):
                        if int(cp_max[k]) == i:
                            theta.append(PC[k])
                    for j in range(len(cnp_max)):
                        if int(cnp_max[j]) == i:
                            theta.append(NP[j])
            '''Componentes alfa'''
            alfa = []
            if ('alfa' or 'alpha') in comp_sel:
                for i in range(9,13):
                    for k in range(len(cp_max)):
                        if int(cp_max[k]) == i:
                            alfa.append(PC[k])
                    for j in range(len(cnp_max)):
                        if int(cnp_max[j]) == i:
                            alfa.append(NP[j])
            '''Componentes beta'''
            beta = []
            if 'beta' in comp_sel:
                for i in range(14,31):
                    for k in range(len(cp_max)):
                        if int(cp_max[k]) == i:
                            beta.append(PC[k])
                    for j in range(len(cnp_max)):
                        if int(cnp_max[j]) == i:
                            beta.append(NP[j])
            '''Componentes gama'''
            gama = []
            if ('gama' or 'gamma') in comp_sel:
                for i in range(32,51):
                    for k in range(len(cp_max)):
                        if int(cp_max[k]) == i:
                            gama.append(PC[k])
                    for j in range(len(cnp_max)):
                        if int(cnp_max[j]) == i:
                            gama.append(NP[j])
            '''Componentes baja frecuencia'''
            bass=[]
            if 'low_f' in comp_sel:
                for i in range(1,3):
                    for k in range(len(cp_max)):
                        if int(cp_max[k]) == i:
                            bass.append(PC[k])
                    for j in range(len(cnp_max)):
                        if int(cnp_max[j]) == i:
                            bass.append(NP[j])
            '''Componentes de alta frecuencia'''
            treble = []
            if 'high_f' in comp_sel:
                limit = int(max(cp_max.max(),cnp_max.max()))+1
                for i in range(52,limit):
                    for k in range(len(cp_max)):
                        if int(cp_max[k]) == i:
                            treble.append(PC[k])
                    for j in range(len(cnp_max)):
                        if int(cnp_max[j]) == i:
                            treble.append(NP[j])
            return theta,alfa,beta,gama,bass,treble


        X_fit = np.zeros([X.shape[0],X.shape[1],X.shape[2]])
        for i in tqdm(range(X.shape[0])):
            for j in range(X.shape[1]):
                channel_signal = X[i,j]
                lam, gklist, wMatrix = SSA(channel_signal,L)
                c_per,c_nper = A_EXTR(channel_signal,gklist,lam)
                theta,alfa,beta,gama,low_f,high_f = GRP(c_per,c_nper)
                comp = []
                for c in comp_sel_m:
                    if c == ('alfa'or 'alpha'):
                        comp.append(alfa)
                    if c=='beta':
                        comp.append(beta)
                    if c==('gamma' or 'gama'):
                        comp.append(gama)
                    if c=='low_f':
                        comp.append(low_f)
                    if c=='high_f':
                        comp.append(high_f)
                    if c=='theta':
                        comp.append(theta)
                comp_plus = np.zeros(X.shape[2])
                for k_ in range(len(comp)):
                    if len(comp[k_]) == 0:
                        comp[k_] = np.zeros(X.shape[2])
                    comp_plus += comp[k_][0]
                for n_ in range(X.shape[2]):
                    X_fit[i,j,n_] = comp_plus[n_]
        return X_fit


    def extract(self,t0,t1,chan):
        extraction = ExtractionTransformer(t0=t0, t1=t1, filter_channels=chan)
        X_dat, y_dat = extraction.fit_transform(1,1)
        return X_dat, y_dat

    def sampling_split(self,X,y,sample_size,split_ratio ):
        sample = np.random.choice(len(X), size=int(len(X)/sample_size))
        n_data = len(sample)
        split_ratio = 0.2
        test_ratio = int(split_ratio*n_data)
        train_ratio = int((1.0-split_ratio)*n_data)
        y_res = tf.keras.utils.to_categorical(y-1)

        data_ = X[sample]
        data_y_ = y_res[sample]

        X_train = data_[0:train_ratio]
        X_test  = data_[train_ratio:]

        y_train=data_y_[0:train_ratio]
        y_test=data_y_[train_ratio:]
        return X_train,y_train, X_test,y_test

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

    def normalizer(self,X,move_ch=True):
        X_ = X.astype('float32')
        if move_ch:
            X_ = np.moveaxis(X_,1,3)
        X_ = (X_ -X_.mean(axis=(0,1,2),keepdims=True)) / X_.std(axis=(0,1,2),keepdims=True)
        return X_

    def batch_generator(self,X,y):
        X_batch = {}
        y_batch = {}
        cores = os.cpu_count()
        batch_size = int(len(X)/cores)
        for i in range(cores):
            X_batch[i] = X[i*batch_size:(i+1)*batch_size]
            y_batch[i] = y[i*batch_size:(i+1)*batch_size]
        return X_batch,y_batch

    def nan_manager(self,X,y):
        sequence = []
        for i in tqdm(range(X.shape[0])):
            if np.isnan(X[i,:,:,:]).any() :
                sequence.append(False)
            else:
                sequence.append(True)
        X_del = X[sequence]
        y_del = y[sequence]
        print(f'After the cleaning {(len(X)-len(X_del))}  of rows were dropped')
        return X_del,y_del

    def transformation(self,X_batch,y_batch,n_c,f_int,fi,ff,id):
        X_cwt = self.CWT(X_batch,n_c=n_c,f_int=f_int,fi=fi,ff=ff)
        X_cwt = self.normalizer(X_cwt)
        X_cwt = np.expand_dims(X_cwt, axis=4)
        X_cwt,y_batch = self.nan_manager(X_cwt,y_batch)
        self.return_dict[id] = (X_cwt,y_batch)
        return
    def image_gen(self,X_batch,y_batch,n_c,f_int,fi,ff,id):
        X_batch = self.normalizer(X_batch,move_ch=False)
        X_batch = np.expand_dims(X_batch, axis=3)
        X_batch,y_batch = self.nan_manager(X_batch,y_batch)
        X_batch = np.moveaxis(X_batch,1,2)
        X_batch = np.squeeze(X_batch,axis=3)
        self.return_dict[id] = (X_batch,y_batch)
        return
    def ssa_transformation(self,X_batch,y_batch,comp_sel_m,id):
        X_ssa = self.SSA_fit(X_batch,L=60,comp_sel_m=comp_sel_m)
        X_ssa = self.normalizer(X_ssa,move_ch=False)
        X_ssa = np.expand_dims(X_ssa, axis=3)
        X_ssa,y_batch = self.nan_manager(X_ssa,y_batch)
        X_ssa = np.moveaxis(X_ssa,1,2)
        X_ssa = np.squeeze(X_ssa,axis=3)
        self.return_dict[id] = (X_ssa,y_batch)


    def preprocessing(self,d_name,t0,t1,chan,comp_sel_m,sample_size,target=ssa_transformation,split_ratio=0.3):
        if __name__ == '__main__':
            print('Extracting data...')
            X_dat,y_dat = self.extract(t0=t0,t1=t1,chan=chan)
            X_train,y_train,X_test,y_test = self.sampling_split(X_dat,y_dat,sample_size=sample_size,split_ratio=split_ratio)
            print('............. \n')
            print('Train preprocessing started...\n')
            manager = multiprocessing.Manager()
            self.return_dict = manager.dict()
            print('\t Creating batches...')
            train_process_list = []
            X_batches,y_batches = self.batch_generator(X_train,y_train)
            print('\t Creating instances...')
            for batch_num in X_batches.keys():
                processs =   multiprocessing.Process(target=target, args=(self,X_batches[batch_num],y_batches[batch_num],comp_sel_m,batch_num,))
                train_process_list.append(processs)
            print('\t Executing jobs...')
            for p in train_process_list:
                p.start()
            print('\t Joining...')
            for p in train_process_list:
                p.join()
            X_sequence = []
            y_sequence = []
            for key,val in self.return_dict.items():
                X_sequence.append(val[0])
                y_sequence.append(val[1])
            X_train_results = np.concatenate(X_sequence, axis=0)
            y_train_results = np.concatenate(y_sequence, axis=0)
            # X_train_results = np.array(X_sequence)
            # y_train_results = np.array(y_sequence)
            ch_name = '-'.join(chan)
            if not(os.path.exists(f'Pre-Dat/{d_name}')):
                os.mkdir(f'Pre-Dat/{d_name}')
            if not(os.path.exists(f'Pre-Dat/{d_name}/{ch_name}')):
                os.mkdir(f'Pre-Dat/{d_name}/{ch_name}')
            np.save(f'Pre-Dat/{d_name}/{ch_name}/X_train',X_train_results)
            np.save(f'Pre-Dat/{d_name}/{ch_name}/y_train',y_train_results)

            print('\t X train shape:',X_train_results.shape)
            print('\t y train shape:',y_train_results.shape)
            print('\t Training preprocessing complete!')
            print('............. \n')
            
            print('Test preprocessing started... \n')
            manager = multiprocessing.Manager()
            self.return_dict = manager.dict()
            print('\t Creating batches...')
            test_process_list = []
            X_batches,y_batches = self.batch_generator(X_test,y_test)
            print('\t Creating instances...')
            for batch_num in X_batches.keys():
                processs = multiprocessing.Process(target=target, args=(self,X_batches[batch_num],y_batches[batch_num],comp_sel_m,batch_num,))
                test_process_list.append(processs)
            print('\t Executing jobs...')
            for p in test_process_list:
                p.start()
            print('\t Joining...')
            for p in test_process_list:
                p.join()
            X_sequence = []
            y_sequence = []
            for key,val in self.return_dict.items():
                X_sequence.append(val[0])
                y_sequence.append(val[1])
            X_test_results = np.concatenate(X_sequence, axis=0)
            y_test_results = np.concatenate(y_sequence, axis=0)
            if not(os.path.exists(f'Pre-Dat/{d_name}')):
                os.mkdir(f'Pre-Dat/{d_name}')
            if not(os.path.exists(f'Pre-Dat/{d_name}/{ch_name}')):
                os.mkdir(f'Pre-Dat/{d_name}/{ch_name}')
            np.save(f'Pre-Dat/{d_name}/{ch_name}/X_test',X_test_results)
            np.save(f'Pre-Dat/{d_name}/{ch_name}/y_test',y_test_results)

            print('\t X test shape:',X_test_results.shape)
            print('\t y test shape:',y_test_results.shape)
            print('\t Test preprocessing complete!')
            print('............. \n')

prep =EEG_preparation()
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
       'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
comb = ['Fp1','F3','P4','O2','A1','T3']
# prep.preprocessing(t0=-0.25,t1=0.75,chan=channels,n_c=3,f_int=0.1,fi=1,ff=10,sample_size=1)
prep.preprocessing(d_name='SSA',t0=-0.25,t1=0.75,chan=comb,comp_sel_m=['alfa','theta'],sample_size=1)
