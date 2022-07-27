import os
import pandas as pd
import os
import numpy as np
import glob
from tqdm import tqdm

class EEGProcess:
    """
    Object created for the extraction, segmentation, tracing and classification of 
    the multiple session datasets
    """
    def __init__(self):
        path = os.getcwd() + '\\CLA'
        self.file_names = []
        for file in os.listdir(path):
            if file.endswith('.csv'):
                self.file_names.append(file[3:-4])
        path = os.getcwd() + '\\CLA\\*.csv'
        file_direction = glob.glob(path)
        self.data = {}
        columns = ['Control','Fp1','Fp2','F3','F4','C3','C4','P3',
           'P4','O1','O2','A1','A2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','X5']
        for i in range(len(file_direction)):
            self.data[self.file_names[i]] = pd.read_csv(file_direction[i],header=0, names= columns)

        pass

    def segmentator(self,file,s1,s2,s3):
        """
        This function recieves a complete patient session and segments it on 3 parts in order to eliminate the resting between 
        the paradigms
        file (Pandas DF) := DataFrame of the complete session 
        si (tuple) := Tuple of the initial and final time markers for each segment, in minutes
        Observed values (approx) s1 = (3,20), s2 = (21,36) , s3 = (40,55)
        """
        sf = 200 #Hz
        minutes = lambda s: round(s*60*sf)
        segments = {}
        segments['s1'] = file.iloc[minutes(s1[0]):minutes(s1[1])]
        segments['s2'] = file.iloc[minutes(s2[0]):minutes(s2[1])]
        segments['s3'] = file.iloc[minutes(s3[0]):minutes(s3[1])]
        return segments
    def traces(self,segment,ctrl_val, t0_s, t1_s):
        """
        This function recieves a DF segment and a Control Value to get and obtains a time-windowed DF around the 
        stimulii according to the t0 and t1 values
        segment (Pandas DF) := Session segment, obtained from the sementator function
        ctr_val (int) := Control "marker" code to filter.
        t0_s (float) := +/- seconds for the starting point of the window according to the moment the stimulli started
        t1_S (float) _= +/- seconds for the ending point of the window according to the moment the stimulli ended
        """
        # Useful function to get the correct order of appearance of the sign change (activation/deactivation of the signal)
        def gen_series(arr,odd=True):
            n = len(arr)
            if odd:
                odds = []
                for i in range(0,n):
                    if i %2 !=0:
                        odds.append(i)
                return odds
            else:
                even = []
                for i in range(0,n):
                    if i%2 ==0:
                        even.append(i)
                return even
        ctrl = segment['Control'].copy().values    #Get the control vector of the segment
        ctrl[np.where(ctrl!=ctrl_val)] =0    #Filter the other states 
        sign = np.sign(ctrl)
        sign_change = ((np.roll(sign,1)-sign) !=0).astype(int)    #Get the sign change in the control vector
        sign_change[0] = 0    #Fix the sign change in the first element
        start_indexes = np.where(sign_change==1)[0][gen_series(np.where(sign_change==1)[0],odd=False)]    #Set the indexes where stimulii started
        # end_indexes = np.where(sign_change==1)[0][gen_series(np.where(sign_change==1)[0],odd=True)]
        traces = []
        seconds = lambda x: int(x*200)    #Transform the seconds to number of sampligs for a 200Hz samplig rate
        for i in range(len(start_indexes)):
            try:
                t0 = seconds(t0_s)
                t1 = seconds(t1_s)
                traces.append(segment.iloc[start_indexes[i]+t0:start_indexes[i]+t1])    #Add the windowed DF to a list
                # print(len(segment.iloc[start_indexes[i]+t0:end_indexes[i]+t1]))
            except IndexError:
                print('Splitted state, ommiting trace')

        return (traces,ctrl_val)    #Return the list of windowed DF and its corresponding label

    def dict_list_appender(self,traces,df_dict):
        """
        This function recieves a tuple of traces and Control Values and updates a Final dictionary for the 
        segmented-traced and classified data. 
        traces (tuple) := Tuple containing the list of traced windowed-DFs and it's corresponding label
        df_dict (dictionary) := Dictionary where the data will be stored
        """
        for trace in traces[0]:    #Cycle arround the obtained traces
            for key in df_dict.keys():    #Cycle through the dictionary keys, they must match
                if key == 'Control':
                    df_dict[key].append(traces[1])    # The Control label is defined
                else:
                    df_dict[key].append(trace[key].values)    #Each channel is stored as an array
        return
    
    def extraction(self,s1,s2,s3,t0_s,t1_s,states):
        """
        This function cycles around all the files and extracts the data and appends it to a final Pandas DF  
        si (tuple) := Tuple of the initial and final time markers for each segment, in minutes
        t0_s (float) := +/- seconds for the starting point of the window according to the moment the stimulli started
        t1_S (float) := +/- seconds for the ending point of the window according to the moment the stimulli ended
        states (list)  := List of the marked values of the mental states
        """
        self.df_dict_ = {'Control':[], 'Fp1':[], 'Fp2':[], 'F3':[], 'F4':[], 'C3':[],
         'C4':[], 'P3':[], 'P4':[], 'O1':[], 'O2':[],'A1':[], 'A2':[], 'F7':[], 'F8':[], 'T3':[], 'T4':[], 'T5':[], 'T6':[], 'Fz':[], 'Cz':[], 'Pz':[], 'X5':[]}
        for file in tqdm(self.file_names):
            segments = self.segmentator(self.data[file],s1,s2,s3)
            for key,segment in segments.items():
                for value in states:
                    self.dict_list_appender(
                        self.traces(segment, value, t0_s=t0_s,t1_s=t1_s), self.df_dict_
                    )
        self.df_dict = pd.DataFrame.from_dict(self.df_dict_)
        return self.df_dict
                    

       

