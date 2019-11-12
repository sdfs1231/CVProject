import numpy as np
# import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math
dataframe=pd.DataFrame({'Outlook':['sunny','sunny','overcast','rainy','rainy','rainy','overcast','sunny','sunny','rainy','sunny','overcast','overcast','rainy'],
                        'Temperature':['hot','hot','hot','mild','cool','cool','cool','mild','cool','mild','mild','mild','hot','mild'],
                        'Humidity':['high','high','high','high','normal','normal','normal','high','normal','normal','normal','high','normal','high'],
                        'Windy':[0,1,0,0,0,1,1,0,0,0,1,1,0,1],
                        'play':[0,0,1,1,1,0,1,0,1,1,1,1,1,0]})

# print(dataframe.shape[0])
# print(dataframe.loc[(dataframe['play']==1)&(dataframe['Outlook']=='sunny')].shape[0])
# exit()

def imformation_entrophy(total_samples,samples):
    #total samples is a int,samples is a dict:{'condition':int,....}
    entropy=0
    for k in samples.keys():
        if samples[k]==0:
            continue
        entropy+=-samples[k]/total_samples*math.log(samples[k]/total_samples,math.e)
    return entropy

# target='play'
# total_samples=dataframe.shape[0]
# samples={'yes':sum(dataframe['play']==1),'no':sum(dataframe['play']==0)}
# print(samples['yes'])
# print(imformation_entrophy(target,total_samples,samples))

def conditional_entropy(conditions,feature,target,dataframe):
    #conditions must be dict

    c_entropy=0
    total_nums=sum(conditions.values())

    for c in conditions.keys():
        # print(c)
        samples={'yes':dataframe.loc[(dataframe[target]==1)&(dataframe[feature]==c)].shape[0],'no':dataframe.loc[(dataframe[target]==0)&(dataframe[feature]==c)].shape[0]}
        # print(samples)
        # print(conditions[c])
        c_entropy+=(conditions[c]/total_nums)*imformation_entrophy(conditions[c],samples)
    return c_entropy

def ratio_entropy(conditions,feature,target,dataframe):
    total_samples=dataframe.shape[0]
    samples = {'yes': sum(dataframe[target] == 1), 'no': sum(dataframe[target] == 0)}
    H=imformation_entrophy(total_samples,samples)
    h=conditional_entropy(conditions,feature,target,dataframe)
    return {feature:H-h}

conditions=dataframe['Outlook'].value_counts()
feature=conditions.name
conditions=dict(conditions)
print(ratio_entropy(conditions,feature,'play',dataframe))