import importlib
import math
import queue

import pandas as pd

def get_replica(requests, response):
    a = 300
    b = -0.00002632
    c = 0
    replicas = -(requests * (b / math.log((response - c) / a)))
    return (replicas)


def get_response_time(requests, replicas):
    # a = 8.24018
    # b = -0.0003632
    # c = -47.27967

    a = 300
    b = -0.00002632
    c = 0

    response=math.floor(a * math.exp(-b * requests / replicas) + c)
    return response

df=pd.read_csv('estimate.csv')
df['response']=0
df['replica_val']=0
count=0
count1=0
for i in range(len(df)):
    df.loc[i,'response']=get_response_time(df['requests'][i],df['replicas'][i])
    df.loc[i,'replica_val']=get_replica(df['requests'][i],df['response'][i])
    if df['response'][i]>500:
        count+=1
    if df['replicas'][i]!=df['replica_val'][i]:
        count1+=1


# replicas和replicas_val不同的行，计总共有多少不同的行
print(count)
print(count1)


df.to_csv('estimate_val.csv',index=False)