import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# a={'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23]}
# print(np.random.choice(a['x']))
# print(np.random.random(1))
# exit()
class kmeans():
    def __init__(self,dataframe,k,centerid={}):
        self.dataframe=dataframe
        self.k=k
        if centerid=={}:
            self.centerid={i:[np.random.randint(min(self.dataframe['x']),max(self.dataframe['x'])),np.random.randint(min(self.dataframe['y']),max(self.dataframe['y']))] for i in range(self.k)}
        else:
            self.centerid=centerid
        self.colormap={0:'r',1:'g',2:'b'}

    def paint_color(self):
        for i in self.centerid.keys():
            self.dataframe['distance_from_{}'.format(i)]=np.sqrt((self.dataframe['x']-self.centerid[i][0])**2+(self.dataframe['y']-self.centerid[i][1])**2)
        distance_from_centroid_id = ['distance_from_{}'.format(i) for i in self.centerid.keys()]
        self.dataframe['closest'] = self.dataframe.loc[:, distance_from_centroid_id].idxmin(axis=1)
        self.dataframe['closest'] = self.dataframe['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        self.dataframe['color']=self.dataframe['closest'].map(lambda x:self.colormap[x])
        return self.dataframe

    def update(self):
        for i in self.centerid.keys():
            self.centerid[i][0]=np.mean(self.dataframe[self.dataframe['closest']==i]['x'])
            self.centerid[i][1] = np.mean(self.dataframe[self.dataframe['closest'] == i]['y'])
        return self.centerid

    def showmap(self):
        plt.scatter(self.dataframe['x'], self.dataframe['y'], color=self.dataframe['color'], alpha=0.5, edgecolor='k')
        for i in self.centerid.keys():
            plt.scatter(*self.centerid[i], color=self.colormap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

    def kmeans_fun(self):
        self.paint_color()
        while True:
            closest_centroids = self.dataframe['closest'].copy(deep=True)
            self.centroids = self.update()

            plt.scatter(self.dataframe['x'], self.dataframe['y'], color=self.dataframe['color'], alpha=0.5, edgecolor='k')
            for i in self.centerid.keys():
                plt.scatter(*self.centerid[i], color=self.colormap[i], linewidths=6)
            plt.xlim(0, 80)
            plt.ylim(0, 80)
            plt.show()

            self.dataframe = self.paint_color()

            if closest_centroids.equals(self.dataframe['closest']):
                break


df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

class kmeans_plus(kmeans):
    def __init__(self,dataframe,k,centerid=None,colormap={0:'r',1:'g',2:'b'}):
        self.dataframe=dataframe
        self.k=k
        r=np.random.randint(0,len(self.dataframe['x']))
        self.centerid={0:[self.dataframe['x'][r],self.dataframe['y'][r]]}
        self.dataframe['D']=(self.dataframe['x']-self.centerid[0][0])**2+(self.dataframe['y']-self.centerid[0][1])**2
        self.dataframe['probs']=self.dataframe['D']/self.dataframe['D'].sum()
        self.dataframe['cumprobs']=self.dataframe['probs'].cumsum()
        for j in range(1,self.k):
            rp = np.random.random(1)
            self.dataframe['D'] = (self.dataframe['x'] - self.centerid[j-1][0]) ** 2 + (
                    self.dataframe['y'] - self.centerid[j-1][1]) ** 2
            self.dataframe['probs'] = self.dataframe['D'] / self.dataframe['D'].sum()
            self.dataframe['cumprobs'] = self.dataframe['probs'].cumsum()
            for i,d in enumerate(self.dataframe['cumprobs']):
                if rp<d:
                    self.centerid[j]=[self.dataframe['x'][i],self.dataframe['y'][i]]
                    break
        self.colormap=colormap
        super(kmeans_plus, self).__init__(self.dataframe,self.k,self.centerid)

    def f(self):
        print(self.centerid)
        return self.centerid

test=kmeans_plus(df,3)
test.f()