

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["figure.figsize"] = (14, 6)

##### Create the colormap

## Color map samples  

data = [
    (0, tuple(np.array([0, 0, 0])/255)),
    (0.0316227766, tuple(np.array([0, 0, 0])/255)),
    (0.04466835922, tuple(np.array([7, 5, 10])/255)),
    (0.06309573445,tuple(np.array([23, 9, 32])/255)),
    (0.07079457844, tuple(np.array([31, 8, 54])/255)),
    (0.07943282347, tuple(np.array([43, 5, 66])/255)),
    (0.08912509381,tuple(np.array([64, 11, 83])/255)),
    (0.1, tuple(np.array([90, 18, 94])/255)),
    (0.1122018454, tuple(np.array([120, 12, 97])/255)),
    (0.1258925412, tuple(np.array([157, 0, 81])/255)),    
    (0.1412537545, tuple(np.array([183, 0, 56])/255)),
    (0.1584893192, tuple(np.array([207, 0, 23])/255)),
    (0.1995262315,tuple(np.array([255, 76, 9])/255)),
    (0.2238721139, tuple(np.array([255, 137, 41])/255)),
    (0.2511886432, tuple(np.array([255, 183, 75])/255)),
    (0.316227766, tuple(np.array([255, 232, 92])/255)),
    (0.3548133892, tuple(np.array([254, 248, 90])/255)),
    (0.5011872336, tuple(np.array([219, 229, 80])/255)),
    (1, tuple(np.array([219, 229, 80])/255)),
     ]

# Create a colormap with using interpolation
amap = LinearSegmentedColormap.from_list('custom_colormap', data, N=256, gamma=1)

# Display the color map

r=[]
g=[]
b=[]
x=[]
for i in range(19):
    rgba=data[i][1]
    r.append(rgba[0])
    g.append(rgba[1])
    b.append(rgba[2])
    x.append(data[i][0])

plt.figure()
# plt.plot(20*np.log10(x),r,color="r")  
# plt.plot(20*np.log10(x),g,color="g")  
# plt.plot(20*np.log10(x),b,color="b")

plt.plot(x,r,color="r")  
plt.plot(x,g,color="g")  
plt.plot(x,b,color="b")
plt.grid("minor")



gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack((gradient, gradient))
plt.figure()
plt.imshow(gradient, aspect='auto', cmap=amap,vmin=0, vmax=1)


#%%

file_path=r"C:\Dropbox\Code\1.wav"

n_fft=1024
hop_length=n_fft//8
audition_dymanic_range=90  ## Dymanic range is not same as Audition in the small values



audio,sr=librosa.load(file_path,sr=None)
stft_calc =np.abs(librosa.stft(audio,n_fft=n_fft,hop_length=hop_length, window='hann'))

fft_norm=(n_fft/4)
stft_dec_norm=stft_calc/fft_norm


factor=24/(audition_dymanic_range*0.5)
stft_dB=(20*np.log10(stft_dec_norm))
stft_dB_Dymanic=stft_dB*factor
stft_dec=10**(stft_dB_Dymanic/20)

stft_dec=(stft_dec+0.06)/(1.06)
plt.figure()
plt.imshow(stft_dec,origin='lower',aspect='auto',cmap=amap,vmin=0, vmax= 1)
plt.title("amap") 
plt.axis('off') 








