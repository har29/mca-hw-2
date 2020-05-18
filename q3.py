import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.io import wavfile
from scipy import signal
from sklearn.svm import LinearSVC
from joblib import load,dump
from sklearn.metrics import accuracy_score

x = []
y = []

def get_train_data():
    path = os.path.join("validation",'zero')
    print(0)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,0)

    path = os.path.join("validation",'one')
    print(1)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,1)

    path = os.path.join("validation",'two')
    print(2)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,2)

    path = os.path.join("validation",'three')
    print(3)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,3)

    path = os.path.join("validation",'four')
    print(4)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,4)

    path = os.path.join("validation",'five')
    print(5)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,5)

    path = os.path.join("validation",'six')
    print(6)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,6)

    path = os.path.join("validation",'seven')
    print(7)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,7)

    path = os.path.join("validation",'eight')
    print(8)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,8)

    path = os.path.join("validation",'nine')
    print(9)
    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:	
        pass
    add_audios(path,files,9)

    x_train = np.array(x)
    y_train = np.array(y)

    return x_train,y_train


def add_audios(path,files,digit):
    for audio in files:
        samplingFrequency, signalData = wavfile.read(os.path.join(path,audio))
        frequencies, times, spectrogram = signal.spectrogram(signalData,fs = 1000)
        spectrogram = spectrogram.flatten()
        if (len(spectrogram)==9159):
            x.append(np.array(spectrogram))
            y.append(digit)


def train(x_train,y_train):

    classifier = LinearSVC(verbose=1,max_iter = 100)
    classifier.fit(x_train,y_train)
    dump(classifier,'cls.joblib')


def test(x_test,y_test):
    classifier = load('cls.joblib')
    y = classifier.predict(x_test)
    ac = accuracy_score(y,y_test)
    print("Accuracy Score",ac)


#x_train,y_train = get_train_data()
#train(x_train,y_train)

x_test,y_test = get_train_data()
print(len(x_test))
test(x_test,y_test)