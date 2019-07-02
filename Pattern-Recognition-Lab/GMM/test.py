import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

"""
#path to training data
source   = "development_set/"   
modelpath = "speaker_models/"
test_file = "development_set_test.txt"        
file_paths = open(test_file,'r')

"""
#path to training data
source   = "SampleData/"   

#path where training speakers will be savecopy_regd
modelpath = "Speakers_models/"

gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
print(gmm_files)
#Load the Gaussian gender Models
models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

error = 0
total_sample = 0.0

test_file = "testSamplePath.txt"        
file_paths = open(test_file,'r')
# Read the test directory and get the list of test audio files 
for path in file_paths:
	total_sample = total_sample + 1.0
	path = path.strip()   
	print("Testing Audio : " + str(path))
	sr,audio = read(source + path)
	vector   = extract_features(audio,sr)
	log_likelihood = np.zeros(len(models)) 
	for i in range(len(models)):
		gmm    = models[i]  #checking with each model one by one
		scores = np.array(gmm.score(vector))
		log_likelihood[i] = scores.sum()

	winner = np.argmax(log_likelihood)
	print("\tdetected as - " + speakers[winner])
	checker_name = path.split("_")[0]
	if speakers[winner] != checker_name:
		error = error + 1
		time.sleep(1.0)

print("erro:\t" + str(error) + "\ttotal:\t" + str(total_sample))
accuracy = ((total_sample - error) / total_sample) * 100

print("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : " + str(accuracy) + "%")
print("Hurrey ! Speaker identified. Mission Accomplished Successfully. ")
