#script organizes passivated detectors by each strip's LC.
#Written by Evan Martinez
#Last Edited: 11/19/2019


import matplotlib.pyplot as plt
#%matplotlib inline
import glob, os, sys, re
import csv
import pylab as pl
import io
import ntpath
import pandas as pd
import numpy as np
import sqlite3
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
#from customplot import *



#path = 'C:/Users/evanm/Desktop/Local Data/CUTestingLocal/Sh0161/IVCurves/2019/06-12-2019/'
#rootdir = 'C:/Program Files (x86)/National Instruments/LabVIEW 2018/instr.lib'
#textfiles = []
LC250 = []
extensions = ('.lvm')
filenames = []
dictionary = {}
dicLCA = {}
dicLCB = {}
dicLCC = {}
dicLCD = {}
dicLCE = {}
dicLCF = {}
dicLCG = {}
dicLCH = {}
dicLCR = {}
basename1 = []
DetName = []

LCAlist = []
LCBlist = []
LCClist = []
LCDlist = []
LCElist = []
LCFlist = []
LCGlist = []
LCHlist = []
LCRlist = []

#list with just detector names "Sh****"
z2 = []
freezerlist = []

"""
###############################Pulling xray data########################
#The x-ray file is on a local drive, so change the file directory in the open() function 
#to point to the csv.
with open('summary_32dets_Evan.csv') as f:
	XR_A_4us = []
	XR_B_4us = []
	XR_C_4us = []
	XR_D_4us = []
	XR_E_4us = []
	XR_F_4us = []
	XR_G_4us = []
	XR_H_4us = []

	
	for line in f:
		line = line.split(',')
		# print (line[0])
		pattern1 = re.compile(r'Sh\d\d\d\d')
		if re.match(pattern1, line[0]):
			if line[1]=='A':
				#XR_A_4us = [x for x in line[8]]
				XR_A_4us.append(line[8])
			if line[1]=='B':
				#XR_A_4us = [x for x in line[8]]
				XR_B_4us.append(line[8])
			if line[1]=='C':
				#XR_A_4us = [x for x in line[8]]
				XR_C_4us.append(line[8])
			if line[1]=='D':
				#XR_A_4us = [x for x in line[8]]
				XR_D_4us.append(line[8])
			if line[1]=='E':
				#XR_A_4us = [x for x in line[8]]
				XR_E_4us.append(line[8])
			if line[1]=='F':
				#XR_A_4us = [x for x in line[8]]
				XR_F_4us.append(line[8])
			if line[1]=='G':
				#XR_A_4us = [x for x in line[8]]
				XR_G_4us.append(line[8])
			if line[1]=='H':
				#XR_A_4us = [x for x in line[8]]
				XR_H_4us.append(line[8])
# print (XR_D_4us, XR_H_4us, XR_G_4us, XR_G_4us)
##########end xray data extraction###############
"""
pathDropbox = 'C:/Users/evanm/Dropbox/CUTesting (1)/**/*.lvm'
pathLocal = 'C:/Users/evanm/Desktop/GAPS/Local Data/**/*.lvm'

#Pull funtion for leakage current of ALL detector strips
def GetLCat250():
	with open(absfname, 'r', errors = 'ignore') as f:
		for line in f:
			line = line.split(',')
			#print (line)
			if float(line[1]) == 250:
				LC = line[2][:5]
				LC250.append(float(line[2][:5]))
				# print(LC)
				return LC


pathDropbox = '/Users/evanm/Dropbox/CUTesting2/**/*.lvm'

count = 0
newdet = False
tempdet = ""
tempdate = ""
for fname in glob.glob(pathDropbox, recursive = True):
	foundA, foundB, foundC, foundD, foundE, foundF, foundG, foundH, foundR = False, False, False, False,False,False, False, False, False
	#print (fname)
	absfname = os.path.abspath(fname) #fixes issue of having both '/' and '\' in fname
	count += 1		#checking number of fnames files
	#Splits up filename by back slash
	x = absfname.split('/')
	#print (x)

	#Creating regex patterns
	pattern1 = re.compile(r'Sh\d\d\d\d')
	pattern2 = re.compile(r'_PS_Pass_[A-R]_\.lvm')
	pattern3 = re.compile(r'\d\d-\d\d-\d\d\d\d')

	#applying regex patterns to fnames
	sh = re.match(pattern1, x[5])
	letter = re.match(pattern2, x[-1])
	if re.match(pattern1, x[-5]) and re.match(pattern2, x[-1]) and re.match(pattern3, x[-2]):
		z = x[-2].split('-')
		day = z[1]
		filenames.append(x[5] + ' ' + day)
		# print (sh, letter)
		y = x[-1].split('_')

		#sorting algorithm that pulls leakage current data from dropbox files and appends to their appropriate lists
		try:
			if '_A_' in absfname:
				LCA = GetLCat250()
				fnameA = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCA[fnameA]=LCA
				LCAlist.append(GetLCat250())
				foundA = True	
			elif '_B_' in absfname:
				LCB = GetLCat250()
				fnameB = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCB[fnameB]=LCB
				LCBlist.append(GetLCat250())
			elif '_C_' in absfname:
				LCC = GetLCat250()
				fnameC = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCC[fnameC]=LCC
				LCClist.append(GetLCat250())
			elif '_D_' in absfname:
				LCD = GetLCat250()
				fnameD = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCD[fnameD]=LCD
				LCDlist.append(GetLCat250())
			elif '_E_' in absfname:
				LCE = GetLCat250()
				fnameE = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCE[fnameE]=LCE
				LCElist.append(GetLCat250())
				#print (fnameE)
			elif '_F_' in absfname:
				LCF = GetLCat250()
				fnameF = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCF[fnameF]=LCF
				LCFlist.append(GetLCat250())
				#print (fnameF)
			elif '_G_' in absfname:
				LCG = GetLCat250()
				fnameG = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCG[fnameG]=LCG
				LCGlist.append(GetLCat250())
			elif '_H_' in absfname:
				LCH = GetLCat250()
				fnameH = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCH[fnameH]=LCH
				LCHlist.append(GetLCat250())
			elif '_R_' in absfname:
				LCR = GetLCat250()
				fnameR = x[6] + ': ' + y[2] + '_' + y[3]
				dicLCR[fnameR]=LCR
				LCRlist.append(GetLCat250())
	
		except IndexError:
			pass
	elif re.match('Freezer', x[-2]):
		freezerlist.append(x[6])

freezerlist = list(set(freezerlist))
#print('The following detectors have freezer data: ', freezerlist)

#print ('the number of fnames is: ', count)
#print (dictionary)

#Remove duplicates from list
filenamesSingle = []
[filenamesSingle.append(x) for x in filenames  if x not in filenamesSingle]

FullName = []
for i in filenamesSingle:
	c = i + " Pass"
	FullName.append(c)
#print (FullName)

# print(LCAlist)
Float_LCAlist = np.array([float(i) if i is not None else -5000 for i in LCAlist])
Float_LCBlist = np.array([float(i) if i is not None else -5000 for i in LCBlist])
Float_LCClist = np.array([float(i) if i is not None else -5000 for i in LCClist])
Float_LCDlist = np.array([float(i) if i is not None else -5000 for i in LCDlist])
Float_LCElist = np.array([float(i) if i is not None else -5000 for i in LCElist])
Float_LCFlist = np.array([float(i) if i is not None else -5000 for i in LCFlist])
Float_LCGlist = np.array([float(i) if i is not None else -5000 for i in LCGlist])
Float_LCHlist = np.array([float(i) if i is not None else -5000 for i in LCHlist])
Float_LCRlist = np.array([float(i) if i is not None else -5000 for i in LCRlist])


print(Float_LCAlist.size,
Float_LCBlist.size,
Float_LCClist.size,
Float_LCDlist.size,
Float_LCElist.size,
Float_LCFlist.size,
Float_LCGlist.size,
Float_LCHlist.size,
Float_LCRlist.size)

# df = pd.DataFrame()
# df['RT_A'] = Float_LCAlist 
# df['RT_B'] = Float_LCBlist 

# Print (df)

"""

#creating dataframe of detector passivated data
seriesName = pd.Series(FullName)

# seriesA = pd.Series(LCAlist).convert_objects(convert_numeric=True)
# seriesB = pd.Series(LCBlist).convert_objects(convert_numeric=True)
# seriesC = pd.Series(LCClist).convert_objects(convert_numeric=True)
# seriesD = pd.Series(LCDlist).convert_objects(convert_numeric=True)
# seriesE = pd.Series(LCElist).convert_objects(convert_numeric=True)
# seriesF = pd.Series(LCFlist).convert_objects(convert_numeric=True)
# seriesG = pd.Series(LCGlist).convert_objects(convert_numeric=True)
# seriesH = pd.Series(LCHlist).convert_objects(convert_numeric=True)
# seriesR = pd.Series(LCRlist).convert_objects(convert_numeric=True)

seriesA = pd.Series(float(Float_LCAlist))
seriesB = pd.Series(float(Float_LCBlist))
seriesC = pd.Series(float(Float_LCClist))
seriesD = pd.Series(float(Float_LCDlist))
seriesE = pd.Series(float(Float_LCElist))
seriesF = pd.Series(float(Float_LCFlist))
seriesG = pd.Series(float(Float_LCGlist))
seriesH = pd.Series(float(Float_LCHlist))
seriesR = pd.Series(float(Float_LCRlist))

print (seriesA)

seriesXR_A = pd.Series(XR_A_4us).convert_objects(convert_numeric=True)
seriesXR_B = pd.Series(XR_B_4us).convert_objects(convert_numeric=True)
seriesXR_C = pd.Series(XR_C_4us).convert_objects(convert_numeric=True)
seriesXR_D = pd.Series(XR_D_4us).convert_objects(convert_numeric=True)
seriesXR_E = pd.Series(XR_E_4us).convert_objects(convert_numeric=True)
seriesXR_F = pd.Series(XR_F_4us).convert_objects(convert_numeric=True)
seriesXR_G = pd.Series(XR_G_4us).convert_objects(convert_numeric=True)
seriesXR_H = pd.Series(XR_G_4us).convert_objects(convert_numeric=True)


df = pd.DataFrame({
	'Detector': seriesName,
	'A':seriesA,
	'B':seriesB,
	'C':seriesC,
	'D':seriesD,
	'E':seriesE,
	'F':seriesF,
	'G':seriesG,
	'H':seriesH,
	'R':seriesR,
	# 'XR_A_4us':seriesXR_A,
	# 'XR_B_4us':seriesXR_B,
	# 'XR_C_4us':seriesXR_C,
	# 'XR_D_4us':seriesXR_D,
	# 'XR_E_4us':seriesXR_E,
	# 'XR_F_4us':seriesXR_F,
	# 'XR_G_4us':seriesXR_G,
	# 'XR_H_4us':seriesXR_H,
	})

print ('The length of xray series is: ', len(seriesXR_A), len(seriesXR_B), len(seriesXR_C), len(seriesXR_D), len(seriesXR_E), len(seriesXR_F), len(seriesXR_G), len(seriesXR_H))

df = df.set_index('Detector')
df = df.reindex(index = sorted(df.index))
print (df)
dftype = df.dtypes

#place three quotes here

#makeing sub dataframe without NAN values 
dfNoNan = df[np.isfinite(df['XR_E_4us'])]
dfNoNanMod = dfNoNan.drop(index = ['Sh0112 10 Pass', 'Sh0125 26 Pass', 'Sh0128 24 Pass'])
# dfTest = dfNoNan.drop(['Sh0112', 'Sh0125', 'Sh0128', 'Sh0132'])


#Adding 'totals' column that sums all RT LC for strips [A-H + GR]
df['Total RT LC'] = df[['A','B','C', 'D', 'E', 'F', 'G', 'H', 'R']].sum(axis=1)
#writing dataframe to csv file
df.to_csv('C:/Users/evanm/Desktop/GAPS/Saved df to csv/dfCSV.csv')

#Comparing columns
# compare = df[:100][['A', 'R']]
# print (compare)

#parsing dataframe to single out high values
print(dfNoNanMod)

#Pearson's correlation coefficient
pearson = df['R'].corr(df['A'])
print ('The Pearson correlation coefficient is: ', pearson)
potentialFeatures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for f in potentialFeatures:
	related = df['R'].corr(df[f])
	print('%s: %f' %(f,related))

#plotting Pearson Coefficients 
correlations = [ df['R'].corr(df[f]) for f in potentialFeatures]
len1 = len(potentialFeatures)
len2 = len(correlations)
print (len1, len2)

def plot_dataframe(df, y_label):
	color='coral'
	fig = plt.gcf()
	fig.set_size_inches(20,12)
	plt.ylabel(y_label)

	ax = df2.correlation.plot(linewidth=3.3, color=color)
	ax.set_xticks(df2.index)
	ax.set_xticklabels(df2.attributes, rotation=75);
	plt.show()

df2 = pd.DataFrame({'attributes':potentialFeatures, 'correlation': correlations})
# plot_dataframe('df2', 'R')


##########Linear regression######################
#creating train and test packets
cdf = dfNoNan[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'R', 'XR_A_4us',
			'XR_B_4us', 'XR_C_4us', 'XR_D_4us', 'XR_E_4us', 'XR_F_4us', 
			'XR_G_4us', 'XR_H_4us']]
viz = cdf[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'R', 'XR_A_4us',
			'XR_B_4us', 'XR_C_4us', 'XR_D_4us', 'XR_E_4us', 'XR_F_4us', 
			'XR_G_4us', 'XR_H_4us']]
# viz.hist()
# plt.show()

#Scatter plotting columns against eachother
# plt.scatter(cdf.A, cdf.XR_A_4us, color='blue')
# plt.scatter(cdf.B, cdf.XR_B_4us, color='yellow')
# plt.scatter(cdf.C, cdf.XR_C_4us, color='black')
# plt.scatter(cdf.E, cdf.XR_D_4us, color='red')
# plt.xlabel("RT LC")
# plt.ylabel("X-Ray LC at 4uS")
# plt.show()

msk = np.random.rand(len(dfNoNan)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#train data distribution
plt.scatter(train.A, train.XR_A_4us,  color='blue')
plt.xlabel("RT LC 'A'")
plt.ylabel("XR_A_4us")
# plt.show()

#coefficients and intercept
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['A']])
train_y = np.asanyarray(train[['XR_A_4us']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

#Plot outputs
plt.scatter(train.A, train.XR_A_4us, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("RT LC 'A'")
plt.ylabel('XR_A_4us')
plt.show()

#calculating Mean Absolute Error and other stats
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['A']])
test_y = np.asanyarray(test[['XR_A_4us']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

##########End of regression analysis#########


#brings up stats, ie: mean, std, etc
# stats = df.describe().transpose()
# print (stats)

# #Is any row NULL?
# null = df.isnull().any().any(), df.shape
# print (null)

# #Finding which columns have null values
# findNull = df.isnull().sum(axis=0)
# print (findNull)

# # Take initial # of rows
# rows = df.shape[0]

#Drop the NULL rows
#df = df.dropna()

#Is any row NULL?
null = df.isnull().any().any(), df.shape
#print (null)

# #Finding which columns have null values
findNull = df.isnull().sum(axis=0)
#print (findNull)

# print (df)

# #First10 = df[:10][['A', 'B']]
# # print (First10)
"""
#Test This PUnk
