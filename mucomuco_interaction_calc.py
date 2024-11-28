import ase
from ase import io
from ase import Atoms
#from ase import get_center_of_mass
import numpy as np
import math
from ase.geometry import get_distances
from ase.visualize import view


#Function to compare two lists and see if they are unique (True) or not (False)#
def search(list, platform):
	for i in range(len(list)):
		for j in range(0,2):
			if list[i][j] == platform:
				return False
	return True

def txt_to_lst(file_path):
    stopword=open(file_path,"r")
    lines = stopword.read().split('\n')
    lines.remove('')

    for i in range(len(lines)):
    	lines[i] = eval(lines[i])
    return np.array(lines)

def convert_strings_to_floats(input_array):
    output_array = []
    for element in input_array:
        converted_float = float(element)
        output_array.append(converted_float)
    return output_array

##NNcutoffs##
hh_nn = 4.60
oo_nn = 2.80
oh_nn = 2.90

##secNN cutoffs##
hh_secnn = hh_nn+0.8
oo_secnn = oo_nn+0.8
oh_secnn = oh_nn+0.8

actual_pt111_ccma = [-0.04,0.11,0.19,0.14,-0.05,0.45,0.71,0.19,0.11,0.20,0.02,0.30,0.06,0.01,0.03,0.04,0.19,0.06,-0.34,-0.27,-0.12,0.03,-0.14,0.02,0.06,0.01,0.08,-0.03,0.03,0.05,0.13,0.08,0.42,0.19,0.43,-0.02,0.10,0.09,0.05,0.15,-0.19,0.04,0.11,-0.15,0.33,0.03,-0.29,-0.02,0.20,-0.03,0.02,-0.15,0.35,0.14,0.32,-0.18,0.24,-0.07,0.36,0.51,-0.11,0.24,0.03,0.03,-0.08,0.02,0.42,0.05,-0.07,0.32,0.04,0.02,0.20,0.03,0.14,0.16,0.12,0.80,0.03,0.40,0.87,0.80,0.32,0.10,0.34,-0.23,0.10,-0.31,-0.20,-0.48,-0.27]
actual_blind_pt111_ccma = [-0.13,-0.18,-0.27,-0.28,-0.52]
actual_pd111_ccma = [-0.14,0.04,0.16,0.22,0.11,0.52,0.50,0.22,-0.03,0.16,0.04,0.04,0.32,0.33,0.40,-0.22,-0.01,0.13,0.01,-0.22,-0.21,0.02,-0.15,-0.01,-0.13,0.09,-0.14,0.02,0.04,0.07,0.02,-0.21,-0.02,0.09,0.06,0.01,0.02,0.36,0.09,0.35,-0.14,0.03,0.11,-0.04,-0.08,0.13,0.06,0.05,-0.34,0.15,-0.13,-0.13,0.03,0.02,-0.34,-0.02,0.02,0.02,0.15,0.12,-0.26,0.11,0.00,0.04,0.12,0.06,0.15,0.02,0.26,-0.15,-0.30,0.11,0.20,0.09,0.05,0.24,0.00,-0.13,-0.02,-0.02,0.20,-0.11,-0.08,0.20,0.33,0.23,0.14,0.16,0.23,0.31]

actual_blind_pd111_ccma = [-0.18,0.09,0.09,0.19,0.24,0.14,0.33,0.16,0.27,0.25]
actual_pd111_t3hda = [-0.34,0.09,-0.22,0.03,0.10,0.23,0.10,-0.30,0.17,0.07,0.16,0.06,0.04,0.20,-0.20,-0.20,0.03,0.16,0.14,-0.06,0.18,0.17,0.02,0.20,-0.26,-0.24,-0.15,0.16,-0.23,0.16,0.07,0.04,0.03,-0.16,0.24,-0.15,0.42,0.17,0.20,-0.27,0.37,-0.31,0.34,-0.32,0.20,0.22,0.54,0.53,0.28,0.22,0.23,0.27,0.32,-0.08,0.28,0.16,-0.12,0.51,0.31,-0.14,-0.13,0.14,0.13,0.43,0.11,0.09,0.23,0.41,0.31,0.52,0.12,0.32,0.07,0.32,0.30,0.22,0.56,0.40,0.37,0.09,0.11,0.29,-0.38,-0.33,-0.11,-0.39,0.16,-0.39,-0.34]
actual_pt111_t3hda = [-0.42,0.14,-0.09,-0.10,-0.02,-0.09,0.05,-0.66,0.53,0.01,0.37,-0.12,-0.28,0.07,-0.07,0.05,-0.39,0.54,0.53,0.06,-0.33,-0.04,0.51,-0.11,0.03,-0.11,0.59,0.00,0.31,-0.33,-0.16,0.51,0.50,-0.01,-0.32,0.56,0.51,0.06,-0.47,-0.18,-0.14,0.15,-0.76,-0.61,-0.47,0.64,0.54,0.23,0.19,0.33,-0.09,0.11,-0.06,0.13,0.55,-0.16,-0.38,0.09,-0.02,-0.36,0.51,0.06,0.13,-0.33,0.20,0.27,0.03,-0.29,-0.07,-0.04,0.10,-0.20,-0.07,-0.08,-0.04,0.17,0.60,0.05,-0.13,-0.10,0.58,0.10,-0.08,-0.37,-0.37,-0.31,-0.63,-0.54,-0.31,-0.50]
actual_pt111_ccmat3hda = [-0.34,0.00,0.09,1.29,0.42,-0.34,1.24,0.08,0.28,-0.06,0.03,0.14,0.14,0.09,0.47,0.12,0.03,0.01,0.05,0.13,-0.04,0.02,0.04,0.52,0.01,0.59,0.06,-0.03,0.56,0.54,-0.15,-0.25,0.07,0.26,0.13,0.50,0.03,0.08,-0.14,0.01,0.13,0.01,-0.34,0.51,0.29,0.50,0.01,-0.03,0.30,0.21,0.68,0.25,0.00,0.65,0.22,0.34,0.36,0.12,0.26,0.49,0.29,0.36,0.56,0.15,0.20,0.18,0.31,0.31,-0.01,0.29,0.66,0.38,0.65,0.68,0.25,0.06,0.30,0.63,0.26,0.02,-0.03,-0.29,0.13,0.14,0.18,0.18,-0.06,0.24,0.24]
actual_pd111_ccmat3hda = [-0.43,0.02,0.07,0.34,0.34,-0.19,0.98,0.18,-0.06,-0.24,0.15,0.11,0.16,0.18,0.23,0.10,0.09,0.16,-0.06,0.10,0.18,0.09,0.15,0.17,0.19,0.10,0.01,-0.29,0.19,0.16,0.21,-0.04,0.20,-0.04,0.01,0.16,0.15,0.12,-0.07,0.36,0.07,-0.30,0.02,-0.37,0.22,-0.07,0.26,0.11,0.25,0.26,0.25,-0.05,0.23,0.26,0.33,-0.01,0.26,0.21,-0.15,-0.08,0.16,0.17,-0.01,0.23,-0.10,0.25,-0.07,0.32,0.14,0.30,0.07,0.29,0.06,0.04,0.09,0.25,0.79,0.15,0.08,0.31,0.05,0.31,0.25,0.27,-0.10]
#print(len(actual_pd111_ccmat3hda))

t = 85

#train = np.concatenate((actual_pd111_ccma[0:t],actual_pt111_ccma[0:t],actual_pd111_t3hda[0:t],actual_pt111_t3hda[0:t],actual_pd111_ccmat3hda[0:t],actual_pt111_ccmat3hda[0:t]))
#pred = np.concatenate((actual_pd111_ccma[t+1:85],actual_pt111_ccma[t+1:85],actual_pd111_t3hda[t+1:85],actual_pt111_t3hda[t+1:85],actual_pd111_ccmat3hda[t+1:85],actual_pt111_ccmat3hda[t+1:85]))
pred = actual_pt111_ccmat3hda[0:t]
train = np.concatenate((actual_pt111_ccma[0:t],actual_pt111_t3hda[0:t]))
#pred = actual_pd111_ccmat3hda[0:t]

print(len(train))
print(pred)

coeff_1 = txt_to_lst('./Pd111_ccMA-ccMA_coeff.txt')
coeff_2 = txt_to_lst('./Pt111_ccMA-ccMA_coeff.txt')
coeff_3 = txt_to_lst('./Pd111_t3HDA-t3HDA_coeff.txt')
coeff_4 = txt_to_lst('./Pt111_t3HDA-t3HDA_coeff.txt')
coeff_5 = txt_to_lst('./Pd111_ccMA-t3HDA_coeff.txt')
coeff_6 = txt_to_lst('./Pt111_ccMA-t3HDA_coeff.txt')

#coeff_train = np.concatenate((coeff_1[0:t],coeff_2[0:t],coeff_3[0:t],coeff_4[0:t],coeff_5[0:t],coeff_6[0:t]))
#coeff_pred = np.concatenate((coeff_1[t+1:85],coeff_2[t+1:85],coeff_3[t+1:85],coeff_4[t+1:85],coeff_5[t+1:85],coeff_6[t+1:85]))

coeff_pred = np.array(coeff_6[0:t])
coeff_train = np.concatenate((coeff_2[0:t],coeff_4[0:t]))

print(coeff_train)
print(coeff_pred)

def adsadscalc(a):
	slab = io.read(a) #YOUR CONTCAR - ENSURE ALL THE ATOMS ARE ORDERED SUCH THAT FIRST n C ATOMS ARE FOR MOL-1, AND REST FOR MOL-2##
	ma_count = int(len(slab[[atom.index for atom in slab if atom.symbol=='C']])/6)
	print(ma_count)
	#del slab[[atom.index for atom in slab if atom.symbol=='Pd']]
	#print(slab)

	##Turn ON for MA-t3HDA systems##
	# if ma_count == 2:
	# 	h = [6,7] ##number of H in MA-1 and MA-2##
	# 	hsum = [0,6,7,0]
	# elif ma_count == 3:
	# 	h = [6,6,7] ##number of H in MA-1, MA-2, and MA-3##
	# 	hsum = [0,6,6,7,0]
	# elif ma_count == 4:
	# 	h = [6,6,6,7] ##number of H in MA-1, MA-2, and MA-3##
	# 	hsum = [0,6,6,6,7,0]

	##Turn ON for MA-MA systems##
	if ma_count == 2:
		h = [6,6] ##number of H in MA-1 and MA-2##
		hsum = [0,6,6,0]
	elif ma_count == 3:
		h = [6,6,6] ##number of H in MA-1, MA-2, and MA-3##
		hsum = [0,6,6,6,0]
	elif ma_count == 4:
		h = [6,6,6,6] ##number of H in MA-1, MA-2, and MA-3##
		hsum = [0,6,6,6,6,0]

	##Turn ON for t3HDA-t3HDA systems##
	# if ma_count == 2:
	# 	h = [8,8] ##number of H in t3HDA-1, and t3HDA-2##
	# 	hsum = [0,8,8,0]
	# elif ma_count == 3:
	# 	h = [8,8,8] ##number of H in t3HDA-1, t3HDA-2, and t3HDA-3##
	# 	hsum = [0,8,8,8,0]
	# elif ma_count == 4:
	# 	h = [8,8,8,8] ##number of H in t3HDA-1, t3HDA-2, and t3HDA-3##
	# 	hsum = [0,8,8,8,8,0]

	##Turn on for t3HDA-MA systems##
	# if ma_count == 2:
	# 	h = [7,6]
	#	hsum = [0,7,6,0]
	# elif ma_count == 3:
	# 	h = [7,6,6]
	#	hsum = [0,7,6,6,0]
	# elif ma_count == 4:
	# 	h = [7,6,6,6]
	#	hsum = [0,7,6,6,6,0]

	dist_HH = []
	dist_OO = []
	dist_OH = []
	HH_index = []
	OO_index = []
	OH_index = []

	indo = []
	indh = []

	mol_h = slab[[atom.index for atom in slab if atom.symbol=='H']]
	mol_o = slab[[atom.index for atom in slab if atom.symbol=='O']]
	mol_oh = slab
	del mol_oh[[atom.index for atom in slab if atom.symbol=='Pd']]
	del mol_oh[[atom.index for atom in slab if atom.symbol=='C']]
	#view(mol_oh)

	for atom in mol_oh:
			if atom.symbol=='O':
				indo.append(atom.index)
			elif atom.symbol=='H':
				indh.append(atom.index)

	for n in range(ma_count-1):
		for m in range(n+1,ma_count):

			##Getting HH distance and index matrix##
			for i in range(int(h[n])):
				for j in range(int(h[m])):
					dist_HH.append(float(mol_h.get_distances(mol_h[i+int(sum(hsum[0:n+1]))].index,mol_h[j+int(sum(hsum[0:m+1]))].index,mic=True)))
					HH_index.append([mol_h[i+int((n)*(len(mol_h)/ma_count))].index,mol_h[j+int(m*(len(mol_h)/ma_count))].index])
			
			##Getting OO distance and index matrix##
			for i in range(int(len(mol_o)/ma_count)):
				for j in range(int(len(mol_o)/ma_count)):
					dist_OO.append(float(mol_o.get_distances(mol_o[i+int((n)*(len(mol_o)/ma_count))].index,mol_o[j+int(m*(len(mol_o)/ma_count))].index,mic=True)))
					OO_index.append([mol_o[i+int((n)*(len(mol_o)/ma_count))].index,mol_o[j+int(m*(len(mol_o)/ma_count))].index])
			
			##Getting OH distance and index matrix##
			for i in range(int(len(indo)/ma_count)):
				for j in range(int(len(indh)/ma_count)):
					dist_OH.append(float(mol_oh.get_distances(indo[i+int((n)*len(indo)/ma_count)],indh[j+int(sum(hsum[0:m+1]))],mic=True)))
					OH_index.append([indo[i+int((n)*len(indo)/ma_count)],indh[j+int(m*(len(indh)/ma_count))]])
				

			for i in range(int(len(indo)/ma_count)):
				for j in range(int(len(indh)/ma_count)):
					dist_OH.append(float(mol_oh.get_distances(indo[i+int(m*(len(indo)/ma_count))],indh[j+int(sum(hsum[0:n+1]))],mic=True)))
					OH_index.append([indo[i+int(m*(len(indo)/ma_count))],indh[j+int((n)*len(indh)/ma_count)]])


	# print(len(dist_HH))
	# print(len(dist_OH))
	#print(indo)
	#print(indh)
	#print(OH_index)
	#print(dist_OH)
	# print(HH_index)
	# print(OH_index)

	##Counting H-H interactions##
	a1_HH = 0
	a2_HH = 0
	a11_HH = 0
	a22_HH = 0
	counted_index_HH_NN = []
	counted_index_HH_secNN = []

	for i in range(len(dist_HH)):
		if round(dist_HH[i],1) <= hh_nn: ##Cut-off for nearest-neighbor H-H interaction##
			if search(counted_index_HH_NN,HH_index[i][0]) or search(counted_index_HH_NN,HH_index[i][1]):
				if search(counted_index_HH_NN,HH_index[i][0]) and search(counted_index_HH_NN,HH_index[i][1]): 
					a1_HH = a1_HH + 2
				else:
					a1_HH = a1_HH + 1
					a2_HH = a2_HH + 1
				counted_index_HH_NN.append(HH_index[i])
			else:
				a2_HH = a2_HH + 2

		elif hh_nn < round(dist_HH[i],1) <= hh_secnn: ##Cut-off for nearest-neighbor H-H interaction##
			if search(counted_index_HH_secNN,HH_index[i][0]) or search(counted_index_HH_secNN,HH_index[i][1]):
				if search(counted_index_HH_secNN,HH_index[i][0]) and search(counted_index_HH_secNN,HH_index[i][1]): 
					a11_HH = a11_HH + 2
				else:
					a11_HH = a11_HH + 1
					a22_HH = a22_HH + 1
				counted_index_HH_secNN.append(HH_index[i])
			else:
				a22_HH = a22_HH + 2

	#print(a1_HH,a2_HH,a11_HH,a22_HH)
	#print(counted_index_HH_NN)
	#print(counted_index_HH_secNN)

	##Counting O-O interactions##
	a1_OO = 0
	a2_OO = 0
	a11_OO = 0
	a22_OO = 0
	counted_index_OO_NN = []
	counted_index_OO_secNN = []

	for i in range(len(dist_OO)):
		if round(dist_OO[i],1) <= oo_nn: ##Cut-off for nearest-neighbor O-O interaction##
			if search(counted_index_OO_NN,OO_index[i][0]) or search(counted_index_OO_NN,OO_index[i][1]):
				if search(counted_index_OO_NN,OO_index[i][0]) and search(counted_index_OO_NN,OO_index[i][1]): 
					a1_OO = a1_OO + 2
				else:
					a1_OO = a1_OO + 1
					a2_OO = a2_OO + 1
				counted_index_OO_NN.append(OO_index[i])
			else:
				a2_OO = a2_OO + 2

		elif oo_nn < round(dist_OO[i],1) <= oo_secnn: ##Cut-off for nearest-neighbor O-O interaction##
			if search(counted_index_OO_secNN,OO_index[i][0]) or search(counted_index_OO_secNN,OO_index[i][1]):
				if search(counted_index_OO_secNN,OO_index[i][0]) and search(counted_index_OO_secNN,OO_index[i][1]): 
					a11_OO = a11_OO + 2
				else:
					a11_OO = a11_OO + 1
					a22_OO = a22_OO + 1
				counted_index_OO_secNN.append(OO_index[i])
			else:
				a22_OO = a22_OO + 2

	#print(a1_OO,a2_OO,a11_OO,a22_OO)
	#print(counted_index_OO_NN)
	#print(counted_index_OO_secNN)


	##Counting O-H and H-O interactions##
	a1_OH = 0
	a2_OH = 0
	a1_HO = 0
	a2_HO = 0
	a11_OH = 0
	a22_OH = 0
	a11_HO = 0
	a22_HO = 0

	counted_index_OH_NN = []
	counted_index_OH_secNN = []

	for i in range(len(dist_OH)):
		if round(dist_OH[i],1) <= oh_nn: ##Cut-off for nearest-neighbor O-O interaction##
			#print(dist_OH[i])
			if search(counted_index_OH_NN,OH_index[i][0]) or search(counted_index_OH_NN,OH_index[i][1]):
				if search(counted_index_OH_NN,OH_index[i][0]) and search(counted_index_OH_NN,OH_index[i][1]): 
					a1_OH = a1_OH + 1
					a1_HO = a1_HO + 1
				elif search(counted_index_OH_NN,OH_index[i][0]) == True: 
					a1_OH = a1_OH + 1
					a2_HO = a2_HO + 1
				elif search(counted_index_OH_NN,OH_index[i][1]) == True: 
					a2_OH = a2_OH + 1
					a1_HO = a1_HO + 1
				counted_index_OH_NN.append(OH_index[i])

		elif oh_nn < round(dist_OH[i],1) <= oh_secnn: ##Cut-off for second-nearest-neighbor O-H interaction##
			#print('Oi')
			if search(counted_index_OH_secNN,OH_index[i][0]) or search(counted_index_OH_secNN,OH_index[i][1]):
				if search(counted_index_OH_secNN,OH_index[i][0]) and search(counted_index_OH_secNN,OH_index[i][1]): 
					a11_OH = a11_OH + 1
					a11_HO = a11_HO + 1
				elif search(counted_index_OH_secNN,OH_index[i][0]) == True: 
					a11_OH = a11_OH + 1
					a22_HO = a22_HO + 1
				elif search(counted_index_OH_secNN,OH_index[i][1]) == True: 
					a22_OH = a22_OH + 1
					a11_HO = a11_HO + 1
				counted_index_OH_secNN.append(OH_index[i])

	#print(a1_OH,a2_OH,a11_OH,a22_OH,a1_HO,a2_HO,a11_HO,a22_HO)
	#print(counted_index_OH_NN)
	#print(counted_index_OH_secNN)
	print('a1_OH,a2_OH,a11_OH,a22_OH,a1_HO,a2_HO,a11_HO,a22_HO,a1_HH,a2_HH,a11_HH,a22_HH,a1_OO,a2_OO,a11_OO,a22_OO:')
	print(a1_OH,a2_OH,a11_OH,a22_OH,a1_HO,a2_HO,a11_HO,a22_HO,a1_HH,a2_HH,a11_HH,a22_HH,a1_OO,a2_OO,a11_OO,a22_OO)
	return [a1_OH,a2_OH,a11_OH,a22_OH,a1_HO,a2_HO,a11_HO,a22_HO,a1_HH,a2_HH,a11_HH,a22_HH,a1_OO,a2_OO,a11_OO,a22_OO]
# file = open('Pt111_ccMA-ccMA_coeff.txt',"a")
# for k in range(85):
# 	coeff.append(adsadscalc(f'./Pt111_ccMA/CONTCAR_{k}'))
# 	file.write(str(coeff[k]))
# 	file.write('\n')
# file.close()
	

#LSR to predict training set MAE, MAX, and R2##
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X = coeff_train
y = train

lr = LinearRegression()
#RegModel = RandomForestRegressor(n_estimators=200,criterion='squared_error')
#RF = RegModel.fit(X,y)
LR=lr.fit(X,y)
y_pred = LR.predict(X)
RSQ = float(metrics.r2_score(y, y_pred))
sumerr = 0
abserr = []
for i in range(len(y)):
	sumerr += abs(y[i]-float(y_pred[i]))
MAE= float(sumerr/len(y))
for i in range(len(y)):
	abserr.append(abs(y[i]-float(y_pred[i])))
MAX = float(max(abserr))

#Blind validation##
# coeff_vali = []

# for l in range(0,11):
# 	coeff_vali.append(adsadscalc(f'./Pt111_ccMA/blind_validation/CONTCAR_{l}'))

X_blind = coeff_pred
y_blind = pred
y_pred_blind = LR.predict(X_blind)
RSQ_blind = float(metrics.r2_score(y_blind, y_pred_blind))
sumerr = 0
abserr = []
for i in range(len(y_blind)):
	sumerr += abs(y_blind[i]-float(y_pred_blind[i]))
MAE_blind= float(sumerr/len(y_blind))
for i in range(len(y_blind)):
	abserr.append(abs(y_blind[i]-float(y_pred_blind[i])))
MAX_blind = float(max(abserr))


# #Plotting the feature importance for Top 10 most important columns
from matplotlib import pyplot as plt

p1 = 0.9
p2 = 0.8
p3 = 0.7
p4 = 0.6
p5 = 0.5
p6 = 0.4

fig,ax = plt.subplots(1)
ax.scatter(y,y_pred)
ax.scatter(y_blind,y_pred_blind)
# print(y_blind,y_pred_blind)
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5]) ##Y = X line##
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('LSR', fontsize=16)
plt.text(-0.5, p1, 'R^2: {:.2f}'.format(RSQ), fontsize=14)
plt.text(-0.5, p2, 'MAE: {:.2f} eV'.format(MAE), fontsize=14)
plt.text(-0.5, p3, 'MAX: {:.2f} eV'.format(MAX), fontsize=14)
#plt.text(-0.5, p4, 'R^2-b: {:.2f}'.format(RSQ_blind), fontsize=14)
plt.text(-0.5, p5, 'MAE-b: {:.2f} eV'.format(MAE_blind), fontsize=14)
plt.text(-0.5, p6, 'MAX-b: {:.2f} eV'.format(MAX_blind), fontsize=14)
plt.draw()
plt.savefig(f'./scatterplot_training_Pt111_universal_LSR')

#RF to predict training set MAE, MAX, and R2##
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

X = coeff_train
y = train

RegModel = RandomForestRegressor(n_estimators=200,criterion='squared_error')
RF = RegModel.fit(X,y)
y_pred = RF.predict(X)
RSQ = float(metrics.r2_score(y, y_pred))
sumerr = 0
abserr = []
for i in range(len(y)):
	sumerr += abs(y[i]-float(y_pred[i]))
MAE= float(sumerr/len(y))
for i in range(len(y)):
	abserr.append(abs(y[i]-float(y_pred[i])))
MAX = float(max(abserr))

#Blind validation##
X_blind = coeff_pred
y_blind = pred
y_pred_blind = RF.predict(X_blind)
RSQ_blind = float(metrics.r2_score(y_blind, y_pred_blind))
sumerr = 0
abserr = []
for i in range(len(y_blind)):
	sumerr += abs(y_blind[i]-float(y_pred_blind[i]))
MAE_blind= float(sumerr/len(y_blind))
for i in range(len(y_blind)):
	abserr.append(abs(y_blind[i]-float(y_pred_blind[i])))
MAX_blind = float(max(abserr))

# #Plotting the feature importance for Top 10 most important columns
from matplotlib import pyplot as plt

p1 = 0.9
p2 = 0.8
p3 = 0.7
p4 = 0.6
p5 = 0.5
p6 = 0.4

fig,ax = plt.subplots(1)
ax.scatter(y,y_pred)
ax.scatter(y_blind,y_pred_blind)
print(y_blind,y_pred_blind)
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('RF', fontsize=16)
plt.text(-0.5, p1, 'R^2: {:.2f}'.format(RSQ), fontsize=14)
plt.text(-0.5, p2, 'MAE: {:.2f} eV'.format(MAE), fontsize=14)
plt.text(-0.5, p3, 'MAX: {:.2f} eV'.format(MAX), fontsize=14)
#plt.text(-0.5, p4, 'R^2-b: {:.2f}'.format(RSQ_blind), fontsize=14)
plt.text(-0.5, p5, 'MAE-b: {:.2f} eV'.format(MAE_blind), fontsize=14)
plt.text(-0.5, p6, 'MAX-b: {:.2f} eV'.format(MAX_blind), fontsize=14)
plt.draw()
plt.savefig(f'./scatterplot_training_Pt111_universal_RF')

# ####### Neural Network algorithm #######

#Separate target variable and predictor variables##

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold

# coeff_train = []

# for l in range(len(train)):
# 	coeff_train.append(adsadscalc(f'./Pd111_ccMA-t3HDA/CONTCAR_{l}'))

X = coeff_train
y = train

Input_Shape = [np.asarray(X).shape[1]]
print(Input_Shape)
NN_model = keras.Sequential([
    layers.BatchNormalization(input_shape = Input_Shape),
    layers.Dense(512, activation = 'relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(256, activation = 'relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(128, activation = 'sigmoid'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(64, activation = 'sigmoid'),
    layers.Dense(1)
    ])

NN_model.compile(optimizer = 'adam', loss = 'mae')
    
NN_model.fit(X,y, batch_size = 100, epochs = 600, verbose = 0)

y_pred = NN_model.predict(X)

RSQ = float(metrics.r2_score(y, y_pred))
sumerr = 0
abserr = []
for i in range(len(y)):
	sumerr += abs(y[i]-float(y_pred[i]))
MAE= float(sumerr/len(y))
for i in range(len(y)):
	abserr.append(abs(y[i]-float(y_pred[i])))
MAX = float(max(abserr))

#Blind validation##
X_blind = coeff_pred
y_blind = pred
y_pred_blind = NN_model.predict(X_blind)
RSQ_blind = float(metrics.r2_score(y_blind, y_pred_blind))
sumerr = 0
abserr = []
for i in range(len(y_blind)):
	sumerr += abs(y_blind[i]-float(y_pred_blind[i]))
MAE_blind= float(sumerr/len(y_blind))
for i in range(len(y_blind)):
	abserr.append(abs(y_blind[i]-float(y_pred_blind[i])))
MAX_blind = float(max(abserr))

# #Plotting the feature importance for Top 10 most important columns
from matplotlib import pyplot as plt

p1 = 0.9
p2 = 0.8
p3 = 0.7
p4 = 0.6
p5 = 0.5
p6 = 0.4

fig,ax = plt.subplots(1)
ax.scatter(y,y_pred)
ax.scatter(y_blind,y_pred_blind)
print(y_blind)
print(y_pred_blind)
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('CNN', fontsize=16)
plt.text(-0.5, p1, 'R^2: {:.2f}'.format(RSQ), fontsize=14)
plt.text(-0.5, p2, 'MAE: {:.2f} eV'.format(MAE), fontsize=14)
plt.text(-0.5, p3, 'MAX: {:.2f} eV'.format(MAX), fontsize=14)
#plt.text(-0.5, p4, 'R^2-b: {:.2f}'.format(RSQ_blind), fontsize=14)
plt.text(-0.5, p5, 'MAE-b: {:.2f} eV'.format(MAE_blind), fontsize=14)
plt.text(-0.5, p6, 'MAX-b: {:.2f} eV'.format(MAX_blind), fontsize=14)
plt.draw()
plt.savefig(f'./scatterplot_training_Pt111_universal_CNN')