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


ma_count = 2 ##INPUT THE NUMBER OF ADSORBATE MOLECULES ON SURFACE#
slab = io.read('CONTCAR') #YOUR CONTCAR - ENSURE ALL THE ATOMS ARE ORDERED SUCH THAT FIRST n C ATOMS ARE FOR MOL-1, AND REST FOR MOL-2##
print(slab.get_cell())
#del slab[[atom.index for atom in slab if atom.symbol=='Pd']]
#print(slab)

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
del mol_oh[[atom.index for atom in slab if atom.symbol=='Pt']]
del mol_oh[[atom.index for atom in slab if atom.symbol=='C']]
view(mol_oh)

for atom in mol_oh:
		if atom.symbol=='O':
			indo.append(atom.index)
		elif atom.symbol=='H':
			indh.append(atom.index)

for n in range(ma_count-1):
	for m in range(n+1,ma_count):

		##Getting HH distance and index matrix##
		for i in range(int(len(mol_h)/ma_count)):
			for j in range(int(len(mol_h)/ma_count)):
				dist_HH.append(float(mol_h.get_distances(mol_h[i+int((n)*(len(mol_h)/ma_count))].index,mol_h[j+int(m*(len(mol_h)/ma_count))].index,mic=True)))
				HH_index.append([mol_h[i+int((n)*(len(mol_h)/ma_count))].index,mol_h[j+int(m*(len(mol_h)/ma_count))].index])
		
		##Getting OO distance and index matrix##
		for i in range(int(len(mol_o)/ma_count)):
			for j in range(int(len(mol_o)/ma_count)):
				dist_OO.append(float(mol_o.get_distances(mol_o[i+int((n)*(len(mol_o)/ma_count))].index,mol_o[j+int(m*(len(mol_o)/ma_count))].index,mic=True)))
				OO_index.append([mol_o[i+int((n)*(len(mol_o)/ma_count))].index,mol_o[j+int(m*(len(mol_o)/ma_count))].index])
		
		##Getting OH distance and index matrix##
		for i in range(int(len(indo)/ma_count)):
			for j in range(int(len(indh)/ma_count)):
				dist_OH.append(float(mol_oh.get_distances(indo[i+int((n)*len(indo)/ma_count)],indh[j+int(m*(len(indh)/ma_count))],mic=True)))
				OH_index.append([indo[i+int((n)*len(indo)/ma_count)],indh[j+int(m*(len(indh)/ma_count))]])
			

		for i in range(int(len(indo)/ma_count)):
			for j in range(int(len(indh)/ma_count)):
				dist_OH.append(float(mol_oh.get_distances(indo[i+int(m*(len(indo)/ma_count))],indh[j+int((n)*len(indh)/ma_count)],mic=True)))
				OH_index.append([indo[i+int(m*(len(indo)/ma_count))],indh[j+int((n)*len(indh)/ma_count)]])


# print(len(dist_HH))
# print(len(dist_OH))
print(indo)
print(indh)
print(OH_index)
print(dist_OH)
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
	if round(dist_HH[i],1) <= 2.2: ##Cut-off for nearest-neighbor H-H interaction##
		if search(counted_index_HH_NN,HH_index[i][0]) or search(counted_index_HH_NN,HH_index[i][1]):
			if search(counted_index_HH_NN,HH_index[i][0]) and search(counted_index_HH_NN,HH_index[i][1]): 
				a1_HH = a1_HH + 2
			else:
				a1_HH = a1_HH + 1
				a2_HH = a2_HH + 1
			counted_index_HH_NN.append(HH_index[i])
		else:
			a2_HH = a2_HH + 2

	elif 2.2 < round(dist_HH[i],1) <= 3.0: ##Cut-off for nearest-neighbor H-H interaction##
		if search(counted_index_HH_secNN,HH_index[i][0]) or search(counted_index_HH_secNN,HH_index[i][1]):
			if search(counted_index_HH_secNN,HH_index[i][0]) and search(counted_index_HH_secNN,HH_index[i][1]): 
				a11_HH = a11_HH + 2
			else:
				a11_HH = a11_HH + 1
				a22_HH = a22_HH + 1
			counted_index_HH_secNN.append(HH_index[i])
		else:
			a22_HH = a22_HH + 2

print(a1_HH,a2_HH,a11_HH,a22_HH)
print(counted_index_HH_NN)
print(counted_index_HH_secNN)

##Counting O-O interactions##
a1_OO = 0
a2_OO = 0
a11_OO = 0
a22_OO = 0
counted_index_OO_NN = []
counted_index_OO_secNN = []

for i in range(len(dist_OO)):
	if round(dist_OO[i],1) <= 2.8: ##Cut-off for nearest-neighbor O-O interaction##
		if search(counted_index_OO_NN,OO_index[i][0]) or search(counted_index_OO_NN,OO_index[i][1]):
			if search(counted_index_OO_NN,OO_index[i][0]) and search(counted_index_OO_NN,OO_index[i][1]): 
				a1_OO = a1_OO + 2
			else:
				a1_OO = a1_OO + 1
				a2_OO = a2_OO + 1
			counted_index_OO_NN.append(OO_index[i])
		else:
			a2_OO = a2_OO + 2

	elif 2.8 < round(dist_OO[i],1) <= 3.6: ##Cut-off for nearest-neighbor O-O interaction##
		if search(counted_index_OO_secNN,OO_index[i][0]) or search(counted_index_OO_secNN,OO_index[i][1]):
			if search(counted_index_OO_secNN,OO_index[i][0]) and search(counted_index_OO_secNN,OO_index[i][1]): 
				a11_OO = a11_OO + 2
			else:
				a11_OO = a11_OO + 1
				a22_OO = a22_OO + 1
			counted_index_OO_secNN.append(OO_index[i])
		else:
			a22_OO = a22_OO + 2

print(a1_OO,a2_OO,a11_OO,a22_OO)
print(counted_index_OO_NN)
print(counted_index_OO_secNN)


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
	if round(dist_OH[i],1) <= 1.8: ##Cut-off for nearest-neighbor O-O interaction##
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

	elif 1.8 < round(dist_OH[i],1) <= 2.6: ##Cut-off for second-nearest-neighbor O-H interaction##
		print('Oi')
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

print(a1_OH,a2_OH,a11_OH,a22_OH,a1_HO,a2_HO,a11_HO,a22_HO)
print(counted_index_OH_NN)
print(counted_index_OH_secNN)
print('a1_OH,a2_OH,a11_OH,a22_OH,a1_HO,a2_HO,a11_HO,a22_HO,a1_HH,a2_HH,a11_HH,a22_HH,a1_OO,a2_OO,a11_OO,a22_OO:')
print(a1_OH,a2_OH,a11_OH,a22_OH,a1_HO,a2_HO,a11_HO,a22_HO,a1_HH,a2_HH,a11_HH,a22_HH,a1_OO,a2_OO,a11_OO,a22_OO)