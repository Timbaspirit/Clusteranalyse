#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:48:49 2019

@author: alagicvn
"""

import numpy as np
import timeit


start = timeit.default_timer()

#data = np.genfromtxt("cluster-artificial/bananas-1-2d.csv", delimiter=",")
#np.savetxt("first10.csv", data[0:10], delimiter=",", fmt="%1.4f")

#Daten aus gegebener .csv-Datei laden
d = 4
data = np.genfromtxt("cluster-artificial/bananas-1-"+str(d)+"d.csv", delimiter=",")
#np.savetxt("bananas-1-"+str(d)+"d.first10.csv", data[0:10], delimiter=",", fmt="%1.4f")

#print("shape of data is " + str(data.shape[0]))
#dissection = {}
#print(data[1,1])

#grundlegende Parameter festlegen, später Parameter der Funktion
n = data.shape[0]
#d = data.shape[1]
delta = 0.04
m = 1 / delta
rho = 1/(n*(delta)**d)
tau = 2 * delta

#anlegen der Zerlegung in sup-Norm-Kuglen, d.h. Quadrate, in einer dictionary
dissection = {}
for i in range(n):
      temp = [0] * d
      for j in range(d):
            #Erkennung des zugehörigen Balls durch Berechnung des Eckpunkts mit minimalen Werten
            temp[j] = int((data[i,j] + 1) * m/2)
      if tuple(temp) in dissection.keys():
            #falls key bereits belegt, anhängen des Punktes in Matrix (/Array)
            dissection[tuple(temp)] = np.vstack([dissection[tuple(temp)], data[i,:]])
      else:
            #falls key noch nicht belegt, anlegen eines neuen key-value
            dissection[tuple(temp)] = data[i,:]
      
      
#ballcount = 0
#for key in dissection:
#      print(str(key))
#      ballcount += 1
#print("Ballcount: " + str(ballcount))


q = n * ((2*delta)**d)

#Berechnen und Speichern des Datensatzes M_rho
M_rho = np.ndarray(shape=(d,0))
removeables = []
for key in dissection.keys():
      if dissection[key].shape[0] >= rho * q:
            if np.any(M_rho):
                  M_rho = np.vstack([M_rho, dissection[key]])
            else:
                  M_rho = dissection[key]
      else:
            #Speichern der Schlüssel der zu entfernenden Kästen
            removeables.append(key)
            
#Entfernen der zu dünn besiedelten Kästchen
for key in removeables:
      dissection.pop(key, None)
      
ballcount = 0
for key in dissection:
      print(str(key))
      ballcount += 1
print("Ballcount: " + str(ballcount))
            
np.savetxt("bananas-1-"+str(d)+"d.M_rho.csv", M_rho, delimiter=",", fmt="%1.4f")
print("Form der M_rho-Matrix: " + str(M_rho.shape))

cluster =  {}


stop = timeit.default_timer()

print("Time: " + str(stop - start))