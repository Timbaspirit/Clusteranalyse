#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:51:38 2019

@author: alagicvn
"""
import numpy as np
import math
import numpy.linalg as LA
import timeit
import itertools


def dissect(data, delta):
      dissection = {}
      n =  data.shape[0]
      d = data.shape[1]
      m = 1/delta
    
      for i in range(n): #loop through all data points
            temp = [0] * d #temporary array to identify boxes via their lowest coordinates
            for j in range(d):
                  #Erkennung des zugehörigen Balls durch Berechnung des "Eckpunkts" mit minimalen Werten
                  temp[j] = int((data[i,j] + 1) * m/2) #key auf jeden Fall int-Tupel
            if tuple(temp) in dissection.keys():
                  #falls key bereits belegt, anhängen des Punktes in Matrix (bzw ndarray)
                  dissection[tuple(temp)] = np.vstack([dissection[tuple(temp)], data[i,:]])
            else:
                  #falls key noch nicht belegt, anlegen eines neuen key-value
                  dissection[tuple(temp)] = data[i,:]
      return dissection
      

def calcM_rho(dissection, rho, q):
      M_rho = {}
      for key in dissection.keys():
            if dissection[key].shape[0] >= rho * q:
                  M_rho[key] = dissection[key]
      return M_rho

def depthFirstSearch(M_rho, B, keylist, key, d, taumult):
      #suboptimal routine disregarding single points within boxes since harder
      
      newcomp = M_rho.pop(key)
#      print("Größe der hinzuzufügenden Komp: " + str(newcomp.shape))
      B = np.vstack((B, newcomp))
#      print("Zsmkomp nach hinzufügen von Kasten" + str(key) + ":" + str(B.shape))
      keylist.append(key)
      
      keys = list(M_rho.keys())
      for candidatekey in keys:
            diff = LA.norm(np.asarray(key)-np.asarray(candidatekey),np.inf)
            if (diff <= taumult/2) and (candidatekey in M_rho.keys()):
#                  print("kommt von " + str(key))
                  B = depthFirstSearch(M_rho, B, keylist, candidatekey, d, taumult)
            
#      for i in range(d):
#            for j in range(1+taumult):
#                  templist = list(key)
#                  templist[i] += j - (taumult/2)
#                  tempkey = tuple(templist)
#                  if tempkey in M_rho.keys():
#                        print("kommt von " + str(key))
#                        B = depthFirstSearch(M_rho, B, keylist, tempkey, d, taumult)
      return B
def cluster(name, epsmult, delta, taumult):
      #name: name of input file name.train.csv and output file name.result.csv
      #eps: width factor of epsilon band
      #delta: fineness of dissection
      #tau: distance factor for connected componenents
      
      data = np.genfromtxt(name + ".csv", delimiter=",")
      
      n =  data.shape[0]
      d = data.shape[1]
      #m = 1/delta
      
      
      dissection = dissect(data, delta)
      testdata = np.ndarray(shape=(0,d))
      for key in dissection.keys():
            testdata = np.vstack((testdata, dissection[key]))
#      print("dissection hat die Form " + str(testdata.shape))
      
      B_delta = 0
      q = n * ((2*delta)**d) #constant for calculation of density function
      
      #Bestimmung der maximalen Dichte
      for key in dissection.keys():
            if dissection[key].shape[0]/q > B_delta:
                   B_delta = dissection[key].shape[0]/q     
                   
      eps = math.sqrt(B_delta / (n*(delta**d))) * epsmult
      rho = 0
#      rho = 604 / (n*(delta**d))
      MM = 1
      while MM == 1:
#            print("j = " + str(int(rho*(n*(delta**d)))))
            B = {}
            keylist = {}
            k = 1
            M_rho = calcM_rho(dissection, rho, q)            
            testdata = np.ndarray(shape=(0,d))
            for key in M_rho.keys():
                  testdata = np.vstack((testdata, M_rho[key]))
#            print("M_rho hat die Form " + str(testdata.shape))
            keys = list(M_rho.keys())
#            print(keys)
            for key in keys:
#                  print("__________ k = " + str(k) + "__________")
                  if key in M_rho.keys():
                        B[k] = np.ndarray(shape=(0,d))
                        keylist[k] = []
                        B[k] = depthFirstSearch(M_rho, B[k], keylist[k], key, d, taumult)
#                        print(B[k])
                        k += 1
            removedB = {}
            for i in range(1,k):
                  remove = 1
                  for key in keylist[i]:
                        if dissection[key].shape[0] >= (rho + 2 * eps) * q:
                              remove = 0
                              break
                  if remove == 1:
#                        if len(B.keys()) == 1:
#                              MM_data = np.ndarray(shape=(0,d + 1))
#                              for key in B.keys():
#                                    classvector = np.ones((B[key].shape[0], 1))
#                                    print("Daten des Schlüssels " + str(key) + ":")
#                                    print(B[key])
#                                    newdata = np.hstack((classvector, B[key]))
#                                    MM_data = np.vstack((MM_data, newdata))
                        removedB[i] = B.pop(i)
            MM = len(B.keys())
            rho += 1 / (n*(delta**d))
#            if MM != 1:
#                  print("Anzahl der Äquivalenzklassen: " + str(MM))
#                  print("Anzahl der Durchläufe: " + str(int(rho*(n*(delta**d)))))
             
      MM_data = np.ndarray(shape=(0,d + 1))
      if MM > 1:
            final_cluster = B
      elif MM == 0:
            final_cluster = removedB
      i = 1
      for key in final_cluster.keys():
            classvector = np.ones((final_cluster[key].shape[0], 1))*i
#                 print("Daten des Schlüssels " + str(key) + ":")
#                 print(B[key])
            newdata = np.hstack((classvector, final_cluster[key]))
            MM_data = np.vstack((MM_data, newdata))
            i += 1
      
#      else:
#            print("Zahl der Zsmhangskomponenten war nicht >1 oder == 0")
            
      output_name = "out/" + name + "." + str(epsmult) + "." + str(delta) + \
      "." + str(taumult) + ".result.csv"
      np.savetxt(output_name, MM_data, delimiter=",", fmt="%1.4f")      
      
#      plot_data = {}
#      colorlist = ['b', 'g', 'r', 'c','m', 'y', 'k', 'w']
#      for key in final_cluster:
      
input_name = "cod-rna.5000"
      
#valuelist = [[1,1.5,2],[.2,.1,.06,.05,.04],[2,4]]
valuelist = [[1],[0.04],[2]]
for [epsmult, delta, taumult] in itertools.product(*valuelist):
      start = timeit.default_timer()

      cluster(input_name, epsmult, delta, taumult)

      stop = timeit.default_timer()
      
#      print(input_name + "." + str(epsmult) + "." + str(delta) + \
#      "." + str(taumult) + ", Time: " + str(stop - start))  
          
      time_file_name = "out/time.txt"
      with open(time_file_name, "a") as time_file:
            line = input_name + ", " + "epsmult = " + str(epsmult) + \
                  ", delta = " + str(delta) + ", taumult = " + str(taumult) + \
                  ". Time: " + str(stop-start) + "\n"
            time_file.write(line)            