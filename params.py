# -*- coding: utf-8 -*
import math
import numpy as npy

G = 6.67408e-08 #Newton’s gravitational constant
mSun= 1.9884754153381438e+33 #mass of the Sun (in grams)
mEarth = 5.965e+27
mElectron = 9.1094e-28 # electron mass
mProton = 1.6726e-24 
mNeutron = 1.6749e-24
mAtom = 1.6605e-24

rSun=696342e3*100 #in cm
rEarth = 6.378e8
rJup = 7.149e9 #jupiter radius 
au = 1.495978707e13 #astronomical unit (in cm)
yr = 2*math.pi /npy.sqrt(G*mSun/au**3) #1 year in seconds
day = 24*3600
GM = G*(mSun+mEarth)

h_Plank = 6.6261e-27 # Plank constant
k_SB = 1.3807e-16 # boltzman constant
sigma_SB = 5.670374e-5 #erg⋅cm−2⋅s−1⋅K−4 stefan-boltzman constant
L_Sun = 3.839e33 # erg s-1  solar luminosity
lyr = 9.461e17 #light year
pc = 3.086e18 #pc

H0 = 72*1e5/1e6/pc

label = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
mass = [0.09*mSun, 1.374*mEarth, 1.308*mEarth, 0.388*mEarth, 0.692*mEarth, 1.039*mEarth, 1.321*mEarth, 0.326*mEarth] # modeling planet masses. Credit: Agol+pre
radii = [0.,1.116,1.097,0.788,0.920,1.045,1.129,0.755] #observational planet radii. Credit: DucrotEtal2020.
