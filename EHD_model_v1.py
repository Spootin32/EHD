import numpy as np
import sys
import scipy.optimize as opt
import matplotlib.pyplot as plt

##############################################
#---------------- Parameters ----------------#
##############################################

# Air properties
rho = 1.293 # kg/m3
viscosity = 1.825*(10**(-5)) # Pa.s
v1 = 0.0 # m/s, free stream velocity

# Electrical properties 
eps0 = 8.854187812*(10**(-12)) # F/m, vacuum permittivity
epsr = 1.0006 # F/m, air relative permittivity
eps = eps0*epsr # F/m, air permittivity
mu =  2.0*(10**(-4)) # m2/V/s, ion mobility
C0 = 0.75 # corona discharge constant
Aconst = 112.5 # /kPa/cm constant for Paschen breakdown voltage def
BConst = 2737.5 # V/kPa/cm constant for Paschen breakdown voltage def
P = 100 # kPa, air absolute pressure
gamaSE = 1.0 # range from 0.01 to 100 second ionizition coefficient

# Geometrical parameters
A2 = 0.1*0.1 # m2, EHD section intake area
Ae = 2.5*0.1*0.1 # m2, exist Area
areaRatio = Ae/A2 # Area ratio
d = 10/1000 # m, emitter/collector spacing in given stage
delta = 0.4*d # m, paralell electrode distance in given stage 
c = 0.7*d # m, collector chord
n = 5 # number of stages

# Aerodynamic properties: lookup tables for drag coeff vs Re for a long cylinder i.e. dia << length
ReDataCylinder = [0.067904957, 0.118260056, 0.204982437, 0.35385122, 0.524660966,
                  0.795568278, 1.254626383, 2.122002193, 3.485387348, 5.674696191,
                  8.480995338, 13.23648631, 19.72837463, 39.821248, 56.5659579, 
                  89.11592048, 136.2832104, 205.647544, 649.576037, 991.0640858, 
                  1730.060761, 3060.642188, 4666.575526, 7865.53357, 11347.22245, 
                  18668.47323, 27118.9722, 49492.35646, 163706.1141, 259279.0254, 
                  333255.9525, 382917.9725, 486793.5059, 854658.5283]

CdDataCylinder = [83.71761038, 55.73147402, 37.01332989, 24.83329427, 18.77217645,
                  13.96583015, 10.20865164, 7.180852949, 5.240601925, 3.901293564,
                  3.129077471, 2.511079511, 2.101867511, 1.649446672, 1.509332144,
                  1.382393024, 1.323893851, 1.261626125, 1.063003731, 0.984057587,
                  0.916939708, 0.923121125, 0.964976845, 1.050696, 1.112696119, 
                  1.174139907, 1.212623037, 1.219158419, 1.186829969, 1.00500968, 
                  0.685941519, 0.480223447, 0.345668913, 0.339835142]

ReDataNaca0012 = ReDataCylinder

CdDataNaca0012 = [44.969, 26.894, 21.461, 12.614, 11.47342,
                  8.43338, 5.96478, 4.4266, 3.30184, 2.3295,
                  2.14712, 1.55247, 1.18007, 0.74439, 0.59728,
                  0.45308, 0.35274, 0.27863, 0.14767, 0.11844,
                  0.08991, 0.06831, 0.05594, 0.04396, 0.03733,
                  0.03019, 0.02598, 0.02085, 0.01205, 0.00841,
                  0.00726, 0.0068, 0.00622, 0.00551]

##############################################
#---------------- Functions -----------------#
##############################################

fV0 = lambda dvar: 134.8*(dvar*1000) + 3419 # V, dvar m, corona inception voltage from experimental data
fVbreakdown = lambda Pvar,dvar: BConst*(Pvar*dvar*100)/(np.log(Aconst*(Pvar*dvar*100))-np.log(np.log(1 + 1/gamaSE))) # Paschen's law voltage breakdown
fKL = lambda Revar: np.interp(Revar, ReDataCylinder, CdDataCylinder)
fKL = lambda Revar: np.interp(Revar, ReDataNaca0012, CdDataNaca0012)

def dP_E(v2, *args):
    V_val, P_val, d_val, delta_val, areaRatio_val, n_val = args
    V0 = fV0(d_val) 
    Vb = fVbreakdown(P_val,d_val)
    if (V_val>V0) & (V_val<Vb):
        finter = 1 - 2*np.exp(-4.0*delta_val/d_val)
        vBar = v2*d_val/mu/V_val
        pcBar = (1 + vBar)*(1 - vBar/3) 
        output = C0*eps*V_val*(V_val - V0)/d_val/delta_val*finter*pcBar
    elif V_val<=V0:
        output = 0
    else:
        output = 0
        #print("\n/!\ Arc event /!\ ")

    return output

def dP_loss(v2, *args):
    V_val, P_val, d_val, delta_val, areaRatio_val, n_val = args
    V0 = fV0(d_val)
    Vb = fVbreakdown(P_val,d_val)
    if (V_val>V0) & (V_val<Vb):
        c_val = 0.7*d_val
        Re = rho*v2*c_val/viscosity
        #print("\n----------\nRe = ", Re)
        KL = fKL(Re)
        output = 0.5*rho*(v2**2)*KL
    elif V_val<=V0:
        output = 0
    else:
        output = 0
        #print("\n/!\ Arc event /!\ ")
    return output

def momentum_solver(dPvar, *args):
    V_val, P_val, d_val, delta_val, areaRatio_val, n_val = args
    v2var = areaRatio_val*np.sqrt(v1**2 + 2*dPvar/rho)
    eq1 = dPvar - n_val*(dP_E(v2var, *args) - dP_loss(v2var, *args))
    return eq1


##############################################
#------------------ Solver ------------------#
##############################################
V = 0.90*fVbreakdown(P,d)
data = (V, P, d, delta, areaRatio, n)

dP_init = 100 #n*(dP_E(v1, *data) - dP_loss(v1, *data))
dP_solved = opt.fsolve(momentum_solver, dP_init, args=data)
v2_solved = areaRatio*np.sqrt(v1**2 + 2*dP_solved/rho)
thrustDensity = rho*v2_solved*(v2_solved/areaRatio - v1)
thrustVolDensity = thrustDensity/((d + 0.7*d)*n)

print("\nMax density = {} kg/L\n".format(np.round(thrustVolDensity/9.81/1000,2)))
