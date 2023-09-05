import numpy as np
import sys
import scipy.optimize as opt

##############################################
#---------------- Parameters ----------------#
##############################################

# Air properties
rho = 1.293 # kg/m3
viscosity = 1.825*(10**(-5)) # Pa.s
v1 = 1 # m/s, free stream velocity

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
V = 10000 # V, applied voltage between electrodes

# Geometrical parameters
A2 = 0.1*0.1 # m2, EHD section intake area
Ae = 0.5*0.1*0.1 # m2, exist Area
areaRatio = Ae/A2 # Area ratio
d = 10/1000 # m, emitter/collector spacing in given stage
delta = 10/1000 # m, paralell electrode distance in given stage 
c = 2/1000 # m, collector chord
n = 10 # number of stages

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


##############################################
#---------------- Functions -----------------#
##############################################

fV0 = lambda x: 1348*x + 3419 # V, corona inception voltage from experimental data
fVbreakdown = lambda Pvar,dvar: BConst*(Pvar*dvar*100)/(np.log(Aconst*(Pvar*dvar*100))-np.log(np.log(1 + 1/gamaSE))) # Paschen's law voltage breakdown
fKL = lambda Revar: np.interp(Revar, ReDataCylinder, CdDataCylinder)

def dP_E(v2):
    V0 = fV0(v2)
    Vb = fVbreakdown(P,d)
    print("\n----------\nV = {}\nV0 = {}\nVb = {}".format(V, V0, Vb)) 
    
    if (V>V0) & (V<Vb):
        finter = 1 - 2*np.exp(-4.0*delta/d)
        vBar = v2*d/mu/V
        pcBar = (1 + vBar)*(1 - vBar/3) 
        output = C0*eps*V*(V - V0)/d/delta*finter*pcBar
    elif V<=V0:
        output = 0
    else:
        output = 0
        print("\n/!\ Arc event /!\ ")

    return output

def dP_loss(v2):
    Re = rho*v2*c/viscosity
    print("\n----------\nRe = ", Re)
    KL = fKL(Re)
    return 0.5*rho*(v2**2)*KL

def momentum_solver(dPvar):
    v2var = areaRatio*np.sqrt(v1**2 + 2*dPvar/rho)
    eq1 = dPvar - n*(dP_E(v2var) - dP_loss(v2var))
    return eq1


##############################################
#------------------ Solver ------------------#
##############################################

dP_init = n*(dP_E(v1) - dP_loss(v1))

dP_solved = opt.fsolve(momentum_solver, dP_init)[0]
v2_solved = areaRatio*np.sqrt(v1**2 + 2*dP_solved/rho)
thrustDensity = rho*v2_solved*(v2_solved/areaRatio - v1)

print("\n----------\ndP = {}\nv2 = {}\nF/A2 = {}\n----------\n".format(dP_solved, v2_solved, thrustDensity))
    