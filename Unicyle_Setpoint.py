""" Credits : MIDHUNTA30 (Midhun T Augustine) for his Non linear setpoint tracking MPC Matlab example.

Point Stabilization of Mobile Robots with Nonlinear Model Predictive Control :
Felipe Kuhne, Walter Fetter Lages and Joao Manoel Gomes da Silva Jr. """

#################################################################################

""" ------------------------ NON LINEAR MPC EXAMPLE ------------------------ """

#################################################################################


import numpy as np 
from numpy import sin as sin , cos as cos , shape as shape , pi as pi , dot as dot , sqrt as sqrt
from numpy.linalg import norm as norm 
from scipy.linalg import block_diag as blkdiag
from scipy.optimize import NonlinearConstraint, fmin_slsqp
from pylab import *
import time
global xek ;  global N ;global n; global m ; global NT ; global T ; global xr ; global ur



start_time = time.time() 
""" Using the error state and control vectors xek = xk - xr, uek = uk - ur"""


def state_function(state,cmd) : 
    x = float(state[0]) ; y = float(state[1]) ; theta= float(state[2]) ; v = float(cmd[0]) ; w = float(cmd[1])
    return np.matrix([ [x + v*T*cos(theta)] , [y + v*T*sin(theta)], [theta + w*T] ])




""" Non linear equality constraints ==> f(xek + xr, uek + ur) - xr = 0 """

def nlcon(z_k) :  # sourcery skip: remove-redundant-slice-index
    global k 
    z_k = np.matrix(z_k).T
    c = np.matrix(np.zeros((n*N,1)))
    for i in range(1,N+1):
        c[(i-1)*n:i*n] = z_k[i*n:(i+1)*n] - ( state_function( z_k[(i-1)*n:i*n] + xr , z_k[(N+1)*n+(i-1)*m:(N+1)*n+i*m] + ur)  - xr ) 
    
    ceq = np.concatenate((z_k[0:n] - x0  , c),axis=0)
    
    return  np.ravel((ceq),order='C')
    

N=8
n=3
m=2
NT = 200
vmin = -1
vmax = 1
wmin = -pi/3
wmax = pi/3
T = 0.1 




xr =  np.matrix([ [1],[1],[0]])
ur =  np.matrix([ [0],[0]])


# Penalties 
Q= 0.5*np.eye(n)
QN=200*Q
R= np.diag([0.1,0.1])


x0 = np.matrix([[0.1],[0.5],[0]])
x = np.matrix(np.zeros((n,NT+1))) # States error (x - xr) vector
x[:,0] = x0 - xr  # Initialization

Xk = np.matrix(np.zeros((n*(N+1),1))) # States error prediction vector 
Xk[:n] = x[:,0]

u=np.matrix(np.zeros((m,NT))) # Error in command value throught the process

Uk = np.matrix(np.zeros((m*N,1))) # Command error prediction vector 
Uk[:m] = - ur

zk = np.concatenate( ( Xk,Uk),axis=0 ) # Error prediction vector [Xek , Uek] 

J = np.matrix(np.zeros(NT)) # COST FUNCTION


"""constructing QX,RU,FX,gX,FU,gU,H"""

QX=Q
RU=R 

for _ in range(N-1) :

  QX= np.matrix(blkdiag(QX,Q)); RU = np.matrix(blkdiag(RU,R))



QX= np.matrix(blkdiag(QX,QN))
H=np.matrix(blkdiag(QX,RU)) 

""" Bounds """

bnds = []
for i in np.arange(0, (N+1)*n + N*m) :
    if i > (N+1)*n and  i % 2 == 0 : 
        bnds.append((wmin, wmax)) 
    elif i >= (N+1)*n and i % 2 == 1 : 
        bnds.append((vmin, vmax))
    else : 
        bnds.append((None,None))


for k in range(NT) :  

    fun = lambda z: z.T @ H @ z # MPC Optimization problem

    x0 = x[:,k] # current state 

    nonlin_con = NonlinearConstraint(nlcon,lb = 0.0 , ub = 0.0 )

    z = fmin_slsqp(fun, zk, f_eqcons=nlcon, bounds= bnds ,disp=False) ; z= np.matrix(z).T # Solution

    J[:,k] = z.T @ H @ z # Cost function 

    u[:,k]= z[(N+1)*n:(N+1)*n+m] # Keep the first sample

    x[:,k+1] = state_function( x[:,k] , u[:,k] )  # Non linear state equation , Xe_k+1 = f(xe,ue)


    zk = z

    #print(f"--- {time.time() - start_time} (Intermediate time ) seconds ---") #une commande toutes les x secondes



print(f"--- {time.time() - start_time} seconds ---")


Time = np.arange(NT+1)
Time_u =np.arange(NT)
xr_1 = np.transpose(np.matrix(xr[0]*np.ones(len(Time))))
xr_2 = np.transpose(np.matrix(xr[1]*np.ones(len(Time))))


figure(1)
plot(Time , np.transpose(x[0]) ,'b',label='X error [m]')
plot(Time , np.transpose(x[1]) ,'r',label='Y error [m]')
xlabel("Time [s]")
legend()
grid()
show()


figure(2)
plot(np.transpose(x[0]) + xr_1 ,np.transpose(x[1]) + xr_2 ,'r',label='X,Y',linestyle='dashed')
xlabel("X [m]")
ylabel("Y [m]")
legend()
grid()
legend()
show() 



U = np.ravel(u[0], order='C')
figure(3)
step(np.arange(NT), U , linewidth=1.5,label='Linear speed [m/s]')
legend()
grid()
xlabel("Time [s]")
show() 


figure(4)
step(np.arange(NT), J.T, linewidth=1.5,label='Cost Function []')
legend()
grid()
xlabel("Time [s]")
show() 

