import numpy as np 
from numpy import sin as sin , arcsin as arcsin , cos as cos , shape as shape , pi as pi , dot as dot , sqrt as sqrt
from numpy.linalg import norm as norm 
from scipy.linalg import block_diag as blkdiag
from scipy.optimize import fmin_slsqp
from pylab import * 
from numpy import arctan as arctan 
import time
global x0 ;  global N ;global n; global m ; global NT ; global T ; global k_iter

# 8 Track shaped trajectory

start_time = time.time() 
""" Using the error state and control vectors xek = xk - xr, uek = uk - ur"""


def state_function(state,cmd) : 
    x = float(state[0]) ; y = float(state[1]) ; theta= float(state[2]) ; v = float(cmd[0]) ; w = float(cmd[1])
    return np.matrix([ [x + v*T*cos(theta)] , [y + v*T*sin(theta)], [theta + w*T] ])


""" Non linear equality constraints ==> dot_xek = f(xek + xr, uek + ur) - dot_xr """

def nlcon(z_k) :
    global k 
    z_k = np.matrix(z_k).T
    c = np.matrix(np.zeros((n*N,1)))
    for i in range(1,N+1):

        # xr and ur must change to fit the horizon window 
        # as the horizon moves over time, the value of the reference must also move 

        xr,ur  = reference(k_iter*N + i-1) # reference value 
        xrp = state_function(xr,ur) # reference value derivative
   
        c[(i-1)*n:i*n] = z_k[i*n:(i+1)*n] - ( state_function( z_k[(i-1)*n:i*n] + xr , z_k[(N+1)*n+(i-1)*m:(N+1)*n+i*m] + ur)  - xrp ) 
      

    ceq = np.concatenate((z_k[:n] - x0  , c),axis=0)
    return  np.ravel((ceq),order='C')
 
nsec = 7 # Number of seconds 
T = 0.1 # Sample time

N=6 ; n=3 ; m=2  ; NT =  int(nsec/T)
vmin = -0.1 ; vmax = 0.1 ; wmin = -pi/10 ; wmax = pi/10
k_iter=0


def reference (k): 
    # returns xr and ur
    dot_x = 1.8*cos(k*T) ; dot_y = 1.2*2*cos(2*k*T) ; ddot_x = - 1.8*sin(k*T) ; ddot_y = - 1.2*4*sin(k*T)
    
    return np.matrix([ [1.8*sin(k*T)],[1.2*sin(2*k*T)],[arctan(dot_y/dot_x)]]) , np.matrix([ [ sqrt(dot_x**2 + dot_y**2) ] , [( ddot_y*dot_x - dot_y*ddot_x) / ( dot_x**2 + dot_y**2 )]  ])

# Reference variables 
xr0 , ur0 = reference(0)

# Penalties 
Q= 0.4*np.eye(n)
QN = np.matrix([ [ 4.3483, 0, 0 ],[ 0, 4.7593 , 4.3374  ],[ 0, 4.3374 , 23.3629] ])
R = 0.5*np.eye(m)


x0 = np.matrix([[0],[0],[pi/3]])

x = np.matrix(np.zeros((n,NT+1))) # States error (x - xr) vector
x[:,0] = x0 -xr0 # Initialization

Xk = np.matrix(np.zeros((n*(N+1),1))) # States error prediction vector 
Xk[:n] = x[:,0] 

u=np.matrix(np.zeros((m,NT))) # Error in command value throught the process
u[:,0] = - ur0

Uk = np.matrix(np.zeros((m*N,1))) # Command error prediction vector 
Uk[:m] = u[:,0] 

zk = np.concatenate( ( Xk,Uk),axis=0 ) # Error prediction vector [Xek , Uek] 

J = np.matrix(np.zeros(NT)) # COST FUNCTION

Xref = np.matrix(np.zeros((n,NT+1))) 
Uref = np.matrix(np.zeros((m,NT)))



"""constructing QX,RU,FX,gX,FU,gU,H"""

QX=Q
RU=R 

for _ in range(1,N):
    QX= np.matrix(blkdiag(QX,Q))
    RU = np.matrix(blkdiag(RU,R))



QX= np.matrix(blkdiag(QX,QN))
H=np.matrix(blkdiag(QX,RU)) 


""" Bounds (on the error) """

bnds = []
for i in np.arange(0, (N+1)*n + N*m) :
    if i > (N+1)*n and  i % 2 == 0 : 
        bnds.append((wmin, wmax)) 
    elif i >= (N+1)*n and i % 2 == 1 : 
        bnds.append((vmin, vmax))
    else : 
        bnds.append((None,None))



for k_iter in range(NT) :  

    #start_time = time.time()

    fun = lambda z: z.T @ H @ z # MPC Optimization problem

    x0 = x[:,k_iter] # current state 

    z = fmin_slsqp(fun, zk, f_eqcons=nlcon, bounds= bnds ,disp=False) ; z= np.matrix(z).T # Solution

    u[:,k_iter]= z[(N+1)*n:(N+1)*n+m] # Keep the first sample

    x[:,k_iter+1] = state_function( x[:,k_iter] , u[:,k_iter] )  # Non linear state equation , Xe_k+1 = f(xe,ue)


    zk = z

    Xref[:,k_iter], Uref[:,k_iter] = reference(k_iter)

    #print(f"--- {time.time() - start_time} (Intermediate time ) seconds ---") 
    


Xref[:,-1] = reference(NT)[0] ; Uref[:,-1] = reference(NT)[1]

print(f"--- {time.time() - start_time} seconds ---")


Time = np.arange(NT+1)
Time_u =np.arange(NT)


figure(1)
plot(Time , np.transpose(x[0]) ,'b',label='X error [m]')
plot(Time , np.transpose(x[1]) ,'r',label='Y error [m]')
xlabel("Time [s]")
legend()
grid()
show()

figure(2)
plot(np.transpose(x[0]) + Xref[0].T , np.transpose(x[1]) + Xref[1].T, 'r',label='2D Position [m]',linestyle='dashed',linewidth=1.5)
plot(Xref[0].T , Xref[1].T ,'dodgerblue',label='Reference',linewidth=0.5)
xlabel("Time [s]")
legend()
grid()
show()

figure(3)
step(Time_u , np.transpose(u[0]) + Uref[0].T,linewidth=1.5,label='Linear speed [m/s]') 
xlabel('Time [s]')
legend()
grid()
show()


figure(3)
step(Time_u , np.transpose(u[1]) + Uref[1].T,linewidth=1.5,label='Angular speed [m/s]') 
xlabel('Time [s]')
legend()
grid()
show()
