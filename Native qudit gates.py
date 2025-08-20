import qutip as qp
import numpy as np
import matplotlib.pyplot as plt



d=3
levels=8

# Adjacency(coupling) matrix for 8 levels
adjacency=np.zeros((levels,levels))
adjacency[0,1]=1
adjacency[1,0]=adjacency[0,1]
adjacency[0,3]=1
adjacency[3,0]=adjacency[0,3]
adjacency[0,4]=1
adjacency[4,0]=adjacency[0,4]
adjacency[0,5]=1
adjacency[5,0]=adjacency[0,5]
adjacency[0,6]=1
adjacency[6,0]=adjacency[0,6]
adjacency[1,2]=1
adjacency[2,1]=adjacency[1,2]
adjacency[2,3]=1
adjacency[3,2]=adjacency[2,3]
adjacency[2,4]=1
adjacency[4,2]=adjacency[2,4]
adjacency[2,5]=1
adjacency[5,2]=adjacency[2,5]
adjacency[2,7]=1
adjacency[7,2]=adjacency[2,7]


def sigmax(i,j):
    sigma=np.ones(d)
    sigma=np.diag(sigma)
    sigma[i,i]=0
    sigma[j,j]=0
    sigma[i,j]=1
    sigma[j,i]=1
    return qp.Qobj(sigma)
def sigmay(i,j):
    sigma=np.ones(d,dtype=np.complex128)
    sigma=np.diag(sigma)
    sigma[i,i]=0
    sigma[j,j]=0
    sigma[i,j]=-1j
    sigma[j,i]=1j
    return qp.Qobj(sigma)
def rotplus(i,j,theta,phi):
    return (-1j*theta*(np.cos(phi)*sigmax(i,j)+np.sin(phi)*sigmay(i,j))/2).expm()
def rotminus(i,j,theta,phi):
    return (-1j*theta*(np.cos(phi)*sigmax(i,j)-np.sin(phi)*sigmay(i,j))/2).expm()
# print(rot(0,1,np.pi,0))
U=qp.rand_unitary(d)
# U=qp.qeye(d)
print(U)
decomp=qp.qeye(d)
for i in range(d):
    for j in range(d-1,i,-1):
        if U[j,i]!=0:
            for pla in range(d):
                if adjacency[j,pla]==1:
                    c=pla
                    print(j,c)
                    break
            if c==0 or c==2:
                theta=2*np.arctan(abs(U[j,i]/U[c,i]))
                phi=(np.angle(U[j,i])-np.angle(U[c,i])-np.pi/2)
                rotation=rotplus(c,j,theta,phi)
                U=rotation*U
                decomp=rotation*decomp
            else:
                theta=2*np.arctan(abs(U[j,i]/U[c,i]))
                phi=-(np.angle(U[j,i])-np.angle(U[c,i])-np.pi/2)
                rotation=rotminus(c,j,theta,phi)
                U=rotation*U
                decomp=rotation*decomp
print(U)
