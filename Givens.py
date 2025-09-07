import qutip as qp
import numpy as np
import matplotlib.pyplot as plt




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

# Rows according to the topology of the coupling network, making sure that sequential nodes are coupled or are at the top of the matrix.
rows=[7,6,5,4,0,1,2,3]

# Sequence of givens rotations
seq=[[0,2,3],[0,0,6],[0,2,5],[0,0,4],[0,1,0],[0,2,1],[1,0,3],[0,7,2],[1,0,4],[1,0,5],[1,1,2],[1,0,1],[2,2,3],[1,6,0],[2,5,2],[2,0,1],[3,2,3],[2,0,4],[2,5,0]
     ,[3,4,2],[3,0,1],[3,4,0],[4,1,2],[4,0,3],[4,0,1],[5,2,3],[5,1,2],[6,2,3]]
# Pauli matrices for subspaces
def sigmax(i,j):
    sigma=np.zeros(levels)
    sigma=np.diag(sigma)
    sigma[i,i]=0
    sigma[j,j]=0
    sigma[i,j]=1
    sigma[j,i]=1
    return qp.Qobj(sigma)
def sigmay(i,j):
    sigma=np.zeros(levels,dtype=np.complex128)
    sigma=np.diag(sigma)
    sigma[i,i]=0
    sigma[j,j]=0
    sigma[i,j]=-1j
    sigma[j,i]=1j
    return qp.Qobj(sigma)
def sigmaz(i,j):
    sigma=np.zeros(levels,dtype=np.complex128)
    sigma=np.diag(sigma)
    sigma[i,i]=1
    sigma[j,j]=-1
    return qp.Qobj(sigma)
# Native roatation gates
def rotplus(i,j,theta,phi):
    return (-1j*theta*(np.cos(phi)*sigmax(i,j)+np.sin(phi)*sigmay(i,j))/2).expm()
def rotminus(i,j,theta,phi):
    return (-1j*theta*(np.cos(phi)*sigmax(i,j)-np.sin(phi)*sigmay(i,j))/2).expm()



Uoriginal=qp.rand_unitary(levels)
numpyU=Uoriginal.full()
Udet=np.linalg.det(numpyU)
Uoriginal=Uoriginal/(Udet**(1/levels))
U=Uoriginal.full()
Um=np.zeros((levels,levels),dtype=complex)
print(U)
checkzero=np.zeros(levels)
decomp=qp.qeye(levels)
base=[qp.basis(levels,i) for i in range(levels)]
basetrans=base[rows[0]]*base[0].dag()
for i in range(1,levels):
    basetrans=basetrans+(base[rows[i]]*base[i].dag())
# for i in range(levels):
#     Um[i]=U[rows[i]]
# Um=qp.Qobj(Um)

Um=basetrans*qp.Qobj(U)*basetrans.dag()
for i in seq:
    if Um[rows.index(i[2]),i[0]]!=0:
        if i[1]==0 or i[1]==2:
            theta=-2*np.arctan(abs(Um[rows.index(i[2]),i[0]]/Um[rows.index(i[1]),i[0]]))
            phi=(np.angle(Um[rows.index(i[2]),i[0]])-np.angle(Um[rows.index(i[1]),i[0]])+np.pi/2)
            rotation=rotplus(rows.index(i[1]),rows.index(i[2]),theta,phi)
            Um=rotation*Um
            decomp=rotation*decomp
        else:
            theta=-2*np.arctan(abs(Um[rows.index(i[2]),i[0]]/Um[rows.index(i[1]),i[0]]))
            phi=-(np.angle(Um[rows.index(i[2]),i[0]])-np.angle(Um[rows.index(i[1]),i[0]])+np.pi/2)
            rotation=rotminus(rows.index(i[1]),rows.index(i[2]),theta,phi)
            Um=rotation*Um
            decomp=rotation*decomp
print(Um)


edges=[[2,7],[2,1],[2,3],[2,4],[2,5],[0,1],[0,6]]
mat=np.zeros((levels,levels))
for i in range(len(edges)):
    mat[rows.index(edges[i][0]),i]=1/2
    mat[rows.index(edges[i][1]),i]=-1/2
mat[:,levels-1]=1
b=np.angle(np.diag(Um.full()))
D=qp.qeye(levels)
s=qp.qeye(levels)
gamma=np.linalg.solve(mat,b)
for i in range(levels-1):
    # if edges[i][0]==2 or edges[i][0]==0:
    D=D*rotplus(rows.index(edges[i][0]),rows.index(edges[i][1]),np.pi/2,np.pi)*rotplus(rows.index(edges[i][0]),rows.index(edges[i][1]),gamma[i],np.pi/2)*rotplus(rows.index(edges[i][0]),rows.index(edges[i][1]),np.pi/2,0)
    s=s*(1j*sigmaz(rows.index(edges[i][0]),rows.index(edges[i][1]))*gamma[i]/2).expm()
    # else:
    #     D=(rotminus(edges[i][0],edges[i][1],np.pi/2,np.pi)*rotminus(edges[i][0],edges[i][1],gamma[i],np.pi/2)*rotminus(edges[i][0],edges[i][1],np.pi/2,0)).dag()*D
print(D*np.exp(1j*gamma[-1]))
# print(s*np.exp(1j*gamma[-1]))
print(Um)
# print(gamma)
# print(b.sum()/8)
# print(np.angle(np.linalg.det(Um.full()))/8)
