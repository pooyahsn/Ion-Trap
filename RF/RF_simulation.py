import qutip as qp
import numpy as np
from scipy.linalg import block_diag
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# plt.rcParams['text.usetex']=True

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",           
    "font.serif": ["Computer Modern"], 
})
plt.rcParams.update({'font.size': 14})

# The constants and splittings are defined
SDomega=411042129776393.2*2*np.pi
SZeemanperB=2.80241e6*2*np.pi
DZeemanperB=1.6800168657e6*np.pi*2
motgroundexpfact=np.sqrt(const.hbar/2/6.63576e-26)
N_ions=1
motmodes=3*N_ions
motdim=1
totmotdim=motdim*motmodes

class RFpulse:
    def __init__(self,N_ions,rabifs,angfrequency,phase):
        self.N_ions=N_ions
        self.angfrequency=angfrequency
        self.Srabif=rabifs
        self.Drabif=rabifs*DZeemanperB/SZeemanperB
        jps=qp.jmat(1/2,'+')
        jpd=qp.jmat(5/2,'+')
        jps=jps.full()*self.Srabif/2
        jpd=jpd.full()*self.Drabif/2
        a=np.zeros((6,6))
        b=np.zeros((2,2))
        if N_ions==1:
            self.jps=qp.tensor(qp.Qobj(block_diag(a,jps).astype(np.complex128)),qp.qeye(totmotdim))
            self.jpd=qp.tensor(qp.Qobj(block_diag(jpd,b).astype(np.complex128)),qp.qeye(totmotdim))
        else:
            identities=[]
            iden=qp.qeye(8)
            identities.append(iden)
            for i in range(1,N_ions-1):
                iden=qp.tensor(iden,qp.qeye(8))
                identities.append(iden)
            self.jps=qp.tensor(qp.Qobj(block_diag(a,jps).astype(np.complex128)),identities[-1],qp.qeye(totmotdim))
            self.jpd=qp.tensor(qp.Qobj(block_diag(jpd,b).astype(np.complex128)),identities[-1],qp.qeye(totmotdim))
            for i in range(1,N_ions):
                if N_ions-i==1:
                    self.jps+=qp.tensor(identities[i-1],qp.Qobj(block_diag(a,jps).astype(np.complex128)),qp.qeye(totmotdim))
                    self.jpd+=qp.tensor(identities[i-1],qp.Qobj(block_diag(jpd,b).astype(np.complex128)),qp.qeye(totmotdim))
                else:
                    self.jps+=qp.tensor(identities[i-1],qp.Qobj(block_diag(a,jps).astype(np.complex128)),identities[N_ions-i-2],qp.qeye(totmotdim))
                    self.jpd+=qp.tensor(identities[i-1],qp.Qobj(block_diag(jpd,b).astype(np.complex128)),identities[N_ions-i-2],qp.qeye(totmotdim))

def coeffs(t,angfrequency,omegas):
    return np.exp(-1j*((t*(angfrequency-omegas))%(2*np.pi)))
def coeffd(t,angfrequency,omegad):
    return np.exp(-1j*((t*(angfrequency-omegad))%(2*np.pi)))
def coeffscon(t,angfrequency,omegas):
    return np.exp(1j*((t*(angfrequency-omegas))%(2*np.pi)))
def coeffdcon(t,angfrequency,omegad):
    return np.exp(1j*((t*(angfrequency-omegad))%(2*np.pi)))

B0=4.148
psi0=(qp.basis(8,0)+qp.basis(8,6)).unit()
# psi0=qp.basis(8,0)
for i in range(1,N_ions):
    psi0=qp.tensor(psi0,(qp.basis(8,0)+qp.basis(8,6)).unit())
    # psi0=qp.basis(8,0)
psi0=qp.tensor(psi0,qp.basis(totmotdim,0))
g=qp.basis(8,7)
e=qp.basis(8,6)
u=qp.basis(8,5)
ud0u=u*u.dag()
ud6=qp.basis(8,0)
ud6u=ud6*ud6.dag()
ud5=qp.basis(8,1)
ud5u=ud5*ud5.dag()
ud4=qp.basis(8,2)
ud4u=ud4*ud4.dag()
ud3=qp.basis(8,3)
ud3u=ud3*ud3.dag()
ud2=qp.basis(8,4)
ud2u=ud2*ud2.dag()
ee=e*e.dag()
gg=g*g.dag()
omegas=SZeemanperB*B0
omegad=DZeemanperB*B0
rfpulse=RFpulse(N_ions,2*np.pi*8.637e3,omegas,0)
print(rfpulse.jpd)
# H=[[-rfpulse.jps,'exp(-1j*t*(angfrequency-omegas))'],[-rfpulse.jps.dag(),'exp(1j*t*(angfrequency-omegas))'],
#    [-rfpulse.jpd,'exp(-1j*t*(angfrequency-omegad))'],[-rfpulse.jpd.dag(),'exp(1j*t*(angfrequency-omegad))']]
# args={'omegas':omegas,'omegad':omegad,'angfrequency':omegad,'pi':np.pi}

H=[[-rfpulse.jps,coeffs],[-rfpulse.jps.dag(),coeffscon],[-rfpulse.jpd,coeffd],[-rfpulse.jpd.dag(),coeffdcon]]
inputangfreq=omegas
args={'omegas':omegas,'omegad':omegad,'angfrequency':inputangfreq}
times=np.linspace(0,600e-6,10000)
# result=qp.mesolve(H,psi0,times,e_ops=[ud6u,ud5u,ud4u,ud3u,ud2u,ud0u,ee,gg],args=args)
result=qp.mesolve(H,psi0,times,args=args)
qp.fileio.qsave(result, 'N_ions='+str(N_ions)+' freq='+str(inputangfreq)+' motdim='+str(motdim)+' B='+str(B0))
Rho=result.states
rho=[qp.ptrace(state,0) for state in Rho]
shownum=100
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.plot(np.array(result.times)*1000, [qp.expect(ud6u,state) for state in rho])
ax.plot(np.array(result.times)*1000, [qp.expect(ud5u,state) for state in rho])
ax.plot(np.array(result.times)*1000, [qp.expect(ud4u,state) for state in rho])
ax.plot(np.array(result.times)*1000, [qp.expect(ud3u,state) for state in rho])
ax.plot(np.array(result.times)*1000, [qp.expect(ud2u,state) for state in rho])
ax.plot(np.array(result.times)*1000, [qp.expect(ud0u,state) for state in rho])

ax.set_xlabel('$t$ (ms)')
ax.set_ylabel("Population")
ax.legend("654321")
plt.show()
fig, ax = plt.subplots()
ax.plot(np.array(result.times)*1000, [qp.expect(ee,state) for state in rho])
ax.plot(np.array(result.times)*1000, [qp.expect(gg,state) for state in rho])

ax.set_xlabel('$t$ (ms)')
ax.set_ylabel('Population')
ax.legend("eg")
plt.show()