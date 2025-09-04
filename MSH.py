import qutip as qp
import numpy as np
import matplotlib.pyplot as plt

#molmer sorensen gate- monochromatic

v=2*np.pi*2*1e5
delta=0.9*v
eta=0.1
rabif=0.1*v
omegaeg=10e15

# Hamiltonian
motionaldim=10
a=qp.destroy(motionaldim)
adag=qp.create(motionaldim)
sigmaplus=qp.create(2)
sigmaminus=qp.destroy(2)


# Full hamiltonian in the interaction picture
def H_t(t):
    return rabif/2*(qp.tensor([sigmaplus,qp.qeye(2),qp.displace(motionaldim,1j*eta*np.exp(1j*((v*t)%(2*np.pi))))])*np.exp(-1j*((delta*t)%(np.pi*2)))+
qp.tensor([sigmaminus,qp.qeye(2),qp.displace(motionaldim,-1j*eta*np.exp(1j*((v*t)%(2*np.pi))))])*np.exp(1j*((delta*t)%(np.pi*2)))+
qp.tensor([qp.qeye(2),sigmaplus,qp.displace(motionaldim,1j*eta*np.exp(1j*((v*t)%(2*np.pi))))])*np.exp(1j*((delta*t)%(np.pi*2)))+
qp.tensor([qp.qeye(2),sigmaminus,qp.displace(motionaldim,-1j*eta*np.exp(1j*((v*t)%(2*np.pi))))])*np.exp(-1j*((delta*t)%(np.pi*2))))

# Hamiltonian up to second order in Lamb-Dicke parameter
def H_LD(t):
    return rabif/2*(qp.tensor([qp.tensor([sigmaplus,qp.qeye(2)])+qp.tensor([qp.qeye(2),sigmaminus]),qp.qeye(motionaldim)])*np.exp(-1j*((delta*t)%(2*np.pi)))+
qp.tensor([qp.tensor([sigmaminus,qp.qeye(2)])+qp.tensor([qp.qeye(2),sigmaplus]),qp.qeye(motionaldim)])*np.exp(1j*((delta*t)%(2*np.pi)))+
qp.tensor([sigmaplus,qp.qeye(2),(1j*(eta*(a*np.exp(-1j*((v*t)%(2*np.pi)))+adag*np.exp(1j*((v*t)%(2*np.pi))))))])*np.exp(-1j*((delta*t)%(2*np.pi)))+
qp.tensor([sigmaminus,qp.qeye(2),(-1j*(eta*(a*np.exp(-1j*((v*t)%(2*np.pi)))+adag*np.exp(1j*((v*t)%(2*np.pi))))))])*np.exp(1j*((delta*t)%(2*np.pi)))+
qp.tensor([qp.qeye(2),sigmaplus,(1j*(eta*(a*np.exp(-1j*((v*t)%(2*np.pi)))+adag*np.exp(-1j*((v*t)%(2*np.pi))))))])*np.exp(1j*((delta*t)%(2*np.pi)))+
qp.tensor([qp.qeye(2),sigmaminus,(-1j*(eta*(a*np.exp(-1j*((v*t)%(2*np.pi)))+adag*np.exp(1j*((v*t)%(2*np.pi))))))])*np.exp(-1j*((delta*t)%(2*np.pi))))

# bichromatic
def H_tbc(t):
    return rabif/2*(qp.tensor([sigmaplus,qp.qeye(2),qp.displace(motionaldim,1j*eta*np.exp(1j*((v*t)%(2*np.pi))))])*np.cos((delta*t)%(np.pi*2))+
qp.tensor([sigmaminus,qp.qeye(2),qp.displace(motionaldim,-1j*eta*np.exp(1j*((v*t)%(2*np.pi))))])*np.cos((delta*t)%(np.pi*2))+
qp.tensor([qp.qeye(2),sigmaplus,qp.displace(motionaldim,1j*eta*np.exp(1j*((v*t)%(2*np.pi))))])*np.cos((delta*t)%(np.pi*2))+
qp.tensor([qp.qeye(2),sigmaminus,qp.displace(motionaldim,-1j*eta*np.exp(1j*((v*t)%(2*np.pi))))])*np.cos((delta*t)%(np.pi*2)))

#initial condition
totaltime=0.005
gg=qp.basis([2,2],[0,0])
ee=qp.basis([2,2],[1,1])
coherent=qp.coherent(motionaldim,np.sqrt(2))
number=qp.basis(motionaldim,0)
initmotion=coherent
rho0=qp.ket2dm(qp.tensor([gg,initmotion]))
t=np.linspace(0,totaltime,100000)
H_tto = qp.QobjEvo(H_t)
H_LDo=qp.QobjEvo(H_LD)
H_tbco=qp.QobjEvo(H_tbc)
# options=qp.Options(nsteps=100000)
# options.nsteps=10000
result=qp.mesolve(H_tto,rho0,t,[],[qp.tensor([qp.ket2dm(gg),qp.qeye(motionaldim)]),qp.tensor([qp.ket2dm(ee),qp.qeye(motionaldim)])
        ,qp.tensor([(ee*gg.dag()+gg*ee.dag())/2,qp.qeye(motionaldim)]),qp.tensor([(ee*gg.dag()-gg*ee.dag())/(1j*2),qp.qeye(motionaldim)])])
# result=qp.sesolve(H_LD,rho0,t,[qp.tensor([qp.ket2dm(gg),qp.qeye(motionaldim)]),qp.tensor([qp.ket2dm(ee),qp.qeye(motionaldim)])
#         ,qp.tensor([(ee*gg.dag()+gg*ee.dag())/2,qp.qeye(motionaldim)]),qp.tensor([(ee*gg.dag()+gg*ee.dag())/(1j*2),qp.qeye(motionaldim)])])
fig, ax = plt.subplots()
ax.plot(result.times, result.expect[0])
ax.plot(result.times, result.expect[1])
ax.plot(result.times, result.expect[2])
ax.plot(result.times, result.expect[3])
ax.set_xlabel('Time')
ax.set_ylabel('Expectation values')
ax.legend(("gg,gg", "ee,ee","Re(gg,ee)","Im(gg,ee)"))
plt.show()