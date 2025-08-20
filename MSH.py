import qutip as qp
import numpy as np
import matplotlib.pyplot as plt

#molmer sorensen gate- monochromatic

v=2*np.pi*200
delta=0.9*v
eta=0.1
rabif=0.1*200000
# omegaeg=5*10e9*v
# omega1=omegaeg+delta
# omega2=omegaeg-delta
# Hamiltonian
motionaldim=20
a=qp.destroy(motionaldim)
adag=qp.create(motionaldim)
sigmaplus=qp.create(2)
sigmaminus=qp.destroy(2)
# coeff1 = qp.coefficient(lambda t: np.exp(-1j*omega1*t))
# coeff1dag = qp.coefficient(lambda t: np.exp(1j*omega1*t))
# coeff2 = qp.coefficient(lambda t: np.exp(-1j*omega2*t))
# coeff2dag = qp.coefficient(lambda t: np.exp(1j*omega2*t))
# H=(qp.tensor([omegaeg*qp.sigmaz()/2,qp.qeye(2),qp.qeye(motionaldim)])+qp.tensor([qp.qeye(2),omegaeg*qp.sigmaz()/2,qp.qeye(motionaldim)])
# +qp.tensor([qp.qeye(2),qp.qeye(2),v*(adag*a+1/2)])+rabif/2*(qp.tensor([sigmaplus,qp.qeye(2),(1j*(eta*(a+adag))).expm()])*coeff1+
# qp.tensor([sigmaminus,qp.qeye(2),(-1j*(eta*(a+adag))).expm()])*coeff1dag+qp.tensor([qp.qeye(2),sigmaplus,(1j*(eta*(a+adag))).expm()])*coeff2+
# qp.tensor([qp.qeye(2),sigmaminus,(-1j*(eta*(a+adag))).expm()])*coeff2dag))

# coeffv = qp.coefficient(lambda t: np.exp(-1j*v*t))
# coeffvdag = qp.coefficient(lambda t: np.exp(1j*v*t))
# coeffdelta = qp.coefficient(lambda t: np.exp(-1j*delta*t))
# coeffdeltadag= qp.coefficient(lambda t: np.exp(1j*delta*t))
# power=(1j*(eta*(a*coeffv+adag*coeffv))).expm()

# Full hamiltonian in the interaction picture
def H_t(t):
    return rabif/2*(qp.tensor([sigmaplus,qp.qeye(2),(1j*(eta*(a*np.exp(-1j*((v*t)%(np.pi*2)))+
adag*np.exp(1j*((v*t)%(np.pi*2)))))).expm()])*np.exp(-1j*((delta*t)%(np.pi*2)))+
qp.tensor([sigmaminus,qp.qeye(2),(-1j*(eta*(a*np.exp(-1j*((v*t)%(np.pi*2)))+
adag*np.exp(1j*((v*t)%(np.pi*2)))))).expm()])*np.exp(1j*((delta*t)%(np.pi*2)))+
qp.tensor([qp.qeye(2),sigmaplus,(1j*(eta*(a*np.exp(-1j*((v*t)%(np.pi*2)))+
adag*np.exp(-1j*((v*t)%(np.pi*2)))))).expm()])*np.exp(-1j*((delta*t)%(np.pi*2)))+
qp.tensor([qp.qeye(2),sigmaminus,(-1j*(eta*(a*np.exp(-1j*((v*t)%(np.pi*2)))+
adag*np.exp(1j*((v*t)%(np.pi*2)))))).expm()])*np.exp(1j*((delta*t)%(np.pi*2))))

# Hamiltonian up to second order in Lamb-Dicke parameter
def H_LD(t):
    return rabif/2*(qp.tensor([sigmaplus,qp.qeye(2),qp.qeye(motionaldim)])*np.exp(-1j*((delta*t)%(2*np.pi)))+
qp.tensor([sigmaminus,qp.qeye(2),qp.qeye(motionaldim)])*np.exp(1j*((delta*t)%(2*np.pi)))+
qp.tensor([qp.qeye(2),sigmaplus,qp.qeye(motionaldim)])*np.exp(-1j*((delta*t)%(2*np.pi)))+
qp.tensor([qp.qeye(2),sigmaminus,qp.qeye(motionaldim)])*np.exp(1j*((delta*t)%(np.pi*2)))+
qp.tensor([sigmaplus,qp.qeye(2),(1j*(eta*(a*np.exp(-1j*((v*t)%(np.pi*2)))+adag*np.exp(1j*((v*t)%(np.pi*2))))))])*np.exp(-1j*((delta*t)%(np.pi*2)))+
qp.tensor([sigmaminus,qp.qeye(2),(-1j*(eta*(a*np.exp(-1j*((v*t)%(np.pi*2)))+adag*np.exp(1j*((v*t)%(np.pi*2))))))])*np.exp(1j*((delta*t)%(np.pi*2)))+
qp.tensor([qp.qeye(2),sigmaplus,(1j*(eta*(a*np.exp(-1j*((v*t)%(np.pi*2)))+adag*np.exp(-1j*((v*t)%(np.pi*2))))))])*np.exp(-1j*((delta*t)%(np.pi*2)))+
qp.tensor([qp.qeye(2),sigmaminus,(-1j*(eta*(a*np.exp(-1j*((v*t)%(np.pi*2)))+adag*np.exp(1j*((v*t)%(np.pi*2))))))])*np.exp(1j*((delta*t)%(np.pi*2))))

# H=rabif/2*(qp.tensor([sigmaplus,qp.qeye(2),power])*coeffdelta+
# qp.tensor([sigmaminus,qp.qeye(2),power.dag()])*coeffdeltadag+qp.tensor([qp.qeye(2),sigmaplus,power])*coeffdelta+
# qp.tensor([qp.qeye(2),sigmaminus,power.dag()])*coeffdeltadag)
#initial conditionqp.tensor([sigmaplus,qp.qeye(2),qp.qeye(motionaldim)])*np.exp(-1j*delta*t)
totaltime=0.1
gg=qp.basis([2,2],[0,0])
ee=qp.basis([2,2],[1,1])
coherent=qp.coherent(motionaldim,np.sqrt(2))
number=qp.basis(motionaldim,0)
initmotion=coherent
rho0=qp.ket2dm(qp.tensor([gg,initmotion]))
t=np.linspace(0,totaltime,10000)
# options=qp.Options(nsteps=100000)
# options.nsteps=10000
result=qp.mesolve(H_t,rho0,t,[],[qp.tensor([qp.ket2dm(gg),qp.qeye(motionaldim)]),qp.tensor([qp.ket2dm(ee),qp.qeye(motionaldim)])
        ,qp.tensor([(ee*gg.dag()+gg*ee.dag())/2,qp.qeye(motionaldim)]),qp.tensor([(ee*gg.dag()+gg*ee.dag())/(1j*2),qp.qeye(motionaldim)])])

fig, ax = plt.subplots()
ax.plot(result.times, result.expect[0])
ax.plot(result.times, result.expect[1])
ax.plot(result.times, result.expect[2])
ax.plot(result.times, result.expect[3])
ax.set_xlabel('Time')
ax.set_ylabel('Expectation values')
ax.legend(("gg,gg", "ee,ee","Re(gg,ee)","Im(gg,ee)"))
plt.show()