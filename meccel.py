#############################################################
#MODULOS Y PAQUETES DEL SISTEMA
#############################################################
import numpy as np
from sys import argv,stdout,stderr,exit,path
import pickle,os,time,glob,collections,warnings,pprint
from scipy.integrate import odeint
from scipy.integrate import quad as integral
from IPython import get_ipython as ipython
from IPython.display import HTML, Image
import IPython.core.autocall as autocall
from collections import OrderedDict
from datetime import datetime

#############################################################
#MODULOS GRAFICOS
#############################################################
QIPY=False
def in_ipynb():
    try:
        cfg = get_ipython().config 
        return True
    except NameError:
        return False

if in_ipynb():
    QIPY=True
    from ipywidgets import interact
    from ipywidgets import widgets,interactive,fixed,interact_manual
    
if not QIPY:
    def Image(url="",filename="",f=""):
        pass
    def get_ipython():
        foo=dictObj(dict())
        foo.run_line_magic=lambda x,y:x
        return foo
    from matplotlib import use
    use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as pat
from mpl_toolkits.mplot3d import Axes3D

#############################################################
#CARGA CONFIGURACION
#############################################################
CONF=dict()
exec(open("meccel.conf").read(),{},CONF)
GRA_DIR=CONF["GRA_DIR"]
FIG_DIR=CONF["FIG_DIR"]
IMA_DIR=CONF["IMA_DIR"]
UTI_DIR=CONF["UTI_DIR"]
PYT_DIR=UTI_DIR+"lib/%s/python3.5/site-packages"%CONF["ARCH"]

#############################################################
#MODULOS Y PAQUETES INCORPORADOS
#############################################################
#Documentation: http://spiceypy.readthedocs.io/en/master/
#path.insert(0,PYT_DIR)
import spiceypy as spy

#############################################################
#MACROS AND CONSTANTS
#############################################################
GRADOS=np.pi/180
RADIAN=1/GRADOS 
norm=np.linalg.norm

#############################################################
#CORE ROUTINES
#############################################################
class ExitClass(autocall.IPyAutocall):
    """ Supposingly an autcall class """
    def __call__(self):
        exit()

def error(msg,code=2,stream=stderr):
    print(msg,file=stderr)
    exit(code)

class dictObj(object):
    def __init__(self,dic={}):self.__dict__.update(dic)
    def __add__(self,other):
        self.__dict__.update(other.__dict__)
        return self

def loadConf(filename):
    d=dict()
    conf=dictObj()
    if os.path.lexists(filename):
        exec(open(filename).read(),{},d)
        conf+=dictObj(d)
        qfile=True
    else:
        error("Configuration file '%s' does not found."%filename)
    return conf

def loadArgv(default):
    d=default
    if QIPY or len(argv)==1:return dictObj(d)
    conf=dictObj()
    try:
        config=";".join(argv[1:]).replace("--","")
        exec(config,{},d)
        conf+=dictObj(d)
    except:
        error("Badformed options:",argv[1:])
    return conf

#Pretty printer
PP=pprint.PrettyPrinter(indent=2).pprint

#############################################################
#COMMON ACTIONS
#############################################################
#AVOID WARNINGS
warnings.filterwarnings('ignore')

#############################################################
#RUTINAS GENERALES
#############################################################
def elipsePlot(p=1,e=0.5,vmax=5,figsize=(6,6)):
    fig,axs=plt.subplots(1,1,figsize=figsize)
    fs=np.linspace(0,2*np.pi,100)
    rs=p/(1+e*np.cos(fs))
    xs=rs*np.cos(fs)
    ys=rs*np.sin(fs)

    axs.plot(xs,ys)
    axs.axis('equal')
    axs.grid()

    ax=axs
    ax.set_xlim((-vmax,vmax))
    ax.set_ylim((-vmax,vmax))
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

def conicaDirectriz(F=1.0,e=1.0,fx=1.2,vmax=1,figsize=(4,4)):
    plt.figure(figsize=figsize)
    xs=np.linspace(F/(1+e),2*F,1000)
    ys1=np.sqrt(xs**2*e**2+2*xs*F-xs**2-F**2)
    ys2=-np.sqrt(xs**2*e**2+2*xs*F-xs**2-F**2)
    xmin=F/(1+e)
    x=fx*xmin
    y=np.sqrt(x**2*(e**2-1)+2*x*F-F**2)
    plt.plot(xs,ys1,'r-')
    plt.plot(xs,ys2,'r-')
    plt.plot([x,F],[y,0],'b-')
    plt.plot([0,x],[y,y],'b-')
    plt.plot([F],[0],'ko',ms=5)
    plt.text(F,0,'F',fontsize=14)
    if y<F:plt.text(x/2,y,"%.2f"%x)
    else:plt.text(x/2,F,"%.2f"%x)
    plt.text(x+(F-x)/2.0,y/2,"%.2f"%np.sqrt((F-x)**2+y**2),ha='right')
    plt.xlim((0,2.0*F))
    plt.ylim((-F,F))
    plt.grid()
    plt.xlim((0,vmax))
    plt.ylim((-2*vmax,2*vmax))
    
def elipseCentrada(a=3,e=0.5,vmax=5,figsize=(4,4)):
    b=a*np.sqrt(1-e**2)
    p=a*(1-e**2)
    
    #Elipse
    xs=np.linspace(-a,a,1000)
    ys1=b*np.sqrt(1-xs**2/a**2)
    ys2=-b*np.sqrt(1-xs**2/a**2)
    
    
    fig,axs=plt.subplots(1,1,figsize=figsize)
    axs.plot(xs,ys1,'r-')
    axs.plot(xs,ys2,'r-')

    #Focci
    axs.plot([a*e,-a*e],[0,0],'ko',ms=5)
    axs.text(a*e,0,"F")
    axs.text(-a*e,0,"F'")
    axs.text(0,0,"C")
    
    #Semiminor axis
    axs.plot([0,0],[0,b],'b-',lw=2)
    axs.text(0,b/2,"b=%.1f"%b,ha='left',va='bottom',rotation=90)
    
    #Semilatus rectum
    axs.plot([a*e,a*e],[0,p],'b-',lw=2)
    axs.text(a*e,p/2,"p=%.1f"%p,ha='left',va='bottom',rotation=90)
    
    axs.set_xlim((-vmax,vmax))
    axs.set_ylim((-vmax,vmax))
    axs.grid()
    
def hiperbolaCentrada(a=3,e=1.5,vmax=10,figsize=(4,4)):
    b=a*np.sqrt(e**2-1)
    p=a*(e**2-1)

    #Elipse                                                                                                                                                                          
    xs=np.linspace(a,5*a,1000)
    xsp=np.linspace(-5*a,-a,1000)
    ys1=b*np.sqrt(xs**2/a**2-1)
    ys2=-b*np.sqrt(xs**2/a**2-1)
    ys1p=b*np.sqrt(xsp**2/a**2-1)
    ys2p=-b*np.sqrt(xsp**2/a**2-1)

    #Asintotas
    xsa=np.linspace(-5*a,5*a,1000)
    ysa1=b/a*xsa
    ysa2=-b/a*xsa
    
    fig,axs=plt.subplots(1,1,figsize=figsize)
    axs.plot(xs,ys1,'r-')
    axs.plot(xs,ys2,'r-')
    axs.plot(xsp,ys1p,'r-')
    axs.plot(xsp,ys2p,'r-')
    axs.plot(xsa,ysa1,'g--')
    axs.plot(xsa,ysa2,'g--')

    #Focci                                                                                                                                                                          
    axs.plot([-a*e,+a*e],[0,0],'ko',ms=5)
    axs.text(a*e,0,"F")
    axs.text(-a*e,0,"F'")
    axs.text(0,0,"C")

    #Semimajor axis                                                                                                                                                                 
    axs.plot([0,a],[0,0],'b-',lw=2)
    axs.text(a/2,0,"a")
    
    #Semiminor axis                                                                                                                                                                 
    axs.plot([a,a],[0,b],'b-',lw=2)
    axs.text(a,b/2,"b")

    axs.set_xlim((-vmax,vmax))
    axs.set_ylim((-vmax,vmax))
    axs.grid()

def conicaEspacio(Omega=0,i=0,omega=0,figsize=(4,4)):
    #Propiedades de la cnica
    p=1
    e=0.7

    #Visual inicial
    A=60
    h=30

    #Puntos de la cnica en el sistema natural de referencia
    fs=np.linspace(0,360*GRADOS,100)
    rs=p/(1+e*np.cos(fs))
    xs=rs*np.cos(fs)
    ys=rs*np.sin(fs)
    zs=np.zeros_like(fs)

    #Vectores en el sistema original
    vecrs=np.array([[x,y,z] for x,y,z in zip(xs,ys,zs)])

    #Matriz de transformacin
    Omega*=GRADOS
    i*=GRADOS
    omega*=GRADOS
    
    #Ntese que en el caso de la cnica, el sistema natural 
    #es el que se encuentra rotado respecto al sistema final 
    #por eso la matriz es la inversa
    Rtot=spy.eul2m(-Omega,-i,-omega,3,1,3)

    #Puntos transformados
    vecrppps=np.array([spy.mxv(Rtot,[vecrs[i,0],vecrs[i,1],vecrs[i,2]]) for i in range(len(fs))])

    #Grafico
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111,projection='3d')

    #Defino desde dnde voy a ver la cnica
    ax.view_init(h,A)
    
    #Grfica de los puntos
    ax.plot(vecrppps[:,0],vecrppps[:,1],vecrppps[:,2],'r-',lw=3)

    #Ejes originales
    vmax=3
    ax.plot([0,vmax,0],[0,0,0],[0,0,0],'k-');ax.text(vmax,0,0,'x')
    ax.plot([0,0,0],[0,vmax,0],[0,0,0],'k-');ax.text(0,vmax,0,'y')
    ax.plot([0,0,0],[0,0,0],[0,0,vmax],'k-');ax.text(0,0,vmax,'z')

    #Decoracin
    xl=ax.set_xlim((-vmax,vmax))
    yl=ax.set_ylim((-vmax,vmax))
    zl=ax.set_zlim((-vmax,vmax))

def EoM_Nbody(y,t,masas):
    M=len(y);N=int(M/6)
    
    #Vector de estado
    r=np.zeros((N,3))
    v=np.zeros((N,3))
    
    #Vectores nulo con las derivadas 
    drdt=np.zeros((N,3))
    dvdt=np.zeros((N,3))    
    
    #Asignacin de los vectores de estado
    for i in range(N):
        r[i]=y[3*i:3*i+3];
        v[i]=y[3*N+3*i:3*N+3*i+3]

    # Ecuaciones de movimiento
    for i in range(N):
        drdt[i]=v[i]
        for j in range(N):
            if i==j:continue
            dvdt[i]+=-masas[j]/spy.vnorm(r[i]-r[j])**3*(r[i]-r[j])

    # Devuelve derivadas
    dydt=np.array([])
    for i in range(N):dydt=np.concatenate((dydt,drdt[i]))
    for i in range(N):dydt=np.concatenate((dydt,dvdt[i]))
    return dydt

def EoM_CRTBP(y,t,alpha):
    r1=np.array([-alpha,0,0])
    r2=np.array([1-alpha,0,0])
    omega=np.array([0,0,1])
    
    r=y[:3]
    v=y[3:]
    
    R1=r-r1
    R2=r-r2
    
    drdt=v
    dvdt=-(1-alpha)/spy.vnorm(R1)**3*R1-alpha/spy.vnorm(R2)**3*R2-np.cross(omega,np.cross(omega,r))-2*np.cross(omega,v)
    
    dydt=drdt.tolist()+dvdt.tolist()

    return dydt

def sis2ini(sistema):
    # Prepara el Sistema de Particulas
    masas=[]
    rs=[];vs=[];ys=[]
    for i in sorted(sistema.keys()):
        particula=sistema[i]
        if particula['m']>0:
            masas+=[particula['m']]
            rs+=particula['r'];vs+=particula['v']
    ys=rs+vs
    M=len(ys)
    N=int(M/6)
    return ys,masas,N

def sol2pos(solucion):
    Nt=int(solucion.shape[0])
    N=int(solucion.shape[1]/6)
    rs=np.zeros((N,Nt,3))
    vs=np.zeros((N,Nt,3))
    for i in range(N):
        n=3*i
        rs[i]=solucion[:,n:n+3]
        m=3*N+3*i
        vs[i]=solucion[:,m:m+3]
    return rs,vs

def solucionNbody(sistema,tini,tend,Nt):
    #Preparacin de la solucin
    y,masas,N=sis2ini(sistema)
    #Solucin
    solucion=odeint(EoM_Nbody,y,np.linspace(tini,tend,Nt),args=(masas,))
    #Extraccin de la solucin
    rs,vs=sol2pos(solucion)
    return rs,vs,masas,N
    
def in2cm(rs,vs,masas,Nt,N):
    Masa=np.array(masas).sum()
    P=np.zeros((Nt,3))
    R=np.zeros((Nt,3))
    for n in range(Nt):
        for i in range(N):
            P[n]+=masas[i]*vs[i,n]
            R[n]+=masas[i]*rs[i,n]/Masa
    V=P/Masa 
    rscm=np.zeros((N,Nt,3))
    vscm=np.zeros((N,Nt,3))
    for n in range(Nt):
        for i in range(N):
            rscm[i,n]=rs[i,n]-R[n]
            vscm[i,n]=vs[i,n]-V[n]
    return rscm,vscm

def sol2rot(rs,vs,masas,Nt,N,i1,i2):

    r=norm(rs[i1,0]-rs[i2,0])
    omega=np.sqrt((masas[i1]+masas[i2])/r**3)
    
    #Angulos de rotacin
    r12=rs[i2,:]-rs[i1,:]
    tetas=np.arctan2(r12[:,1],r12[:,0])
    
    # Rota cada una de las posiciones para encontrar la posicin en el sistema rotante
    rsr=np.zeros_like(rs)
    for i in range(len(tetas)):
        Rtot=rotationMatrix(tetas[i],'z')
        for n in range(N):
            rsr[n,i]=Rtot.dot(rs[n,i])

    # Encuentra la velocidad en el sistema rotante
    vsr=np.zeros((Nt,3))
    for i in range(Nt):
        vsr[i]=vs[2,i]-np.cross(np.array([0,0,omega]),rs[2,i])

    return rsr,vsr

def CJacobi(rsr,vsr,masas,Nt,i1,i2,i3):    
    r=norm(rsr[i1,0]-rsr[i2,0])
    omega=np.sqrt((masas[i1]+masas[i2])/r**3)
    rho=np.sqrt(rsr[i3,:,0]**2+rsr[i3,:,1]**2)
    r1=rsr[i3,:]-rsr[i1,:]
    r2=rsr[i3,:]-rsr[i2,:]
    C=[]
    for i in range(Nt):
        C+=[2*masas[i1]/norm(r1[i])+2*masas[i2]/norm(r2[i])+omega**2*rho[i]**2-norm(vsr[i])**2]
    return np.array(C)

def sex2dec(d,m,s):
    dd = d+float(m)/60+float(s)/3600
    return dd

def dec2sex(dd):
    s=np.sign(dd)
    dd=np.abs(dd)
    dg=np.int(dd)
    mm=(dd-dg)*60.0
    mn=np.int(mm)
    ss=(mm-mn)*60.0
    return np.int(s*dg),mn,ss

def rotationMatrix(t,axis):
    R=np.identity(3)
    r=np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    if axis=='z':R[0:2,0:2]=r
    elif axis=='x':R[1:3,1:3]=r
    else:
        R[0,0]=r[0,0];R[0,2]=r[0,1]
        R[2,0]=r[1,0];R[2,2]=r[1,1]
    return R

if __name__=="__main__":
    print("Funciona!")

    print(dec2sex(1.9968365182538348))
