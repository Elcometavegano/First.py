from numpy import genfromtxt
import numpy as np
import math
import matplotlib.pyplot as plt
energy = genfromtxt('energy.csv', delimiter=',')
nuclear=0.15 #kw per capita
hydro=20 #kw per km2
inv=1000 #euros per capita
cpm=6500000 #euros per mill
cpsp=110 #euros per solar panel
numberd=4 #number of dams
population=7848000 #people
area=44889 #km2 44889
tinv=inv*population
winds= genfromtxt('Wind speed.csv', delimiter=',')
h2=140
h1=50
z0=1.6
ln1=np.log(h2/z0)
ln2=np.log(h1/z0)
cutin=3 #m/s
cutout=27 #m/s
wtratedpower=5000 #kw
windsco=winds*(ln1/ln2)
rho=1.225 #kg/m3
storagev=13000000 #m3 per dam
volumecap=storagev*numberd
energyp=energy*population
startingpercentage=0.5
qt = 30  # m3/s
qp = 26  # m3/s
height = 762.5  # m
turbinecap=201900
pumpcap=226950
percentage=0.81
wt=int((tinv/cpm)*percentage)
spv=int((tinv/cpsp)*(1-percentage))
solarir = genfromtxt('solarir.csv', delimiter=',')
zenith = genfromtxt('Zenith.csv', delimiter=',')
T=list(range(1,8785))
volumediff = 0
volume = numberd * storagev * startingpercentage
y1=0
totalhydro=[]
totalwind=[]
totalsolar=[]
totalccpp=[]
totalnp=[]
totalrun=[]
sobra=[]
waste=[]
volumen=[]
for t,a,b,c,d in zip(T,energyp,zenith,solarir,windsco):
    # Hydrorun
    hydrorun = hydro * area  # kWh
    # Nuclear power
    nup = nuclear * population # kWh
    # hydro and evaluation
    ef=a-hydrorun-nup
    volume = volume + volumediff
    if y1==0 and volume > 0:
        time1 = (ef/ turbinecap)
        p=0
        if 0 <= time1 <= 1 and volume <= 108000:
            volumeout = time1 * 3600 * qt
            z=ef
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif 0 <= time1 <= 1 and volume > 108000:
            volumeout = time1 * 3600 * qt
            z=ef
            volumein = 0
        elif 1 < time1 <= 2 and volume <= 216000:
            volumeout = (1 + time1 - 1) * 3600 * qt
            z=ef
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif 1 < time1 <= 2 and volume > 216000:
            volumeout = (1 + time1 - 1) * 3600 * qt
            z=ef
            volumein = 0
        elif 2 < time1 <= 3 and volume <= 324000:
            volumeout = (2 + time1 - 2) * 3600 * qt
            z=ef
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif 2 < time1 <= 3 and volume > 324000:
            volumeout = (2 + time1 - 2) * 3600 * qt
            z=ef
            volumein = 0
        elif 3 < time1 <= 4 and volume <= 432000:
            volumeout = (3 + time1 - 3) * 3600 * qt
            z=ef
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif 3 < time1 <= 4 and volume > 432000:
            volumeout = (3 + time1 - 3) * 3600 * qt
            z=ef
            volumein = 0
        elif time1 > 4 and volume <= 432000:
            volumeout = 4 * 3600 * qt
            z = 4 * turbinecap
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif time1 > 4 and volume > 432000:
            volumeout = 4 * 3600 * qt
            z = 4 * turbinecap
            volumein = 0
    elif y1 == 0 and volume ==0:
        volume = 0
        volumein=0
        volumeout = 0
        z=0
        p=0
    elif y1 > 0 and volume==0:
        time1 = (y1 / pumpcap)
        z = 0
        if time1 <= 4:
            volumein = time1 * 3600 * qp
            p = 0
            volumeout=0
        else:
            volumein = 4 * 3600 * qp
            p = y1 - (4 * pumpcap)
            volumeout = 0
    elif y1>0 and volume>0:
        time1 = (y1 / pumpcap)
        z=0
        if time1 <= 4 and volume<volumecap:
            volumein = time1 * 3600 * qp
            p = 0
            volumeout=0
            if volume + volumein >= volumecap:
                volumein=volumecap-volume
                p=y1-((volumein/(3600*qp))*pumpcap)
        elif time1>4 and volume<volumecap:
            volumein = 4 * 3600 * qp
            p = y1 - (4 * pumpcap)
            volumeout = 0
            if volume + volumein >= volumecap:
                volumein=volumecap-volume
                p=y1-((volumein/(3600*qp))*pumpcap)
    if y1>0 and volume == volumecap:
        time1 = (ef/ turbinecap)
        p=0
        if 0 <= time1 <= 1 and volume <= 108000:
            volumeout = time1 * 3600 * qt
            z=ef
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif 0 <= time1 <= 1 and volume > 108000:
            volumeout = time1 * 3600 * qt
            z=ef
            volumein = 0
        elif 1 < time1 <= 2 and volume <= 216000:
            volumeout = (1 + time1 - 1) * 3600 * qt
            z=ef
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif 1 < time1 <= 2 and volume > 216000:
            volumeout = (1 + time1 - 1) * 3600 * qt
            z=ef
            volumein = 0
        elif 2 < time1 <= 3 and volume <= 324000:
            volumeout = (2 + time1 - 2) * 3600 * qt
            z=ef
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif 2 < time1 <= 3 and volume > 324000:
            volumeout = (2 + time1 - 2) * 3600 * qt
            z=ef
            volumein = 0
        elif 3 < time1 <= 4 and volume <= 432000:
            volumeout = (3 + time1 - 3) * 3600 * qt
            z=ef
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif 3 < time1 <= 4 and volume > 432000:
            volumeout = (3 + time1 - 3) * 3600 * qt
            z=ef
            volumein = 0
        elif time1 > 4 and volume <= 432000:
            volumeout = 4 * 3600 * qt
            z = 4 * turbinecap
            volumein = 0
            if volumeout > volume:
                volumeout = volume
                z = (volumeout * turbinecap) / (3600 * qt)
        elif time1 > 4 and volume > 432000:
            volumeout = 4 * 3600 * qt
            z = 4 * turbinecap
            volumein = 0
    total1=ef-z
    if total1>0:
        if d < cutin or d >= cutout:
            cp = 0
        elif d > 13 and d < cutout:
            cp = 0.149388
        else:
            cp = -0.0084 * d ** 2 + 0.1332 * d - 0.0491
        windpowerss = (0.5 * rho * area * d ** 3 * cp) / 1000  # kWh
        if windpowerss > wtratedpower:
            wtr = wtratedpower
        else:
            wtr = windpowerss
        x=wt*wtr
        total2=total1-x
        if total2>0:
            pvsurface = 0.82  # m2
            powerpeak = 110  # W
            pvefficiency = (powerpeak / (1000 * pvsurface))  # powerpeak/(1000*pvsurface)
            inverterefficiency = 0.96
            if b < 5 or b > 85:  # min and max zenith angle
                a1 = 0
            else:
                a1 = ((c * pvsurface * pvefficiency * inverterefficiency) / math.sin(
                    b * (math.pi / 180))) / 1000
            w=spv*a1
            total3=total2-w
            if total3>0:
                v = total3
                y = 0
            else:
                y = -total3
                v = 0
        else:
            y = -total2
            w = 0
            v = 0
    else:
        y=-total1
        x=0
        w=0
        v=0
    volumediff = volumein - volumeout
    y1=y
    totalhydro.append(z)
    totalwind.append(x)
    totalsolar.append(w)
    totalccpp.append(v)
    totalnp.append(nup)
    totalrun.append(hydrorun)
    sobra.append(y1)
    waste.append(p)
    volumen.append(volume)

a11=sum(totalwind)
a12=sum(totalsolar)
a13=sum(totalccpp)
a14=sum(totalnp)
a15=sum(totalrun)
a17=sum(totalhydro)
a16=a11+a12+a13+a14+a15+a17
a21=a11/a16*100
a22=a12/a16*100
a23=a13/a16*100
a24=a14/a16*100
a25=a15/a16*100
a26=a17/a16*100

labels=["Wind","Solar","Combined cycle","Nuclear","Run of river","Hydro"]
sizes=[a21,a22,a23,a24,a25,a26]
fig,ax=plt.subplots()
ax.pie(sizes,labels=labels,autopct="%1.1f%%")
plt.legend()
plt.show()

plt.plot(T,volumen)
plt.ylabel("m3")
plt.xlabel("Hours")
plt.show()

ca=[totalrun[i]+totalnp[i] for i in range(len(totalrun))]
cb=[ca[i]+totalsolar[i] for i in range(len(ca))]
cc=[cb[i]+totalhydro[i] for i in range(len(cb))]
cd=[cc[i]+totalccpp[i] for i in range(len(cc))]
ce=[cd[i]+totalwind[i] for i in range(len(cd))]
cf=[energyp[i]-totalwind[i]-totalccpp[i]-totalhydro[i]-totalsolar[i]-totalnp[i]-totalrun[i]+sobra[i] for i in range(len(totalwind))]
y=np.vstack([totalrun[0:168],ca[0:168],cb[0:168],cc[0:168],cd[0:168],ce[0:168],cf[0:168]])
labels=["Run of the river","Nuclear","Solar","Hydropower","Combined cycle","Wind","Demand",]
fig,ax=plt.subplots()
ax.stackplot(T[0:168],totalrun[0:168],ca[0:168],cb[0:168],cc[0:168],cd[0:168],ce[0:168],cf[0:168],labels=labels)
ax.legend(loc="upper left")
plt.ylabel("kW")
plt.xlabel("Hours")
plt.show()
