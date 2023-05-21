import math
from tkinter import *
import numpy
from PIL import Image,ImageTk
import numpy as np
from numpy import genfromtxt
from pulp import *
import matplotlib.pyplot as plt
from tkinter import messagebox
root= Tk()
root.geometry("1150x700")
root.title("Battery sizing software")
fn=IntVar()
ln=IntVar()
tn=IntVar()
rn=IntVar()
sn=IntVar()
zn=StringVar()
rad=StringVar()
var1=IntVar()
var2=IntVar()
var3=DoubleVar()
ratio=5000000 #SEK/MW
def val():
    m=fn.get()
    n=rad.get()
    if n=="Frequency regulation":
        r = m / ratio  # MW
        c = r * 2  # MWh
        ffr=449550*c
        fcrd=117700*c
        T = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]
        CAPEX = m
        eff=0.95
        SoC=0.9
        prof = (ffr+fcrd)*eff*SoC
        ccost = 290875 * (c / 2)
        inf = 0.013
        fixedop = 0.01
        marketsh = 0.02
        finalb = -CAPEX
        OPEX = CAPEX * fixedop
        OPEX1 = 0
        prof1 = 0
        ccost1=0
        OPEXi1 = []
        mfi1 = []
        bli1 = []
        pri1 = []
        npvi1 = []
        finalbi1 = []
        ccostsi1=[]
        i1 = 0.01
        i2 = 0.03
        i3 = 0.05
        i4 = 0.07
        for g in T:
            OPEX = OPEX + (OPEX1 * inf)
            prof = prof + (prof1 * inf)
            marketf = (prof * marketsh)
            ccost=ccost+(ccost1*inf)
            balance = prof - OPEX - marketf-ccost
            NPV = balance / (1.00 + i1) ** g
            npvi1.append(NPV)
            finalb = finalb + NPV
            finalbi1.append(finalb)
            prof1 = prof
            OPEX1 = OPEX
            ccost1=ccost
            OPEXi1.append(OPEX)
            mfi1.append(marketf)
            bli1.append(balance)
            pri1.append(prof)
            ccostsi1.append(ccost)
        previous_values1 = [n for n in finalbi1]  # create a copy of the original list to track previous values
        for i, j in zip(range(len(finalbi1)), T):
            if i > 0 and finalbi1[i] > 0 and previous_values1[i - 1] < 0:
                emptylabel3.config(text=str(math.trunc(j)))
            previous_values1[i] = finalbi1[i]  # update the previous_values list with the current value
        positive_sum1 = 0
        for num in finalbi1:
            if num > 0:
                positive_sum1 += num
        emptylabel2.config(text=str(math.trunc(positive_sum1)))
        OPEXi2 = []
        mfi2 = []
        bli2 = []
        pri2 = []
        npvi2 = []
        finalbi2 = []
        ccostsi2 = []
        CAPEX = m
        prof = (ffr + fcrd) * eff * SoC
        ccost = 290875 * (c / 2)
        inf = 0.013
        fixedop = 0.01
        marketsh = 0.02
        finalb = -CAPEX
        OPEX = CAPEX * fixedop
        OPEX1 = 0
        prof1 = 0
        ccost1 = 0
        for g in T:
            OPEX = OPEX + (OPEX1 * inf)
            prof = prof + (prof1 * inf)
            marketf = (prof * marketsh)
            ccost = ccost + (ccost1 * inf)
            balance = prof - OPEX - marketf - ccost
            NPV = balance / (1.00 + i2) ** g
            npvi2.append(NPV)
            finalb = finalb + NPV
            finalbi2.append(finalb)
            prof1 = prof
            OPEX1 = OPEX
            ccost1=ccost
            OPEXi2.append(OPEX)
            mfi2.append(marketf)
            bli2.append(balance)
            pri2.append(prof)
            ccostsi2.append(ccost)
        previous_values2 = [n for n in finalbi2]  # create a copy of the original list to track previous values
        for i, j in zip(range(len(finalbi2)), T):
            if i > 0 and finalbi2[i] > 0 and previous_values2[i - 1] < 0:
                emptylabel5.config(text=str(math.trunc(j)))
            previous_values2[i] = finalbi2[i]  # update the previous_values list with the current value
        positive_sum2 = 0
        for num in finalbi2:
            if num > 0:
                positive_sum2 += num
        emptylabel4.config(text=str(math.trunc(positive_sum2)))
        OPEXi3 = []
        mfi3 = []
        bli3 = []
        pri3 = []
        npvi3 = []
        finalbi3 = []
        ccostsi3 = []
        CAPEX = m
        prof = (ffr + fcrd) * eff * SoC
        ccost = 290875 * (c / 2)
        inf = 0.013
        fixedop = 0.01
        marketsh = 0.02
        finalb = -CAPEX
        OPEX = CAPEX * fixedop
        OPEX1 = 0
        prof1 = 0
        ccost1 = 0
        for g in T:
            OPEX = OPEX + (OPEX1 * inf)
            prof = prof + (prof1 * inf)
            marketf = (prof * marketsh)
            ccost = ccost + (ccost1 * inf)
            balance = prof - OPEX - marketf - ccost
            NPV = balance / (1.00 + i3) ** g
            npvi3.append(NPV)
            finalb = finalb + NPV
            finalbi3.append(finalb)
            prof1 = prof
            OPEX1 = OPEX
            ccost1 = ccost
            OPEXi3.append(OPEX)
            mfi3.append(marketf)
            bli3.append(balance)
            pri3.append(prof)
            ccostsi3.append(ccost)
        previous_values3 = [n for n in finalbi3]  # create a copy of the original list to track previous values
        for i, j in zip(range(len(finalbi3)), T):
            if i > 0 and finalbi3[i] > 0 and previous_values3[i - 1] < 0:
                emptylabel7.config(text=str(math.trunc(j)))
            previous_values3[i] = finalbi3[i]  # update the previous_values list with the current value
        positive_sum3 = 0
        for num in finalbi3:
            if num > 0:
                positive_sum3 += num
        emptylabel6.config(text=str(math.trunc(positive_sum3)))
        OPEXi4 = []
        mfi4 = []
        bli4 = []
        pri4 = []
        npvi4 = []
        finalbi4 = []
        ccostsi4 = []
        CAPEX = m
        prof = (ffr + fcrd) * eff * SoC
        ccost = 290875 * (c / 2)
        inf = 0.013
        fixedop = 0.01
        marketsh = 0.02
        finalb = -CAPEX
        OPEX = CAPEX * fixedop
        OPEX1 = 0
        prof1 = 0
        ccost1 = 0
        for g in T:
            OPEX = OPEX + (OPEX1 * inf)
            prof = prof + (prof1 * inf)
            marketf = (prof * marketsh)
            ccost = ccost + (ccost1 * inf)
            balance = prof - OPEX - marketf - ccost
            NPV = balance / (1.00 + i4) ** g
            npvi4.append(NPV)
            finalb = finalb + NPV
            finalbi4.append(finalb)
            prof1 = prof
            OPEX1 = OPEX
            ccost1=ccost
            OPEXi4.append(OPEX)
            mfi4.append(marketf)
            bli4.append(balance)
            pri4.append(prof)
            ccostsi4.append(ccost)
        previous_values4 = [n for n in finalbi4]  # create a copy of the original list to track previous values
        for i, j in zip(range(len(finalbi4)), T):
            if i > 0 and finalbi4[i] > 0 and previous_values4[i - 1] < 0:
                emptylabel9.config(text=str(math.trunc(j)))
            previous_values4[i] = finalbi4[i]  # update the previous_values list with the current value
        positive_sum4 = 0
        for num in finalbi4:
            if num > 0:
                positive_sum4 += num
        emptylabel8.config(text=str(math.trunc(positive_sum4)))
        emptylabel1.config(text="The power and capacity purchased are " + str(
            r) + " MW and " + str(c) + " MWh, " + " respectively")
        # Create two series
        x = numpy.array(T)
        y1 = numpy.array(finalbi1)
        y2 = numpy.array(finalbi2)
        y3 = numpy.array(finalbi3)
        y4 = numpy.array(finalbi4)

        # Set the width of the bars
        width = 0.2

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the four series as grouped bars
        ax.bar(x - (1.5 * width), y1, width, label='i=1%', color="g")
        ax.bar(x - (0.5 * width), y2, width, label='i=3%', color="b")
        ax.bar(x + (0.5 * width), y3, width, label='i=5%', color="y")
        ax.bar(x + (1.5 * width), y4, width, label='i=7%', color="r")

        # Add x-axis tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(x)

        # Add a legend
        ax.legend()

        # Add axis labels
        ax.set_xlabel('Years')
        ax.set_ylabel('SEK')
        ax.set_title("Project's cashflow")
        # Show the plot
        plt.show()
        battsh = m*0.5978
        peesh = m*0.2105
        staffsh = m*0.1014
        materialssh = m*0.0199
        constrush = m*0.0238
        subsh = m*0.0209
        missh = m*0.002
        persh = m*0.0238
        shares = [battsh, peesh, staffsh, materialssh, constrush, subsh, missh, persh]

        # Create a list of 8 colors for the shares
        labels = ['Battery', 'PCS', 'Staff', 'Materials', 'Construction', 'Subcontractors', 'Miscellaneous',
                  'Personnel']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

        # Create a pie chart with the shares and colors
        plt.pie(shares, colors=colors, labels=labels, autopct=lambda pct: f'{pct:.2f}% ({int((pct * sum(shares))/100):,})')

        # Add a title to the pie chart
        plt.title('CAPEX distribution (SEK)')
        plt.legend(loc='best')

        # Show the plot
        plt.show()
    if n=="Arbitrage":
        # Define the time horizon (in hours)
        r = m / ratio  # MW
        c = r * 2  # MWh
        socmax=0.95
        socmin=0.05
        T1 = 8760
        # Define the electricity prices
        electricity_prices = np.genfromtxt('price.csv', delimiter=',', filling_values=200)

        # Define the charging efficiency
        eta_charging = 0.95

        # Define the constraints
        max_energy_storage = c*socmax
        min_energy_storage = c*socmin
        max_charging_power = r
        max_discharging_power = r

        # Define the initial energy storage level
        initial_energy = min_energy_storage

        # Create the problem object
        prob = LpProblem("Maximize Electricity Profits", LpMaximize)

        # Define the variables
        charging_power = LpVariable.dicts("Charging Power", range(T1), lowBound=0, upBound=max_charging_power)
        discharging_power = LpVariable.dicts("Discharging Power", range(T1), lowBound=0, upBound=max_discharging_power)
        energy_storage = LpVariable.dicts("Energy Storage", range(T1 + 1), lowBound=min_energy_storage,
                                          upBound=max_energy_storage)
        b1 = LpVariable.dicts("Binary", range(T1), cat=LpBinary)
        b2 = LpVariable.dicts("Binary2", range(T1), cat=LpBinary)
        # Define the objective function
        profit = lpSum([electricity_prices[t] * (charging_power[t] - discharging_power[t])
                        for t in range(T1)])
        prob += profit, "Electricity Profits"

        # Define the constraints
        for t in range(T1):
            # Charging constraint
            prob += charging_power[t] <= (max_energy_storage - energy_storage[t]) / eta_charging

            # Discharging constraint
            prob += discharging_power[t] <= (energy_storage[t] - min_energy_storage) * eta_charging

            # Charging constraint
            prob += charging_power[t] <= max_charging_power * b1[t]
            # Discharging constraint
            prob += discharging_power[t] <= max_discharging_power * b2[t]

            prob += b1[t] + b2[t] <= 1

            # Limits
            prob += energy_storage[t] <= max_energy_storage
            prob += energy_storage[t] >= min_energy_storage
            prob += energy_storage[t + 1] == energy_storage[t] + (
                        charging_power[t] * eta_charging - discharging_power[t] * (1 / eta_charging))
        # Solve the problem
        prob.solve()

        # Print the status of the solution
        print("Status:", LpStatus[prob.status])

        charging = []
        discharging = []
        energys = []
        # Print the optimal values of the variables
        for t in range(T1):
            charging.append(value(charging_power[t]))
            discharging.append(value(discharging_power[t]))
            energys.append(value(energy_storage[t]))
        T = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]
        CAPEX = m
        prof = value(profit)
        inf = 0.013
        fixedop = 0.01
        marketsh = 0.02
        finalb = -CAPEX
        OPEX = CAPEX * fixedop
        OPEX1 = 0
        prof1 = 0
        OPEXi1 = []
        mfi1 = []
        bli1 = []
        pri1 = []
        npvi1 = []
        finalbi1 = []
        i1 = 0.01
        i2 = 0.03
        i3 = 0.05
        i4 = 0.07
        for g in T:
            OPEX = OPEX + (OPEX1 * inf)
            prof = prof + (prof1 * inf)
            marketf = (prof * marketsh)
            balance = prof - OPEX - marketf
            NPV = balance / (1.00 + i1) ** g
            npvi1.append(NPV)
            finalb = finalb + NPV
            finalbi1.append(finalb)
            prof1 = prof
            OPEX1 = OPEX
            OPEXi1.append(OPEX)
            mfi1.append(marketf)
            bli1.append(balance)
            pri1.append(prof)
        previous_values1 = [n for n in finalbi1]  # create a copy of the original list to track previous values
        for i, j in zip(range(len(finalbi1)), T):
            if i > 0 and finalbi1[i] > 0 and previous_values1[i - 1] < 0:
                emptylabel3.config(text=str(math.trunc(j)))
            previous_values1[i] = finalbi1[i]  # update the previous_values list with the current value
        positive_sum1 = 0
        for num in finalbi1:
            if num > 0:
                positive_sum1 += num
        emptylabel2.config(text=str(math.trunc(positive_sum1)))
        OPEXi2 = []
        mfi2 = []
        bli2 = []
        pri2 = []
        npvi2 = []
        finalbi2 = []
        CAPEX = m
        prof = value(profit)
        inf = 0.013
        fixedop = 0.01
        marketsh = 0.02
        finalb = -CAPEX
        OPEX = CAPEX * fixedop
        OPEX1 = 0
        prof1 = 0
        for g in T:
            OPEX = OPEX + (OPEX1 * inf)
            prof = prof + (prof1 * inf)
            marketf = (prof * marketsh)
            balance = prof - OPEX - marketf
            NPV = balance / (1.00 + i2) ** g
            npvi2.append(NPV)
            finalb = finalb + NPV
            finalbi2.append(finalb)
            prof1 = prof
            OPEX1 = OPEX
            OPEXi2.append(OPEX)
            mfi2.append(marketf)
            bli2.append(balance)
            pri2.append(prof)
        previous_values2 = [n for n in finalbi2]  # create a copy of the original list to track previous values
        for i, j in zip(range(len(finalbi2)), T):
            if i > 0 and finalbi2[i] > 0 and previous_values2[i - 1] < 0:
                emptylabel5.config(text=str(math.trunc(j)))
            previous_values2[i] = finalbi2[i]  # update the previous_values list with the current value
        positive_sum2 = 0
        for num in finalbi2:
            if num > 0:
                positive_sum2 += num
        emptylabel4.config(text=str(math.trunc(positive_sum2)))
        OPEXi3 = []
        mfi3 = []
        bli3 = []
        pri3 = []
        npvi3 = []
        finalbi3 = []
        CAPEX = m
        prof = value(profit)
        inf = 0.013
        fixedop = 0.01
        marketsh = 0.02
        finalb = -CAPEX
        OPEX = CAPEX * fixedop
        OPEX1 = 0
        prof1 = 0
        for g in T:
            OPEX = OPEX + (OPEX1 * inf)
            prof = prof + (prof1 * inf)
            marketf = (prof * marketsh)
            balance = prof - OPEX - marketf
            NPV = balance / (1.00 + i3) ** g
            npvi3.append(NPV)
            finalb = finalb + NPV
            finalbi3.append(finalb)
            prof1 = prof
            OPEX1 = OPEX
            OPEXi3.append(OPEX)
            mfi3.append(marketf)
            bli3.append(balance)
            pri3.append(prof)
        previous_values3 = [n for n in finalbi3]  # create a copy of the original list to track previous values
        for i, j in zip(range(len(finalbi3)), T):
            if i > 0 and finalbi3[i] > 0 and previous_values3[i - 1] < 0:
                emptylabel7.config(text=str(math.trunc(j)))
            previous_values3[i] = finalbi3[i]  # update the previous_values list with the current value
        positive_sum3 = 0
        for num in finalbi3:
            if num > 0:
                positive_sum3 += num
        emptylabel6.config(text=str(math.trunc(positive_sum3)))
        OPEXi4 = []
        mfi4 = []
        bli4 = []
        pri4 = []
        npvi4 = []
        finalbi4 = []
        CAPEX = m
        prof = value(profit)
        inf = 0.013
        fixedop = 0.01
        marketsh = 0.02
        finalb = -CAPEX
        OPEX = CAPEX * fixedop
        OPEX1 = 0
        prof1 = 0
        for g in T:
            OPEX = OPEX + (OPEX1 * inf)
            prof = prof + (prof1 * inf)
            marketf = (prof * marketsh)
            balance = prof - OPEX - marketf
            NPV = balance / (1.00 + i4) ** g
            npvi4.append(NPV)
            finalb = finalb + NPV
            finalbi4.append(finalb)
            prof1 = prof
            OPEX1 = OPEX
            OPEXi4.append(OPEX)
            mfi4.append(marketf)
            bli4.append(balance)
            pri4.append(prof)
        previous_values4 = [n for n in finalbi4]  # create a copy of the original list to track previous values
        for i, j in zip(range(len(finalbi4)), T):
            if i > 0 and finalbi4[i] > 0 and previous_values4[i - 1] < 0:
                emptylabel9.config(text=str(math.trunc(j)))
            previous_values4[i] = finalbi4[i]  # update the previous_values list with the current value
        positive_sum4 = 0
        for num in finalbi4:
            if num > 0:
                positive_sum4 += num
        emptylabel8.config(text=str(math.trunc(positive_sum4)))
        emptylabel1.config(text="The power and capacity purchased are " + str(
            r) + " MW and " + str(c) + " MWh, " + " respectively")
        # Create two series
        x=numpy.array(T)
        y1 = numpy.array(finalbi1)
        y2 = numpy.array(finalbi2)
        y3 = numpy.array(finalbi3)
        y4 = numpy.array(finalbi4)

        # Set the width of the bars
        width = 0.2

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the four series as grouped bars
        ax.bar(x- (1.5*width), y1, width, label='i=1%',color="g")
        ax.bar(x- (0.5*width), y2, width, label='i=3%',color="b")
        ax.bar(x+ (0.5*width), y3, width, label='i=5%',color="y")
        ax.bar(x+ (1.5*width), y4, width, label='i=7%',color="r")

        # Add x-axis tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(x)

        # Add a legend
        ax.legend()

        # Add axis labels
        ax.set_xlabel('Years')
        ax.set_ylabel('SEK')
        ax.set_title("Project's cashflow")
        # Show the plot
        plt.show()
        u = list(range(0,120))
        p=energys[0:120]
        plt.plot(u, p)
        plt.xlabel('Hours')
        plt.ylabel('Battery capacity (MWh)')
        plt.title("Battery operation for the first week")
        plt.show()
        # Create a list of 8 share values
        battsh = m * 0.5978
        peesh = m * 0.2105
        staffsh = m * 0.1014
        materialssh = m * 0.0199
        constrush = m * 0.0238
        subsh = m * 0.0209
        missh = m * 0.002
        persh = m * 0.0238
        shares = [battsh, peesh, staffsh, materialssh, constrush, subsh, missh, persh]

        # Create a list of 8 colors for the shares
        labels = ['Battery', 'PCS', 'Staff', 'Materials', 'Construction', 'Subcontractors', 'Miscellaneous',
                  'Personnel']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

        # Create a pie chart with the shares and colors
        plt.pie(shares, colors=colors, labels=labels,
                autopct=lambda pct: f'{pct:.2f}% ({int((pct * sum(shares)) / 100):,})')

        # Add a title to the pie chart
        plt.title('CAPEX distribution (SEK)')
        plt.legend(loc='best')

        # Show the plot
        plt.show()
    if n=="Peak shaving":
        a = sn.get()
        cc = tn.get()
        d = float(var3.get())
        e = var2.get()
        f = 0
        g = var1.get()
        ep=float(zn.get())
        xs = ratio / 1000
        maxsoc = 0.95
        r = m / xs  # kW
        c = (r * d)  # kWh
        p = c - (cc*d)
        eff=0.95
        if p < 0 and c>(a*float(d)):
            T = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * c * float(g) * 52 *eff *maxsoc
            energy_penalty = ((ep / 3) * c * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            energy_penalty1=0
            OPEXi1 = []
            mfi1 = []
            bli1 = []
            pri1 = []
            npvi1 = []
            finalbi1 = []
            i1 = 0.01
            i2 = 0.03
            i3 = 0.05
            i4 = 0.07
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty+(energy_penalty1*inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i1) ** g
                npvi1.append(NPV)
                finalb = finalb + NPV
                finalbi1.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1=energy_penalty
                OPEXi1.append(OPEX)
                mfi1.append(energy_penalty)
                bli1.append(balance)
                pri1.append(prof)
            previous_values1 = [n for n in finalbi1]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi1)), T):
                if i > 0 and finalbi1[i] > 0 and previous_values1[i - 1] < 0:
                    emptylabel3.config(text=str(math.trunc(j)))
                previous_values1[i] = finalbi1[i]  # update the previous_values list with the current value
            positive_sum1 = 0
            for num in finalbi1:
                if num > 0:
                    positive_sum1 += num
            emptylabel2.config(text=str(math.trunc(positive_sum1)))
            OPEXi2 = []
            mfi2 = []
            bli2 = []
            pri2 = []
            npvi2 = []
            finalbi2 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * c * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * c * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            energy_penalty1 = 0
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty+(energy_penalty1*inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i2) ** g
                npvi2.append(NPV)
                finalb = finalb + NPV
                finalbi2.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi2.append(OPEX)
                mfi2.append(energy_penalty)
                bli2.append(balance)
                pri2.append(prof)
            previous_values2 = [n for n in finalbi2]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi2)), T):
                if i > 0 and finalbi2[i] > 0 and previous_values2[i - 1] < 0:
                    emptylabel5.config(text=str(math.trunc(j)))
                previous_values2[i] = finalbi2[i]  # update the previous_values list with the current value
            positive_sum2 = 0
            for num in finalbi2:
                if num > 0:
                    positive_sum2 += num
            emptylabel4.config(text=str(math.trunc(positive_sum2)))
            OPEXi3 = []
            mfi3 = []
            bli3 = []
            pri3 = []
            npvi3 = []
            finalbi3 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * c * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * c * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            energy_penalty1 = 0
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty+(energy_penalty1*inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i3) ** g
                npvi3.append(NPV)
                finalb = finalb + NPV
                finalbi3.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi3.append(OPEX)
                mfi3.append(energy_penalty)
                bli3.append(balance)
                pri3.append(prof)
            previous_values3 = [n for n in finalbi3]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi3)), T):
                if i > 0 and finalbi3[i] > 0 and previous_values3[i - 1] < 0:
                    emptylabel7.config(text=str(math.trunc(j)))
                previous_values3[i] = finalbi3[i]  # update the previous_values list with the current value
            positive_sum3 = 0
            for num in finalbi3:
                if num > 0:
                    positive_sum3 += num
            emptylabel6.config(text=str(math.trunc(positive_sum3)))
            OPEXi4 = []
            mfi4 = []
            bli4 = []
            pri4 = []
            npvi4 = []
            finalbi4 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * c * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * c * maxsoc * float(g) * 52) / eff
            energy_penalty1 = 0
            inf = 0.013
            fixedop = 0.01
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty+(energy_penalty1*inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i4) ** g
                npvi4.append(NPV)
                finalb = finalb + NPV
                finalbi4.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi4.append(OPEX)
                mfi4.append(energy_penalty)
                bli4.append(balance)
                pri4.append(prof)
            previous_values4 = [n for n in finalbi4]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi4)), T):
                if i > 0 and finalbi4[i] > 0 and previous_values4[i - 1] < 0:
                    emptylabel9.config(text=str(math.trunc(j)))
                previous_values4[i] = finalbi4[i]  # update the previous_values list with the current value
            positive_sum4 = 0
            for num in finalbi4:
                if num > 0:
                    positive_sum4 += num
            emptylabel8.config(text=str(math.trunc(positive_sum4)))
            emptylabel1.config(text="The power and capacity purchased are " + str(
                r) + " kW and " + str(c) + " kWh, " + " respectively")
            # Create two series
            x = numpy.array(T)
            y1 = numpy.array(finalbi1)
            y2 = numpy.array(finalbi2)
            y3 = numpy.array(finalbi3)
            y4 = numpy.array(finalbi4)
            # Set the width of the bars
            width = 0.2

            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the four series as grouped bars
            ax.bar(x - (1.5 * width), y1, width, label='i=1%', color="g")
            ax.bar(x - (0.5 * width), y2, width, label='i=3%', color="b")
            ax.bar(x + (0.5 * width), y3, width, label='i=5%', color="y")
            ax.bar(x + (1.5 * width), y4, width, label='i=7%', color="r")

            # Add x-axis tick labels
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.set_title("Project's cashflow")
            # Add a legend
            ax.legend()

            # Add axis labels
            ax.set_xlabel('Years')
            ax.set_ylabel('SEK')
            # Show the plot
            plt.show()
            load = []
            load1 = []
            if d>=1:
                for i in range(24):
                    if i < e:
                        l = a
                        f = 0
                    elif i == e:
                        l = cc
                        f = 1
                    elif f > 0 and f < d:
                        l = cc
                        f = f + 1
                    elif f == d:
                        l = a
                        f = 0
                    load.append(l)
                gg = []
                if g == 1:
                    for i in range(96):
                        gg.append(a)
                    ll = load + gg
                if g == 2:
                    for i in range(72):
                        gg.append(a)
                    ll = (load * 2) + gg
                if g == 3:
                    for i in range(48):
                        gg.append(a)
                    ll = (load * 3) + gg
                if g == 4:
                    for i in range(24):
                        gg.append(a)
                    ll = (load * 4) + gg
                if g == 5:
                    ll = load * 5
                for i in range(24):
                    if i < e:
                        la = a
                        f = 0
                    elif i == e:
                        la = r
                        f = 1
                    elif f > 0 and f < d:
                        la = r
                        f = f + 1
                    elif f == d:
                        la = a
                        f = 0
                    load1.append(la)
                hh = []
                if g == 1:
                    for i in range(96):
                        hh.append(a)
                    xx = load1 + hh
                if g == 2:
                    for i in range(72):
                        hh.append(a)
                    xx = (load1 * 2) + hh
                if g == 3:
                    for i in range(48):
                        hh.append(a)
                    xx = (load1 * 3) + hh
                if g == 4:
                    for i in range(24):
                        hh.append(a)
                    xx = (load1 * 4) + hh
                if g == 5:
                    xx = load1 * 5
                # Plot the values of ll and xx against the time axis
                fig, ax = plt.subplots()
                ax.plot(range(120), ll, label='Load')
                ax.plot(range(120), xx, label='Battery operation')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('Load (kW)')
                ax.set_title('Load profile for one week')
                ax.legend()
                plt.show()
                fig, ax = plt.subplots()
                lll = []
                for i in xx:
                    if i == r:
                        sha = 0
                    else:
                        sha = 100
                    lll.append(sha)
                ax.plot(range(120), lll, label='State of charge')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('SoC (%)')
                ax.set_title('SoC profile for one week')
                ax.legend()
                plt.show()
            else:
                for i in range(24):
                    if i < e:
                        l = a
                        f = 0
                    elif i == e:
                        l = cc
                        f = 0.5
                    elif f == d:
                        l = a
                        f = 0
                    load.append(l)
                gg = []
                if g == 1:
                    for i in range(96):
                        gg.append(a)
                    ll = load + gg
                if g == 2:
                    for i in range(72):
                        gg.append(a)
                    ll = (load * 2) + gg
                if g == 3:
                    for i in range(48):
                        gg.append(a)
                    ll = (load * 3) + gg
                if g == 4:
                    for i in range(24):
                        gg.append(a)
                    ll = (load * 4) + gg
                if g == 5:
                    ll = load * 5
                for i in range(24):
                    if i < e:
                        la = a
                        f = 0
                    elif i == e:
                        la = r
                        f = 0.5
                    elif f == d:
                        la = a
                        f = 0
                    load1.append(la)
                hh = []
                if g == 1:
                    for i in range(96):
                        hh.append(a)
                    xx = load1 + hh
                if g == 2:
                    for i in range(72):
                        hh.append(a)
                    xx = (load1 * 2) + hh
                if g == 3:
                    for i in range(48):
                        hh.append(a)
                    xx = (load1 * 3) + hh
                if g == 4:
                    for i in range(24):
                        hh.append(a)
                    xx = (load1 * 4) + hh
                if g == 5:
                    xx = load1 * 5
                # Plot the values of ll and xx against the time axis
                fig, ax = plt.subplots()
                ax.plot(range(120), ll, label='Load')
                ax.plot(range(120), xx, label='Battery operation')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('Load (kW)')
                ax.set_title('Load profile for one week')
                ax.legend()
                plt.show()
                fig, ax = plt.subplots()
                lll = []
                for i in xx:
                    if i == r:
                        sha = 0
                    else:
                        sha = 100
                    lll.append(sha)
                ax.plot(range(120), lll, label='State of charge')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('SoC (%)')
                ax.set_title('SoC profile for one week')
                ax.legend()
                plt.show()
            battsh = m * 0.5978
            peesh = m * 0.2105
            staffsh = m * 0.1014
            materialssh = m * 0.0199
            constrush = m * 0.0238
            subsh = m * 0.0209
            missh = m * 0.002
            persh = m * 0.0238
            shares = [battsh, peesh, staffsh, materialssh, constrush, subsh, missh, persh]

            # Create a list of 8 colors for the shares
            labels = ['Battery', 'PCS', 'Staff', 'Materials', 'Construction', 'Subcontractors', 'Miscellaneous',
                      'Personnel']
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:gray']

            # Create a pie chart with the shares and colors
            plt.pie(shares, colors=colors, labels=labels,
                    autopct=lambda pct: f'{pct:.2f}% ({int((pct * sum(shares)) / 100):,})')

            # Add a title to the pie chart
            plt.title('CAPEX distribution (SEK)')
            plt.legend(loc='best')

            # Show the plot
            plt.show()
        elif p > 0 and c>(a*float(d)):
            opw=((cc*0.05)+cc)*float(d)
            opwh = (opw*(xs))/float(d)
            T = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * (cc * float(d)) * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * (cc * float(d)) * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            energy_penalty1 = 0
            OPEXi1 = []
            mfi1 = []
            bli1 = []
            pri1 = []
            npvi1 = []
            finalbi1 = []
            i1 = 0.01
            i2 = 0.03
            i3 = 0.05
            i4 = 0.07
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty + (energy_penalty1 * inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i1) ** g
                npvi1.append(NPV)
                finalb = finalb + NPV
                finalbi1.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi1.append(OPEX)
                mfi1.append(energy_penalty)
                bli1.append(balance)
                pri1.append(prof)
            previous_values1 = [n for n in finalbi1]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi1)), T):
                if i > 0 and finalbi1[i] > 0 and previous_values1[i - 1] < 0:
                    emptylabel3.config(text=str(math.trunc(j)))
                previous_values1[i] = finalbi1[i]  # update the previous_values list with the current value
            positive_sum1 = 0
            for num in finalbi1:
                if num > 0:
                    positive_sum1 += num
            emptylabel2.config(text=str(math.trunc(positive_sum1)))
            OPEXi2 = []
            mfi2 = []
            bli2 = []
            pri2 = []
            npvi2 = []
            finalbi2 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * (cc * float(d)) * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * (cc * float(d)) * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            energy_penalty1 = 0
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty + (energy_penalty1 * inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i2) ** g
                npvi2.append(NPV)
                finalb = finalb + NPV
                finalbi2.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi2.append(OPEX)
                mfi2.append(energy_penalty)
                bli2.append(balance)
                pri2.append(prof)
            previous_values2 = [n for n in finalbi2]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi2)), T):
                if i > 0 and finalbi2[i] > 0 and previous_values2[i - 1] < 0:
                    emptylabel5.config(text=str(math.trunc(j)))
                previous_values2[i] = finalbi2[i]  # update the previous_values list with the current value
            positive_sum2 = 0
            for num in finalbi2:
                if num > 0:
                    positive_sum2 += num
            emptylabel4.config(text=str(math.trunc(positive_sum2)))
            OPEXi3 = []
            mfi3 = []
            bli3 = []
            pri3 = []
            npvi3 = []
            finalbi3 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * (cc * float(d)) * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * (cc * float(d)) * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            energy_penalty1 = 0
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty + (energy_penalty1 * inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i3) ** g
                npvi3.append(NPV)
                finalb = finalb + NPV
                finalbi3.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi3.append(OPEX)
                mfi3.append(energy_penalty)
                bli3.append(balance)
                pri3.append(prof)
            previous_values3 = [n for n in finalbi3]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi3)), T):
                if i > 0 and finalbi3[i] > 0 and previous_values3[i - 1] < 0:
                    emptylabel7.config(text=str(math.trunc(j)))
                previous_values3[i] = finalbi3[i]  # update the previous_values list with the current value
            positive_sum3 = 0
            for num in finalbi3:
                if num > 0:
                    positive_sum3 += num
            emptylabel6.config(text=str(math.trunc(positive_sum3)))
            OPEXi4 = []
            mfi4 = []
            bli4 = []
            pri4 = []
            npvi4 = []
            finalbi4 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * (cc * float(d)) * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * (cc * float(d)) * maxsoc * float(g) * 52) / eff
            energy_penalty1 = 0
            inf = 0.013
            fixedop = 0.01
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty + (energy_penalty1 * inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i4) ** g
                npvi4.append(NPV)
                finalb = finalb + NPV
                finalbi4.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi4.append(OPEX)
                mfi4.append(energy_penalty)
                bli4.append(balance)
                pri4.append(prof)
            previous_values4 = [n for n in finalbi4]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi4)), T):
                if i > 0 and finalbi4[i] > 0 and previous_values4[i - 1] < 0:
                    emptylabel9.config(text=str(math.trunc(j)))
                previous_values4[i] = finalbi4[i]  # update the previous_values list with the current value
            positive_sum4 = 0
            for num in finalbi4:
                if num > 0:
                    positive_sum4 += num
            emptylabel8.config(text=str(math.trunc(positive_sum4)))
            emptylabel1.config(text="According to your load, the optimal BESS size for your company is " + str(
                opw) + " kWh instead of " + str(c) + " kWh" + " with an investment of " + str(
                math.trunc(opwh)) + " SEK")
            emptylabel10.config(text="You are currently using only the " + str(math.trunc(((cc*d)/c)*100)) + " % of the battery you purchased")
            # Create two series
            x = numpy.array(T)
            y1 = numpy.array(finalbi1)
            y2 = numpy.array(finalbi2)
            y3 = numpy.array(finalbi3)
            y4 = numpy.array(finalbi4)

            # Set the width of the bars
            width = 0.2

            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the four series as grouped bars
            ax.bar(x - (1.5 * width), y1, width, label='i=1%', color="g")
            ax.bar(x - (0.5 * width), y2, width, label='i=3%', color="b")
            ax.bar(x + (0.5 * width), y3, width, label='i=5%', color="y")
            ax.bar(x + (1.5 * width), y4, width, label='i=7%', color="r")

            # Add x-axis tick labels
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.set_title("Project's cashflow")
            # Add a legend
            ax.legend()

            # Add axis labels
            ax.set_xlabel('Years')
            ax.set_ylabel('SEK')
            # Show the plot
            plt.show()
            load = []
            if d>=1:
                for i in range(24):
                    if i < e:
                        l = a
                        f = 0
                    elif i == e:
                        l = cc
                        f = 1
                    elif f > 0 and f < d:
                        l = cc
                        f = f + 1
                    elif f == d:
                        l = a
                        f = 0
                    load.append(l)
                gg = []
                if g == 1:
                    for i in range(96):
                        gg.append(a)
                    ll = load + gg
                if g == 2:
                    for i in range(72):
                        gg.append(a)
                    ll = (load * 2) + gg
                if g == 3:
                    for i in range(48):
                        gg.append(a)
                    ll = (load * 3) + gg
                if g == 4:
                    for i in range(24):
                        gg.append(a)
                    ll = (load * 4) + gg
                if g == 5:
                    ll = load * 5
                fig, ax = plt.subplots()
                ax.plot(range(120), ll, label='Battery operation')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('Load (kW)')
                ax.set_title('Load profile for one week')
                ax.legend()
                plt.show()
                fig, ax = plt.subplots()
                lll = []
                for i in ll:
                    if i == cc:
                        sha = 0
                    else:
                        sha = 100
                    lll.append(sha)
                ax.plot(range(120), lll, label='State of charge')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('SoC (%)')
                ax.set_title('SoC profile for one week')
                ax.legend()
                plt.show()
            else:
                for i in range(24):
                    if i < e:
                        l = a
                        f = 0
                    elif i == e:
                        l = cc
                        f = 0.5
                    elif f == d:
                        l = a
                        f = 0
                    load.append(l)
                gg = []
                if g == 1:
                    for i in range(96):
                        gg.append(a)
                    ll = load + gg
                if g == 2:
                    for i in range(72):
                        gg.append(a)
                    ll = (load * 2) + gg
                if g == 3:
                    for i in range(48):
                        gg.append(a)
                    ll = (load * 3) + gg
                if g == 4:
                    for i in range(24):
                        gg.append(a)
                    ll = (load * 4) + gg
                if g == 5:
                    ll = load * 5
                fig, ax = plt.subplots()
                ax.plot(range(120), ll, label='Battery operation')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('Load (kW)')
                ax.set_title('Load profile for one week')
                ax.legend()
                plt.show()
                fig, ax = plt.subplots()
                lll = []
                for i in ll:
                    if i == cc:
                        sha = 0
                    else:
                        sha = 100
                    lll.append(sha)
                ax.plot(range(120), lll, label='State of charge')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('SoC (%)')
                ax.set_title('SoC profile for one week')
                ax.legend()
                plt.show()
            battsh = m * 0.5978
            peesh = m * 0.2105
            staffsh = m * 0.1014
            materialssh = m * 0.0199
            constrush = m * 0.0238
            subsh = m * 0.0209
            missh = m * 0.002
            persh = m * 0.0238
            shares = [battsh, peesh, staffsh, materialssh, constrush, subsh, missh, persh]

            # Create a list of 8 colors for the shares
            labels = ['Battery', 'PCS', 'Staff', 'Materials', 'Construction', 'Subcontractors', 'Miscellaneous',
                      'Personnel']
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:gray']

            # Create a pie chart with the shares and colors
            plt.pie(shares, colors=colors, labels=labels,
                    autopct=lambda pct: f'{pct:.2f}% ({int((pct * sum(shares)) / 100):,})')

            # Add a title to the pie chart
            plt.title('CAPEX distribution (SEK)')
            plt.legend(loc='best')

            # Show the plot
            plt.show()
        elif p == 0 and c>(a*float(d)):
            opw = ((cc * 0.05) + cc) * float(d)
            opwh = (opw * (xs)) / float(d)
            T = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * (cc * float(d)) * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * (cc * float(d)) * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            energy_penalty1 = 0
            OPEXi1 = []
            mfi1 = []
            bli1 = []
            pri1 = []
            npvi1 = []
            finalbi1 = []
            i1 = 0.01
            i2 = 0.03
            i3 = 0.05
            i4 = 0.07
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty + (energy_penalty1 * inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i1) ** g
                npvi1.append(NPV)
                finalb = finalb + NPV
                finalbi1.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi1.append(OPEX)
                mfi1.append(energy_penalty)
                bli1.append(balance)
                pri1.append(prof)
            previous_values1 = [n for n in finalbi1]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi1)), T):
                if i > 0 and finalbi1[i] > 0 and previous_values1[i - 1] < 0:
                    emptylabel3.config(text=str(math.trunc(j)))
                previous_values1[i] = finalbi1[i]  # update the previous_values list with the current value
            positive_sum1 = 0
            for num in finalbi1:
                if num > 0:
                    positive_sum1 += num
            emptylabel2.config(text=str(math.trunc(positive_sum1)))
            OPEXi2 = []
            mfi2 = []
            bli2 = []
            pri2 = []
            npvi2 = []
            finalbi2 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * (cc * float(d)) * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * (cc * float(d)) * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            energy_penalty1 = 0
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty + (energy_penalty1 * inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i2) ** g
                npvi2.append(NPV)
                finalb = finalb + NPV
                finalbi2.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi2.append(OPEX)
                mfi2.append(energy_penalty)
                bli2.append(balance)
                pri2.append(prof)
            previous_values2 = [n for n in finalbi2]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi2)), T):
                if i > 0 and finalbi2[i] > 0 and previous_values2[i - 1] < 0:
                    emptylabel5.config(text=str(math.trunc(j)))
                previous_values2[i] = finalbi2[i]  # update the previous_values list with the current value
            positive_sum2 = 0
            for num in finalbi2:
                if num > 0:
                    positive_sum2 += num
            emptylabel4.config(text=str(math.trunc(positive_sum2)))
            OPEXi3 = []
            mfi3 = []
            bli3 = []
            pri3 = []
            npvi3 = []
            finalbi3 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * (cc * float(d)) * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * (cc * float(d)) * maxsoc * float(g) * 52) / eff
            inf = 0.013
            fixedop = 0.01
            energy_penalty1 = 0
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty + (energy_penalty1 * inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i3) ** g
                npvi3.append(NPV)
                finalb = finalb + NPV
                finalbi3.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi3.append(OPEX)
                mfi3.append(energy_penalty)
                bli3.append(balance)
                pri3.append(prof)
            previous_values3 = [n for n in finalbi3]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi3)), T):
                if i > 0 and finalbi3[i] > 0 and previous_values3[i - 1] < 0:
                    emptylabel7.config(text=str(math.trunc(j)))
                previous_values3[i] = finalbi3[i]  # update the previous_values list with the current value
            positive_sum3 = 0
            for num in finalbi3:
                if num > 0:
                    positive_sum3 += num
            emptylabel6.config(text=str(math.trunc(positive_sum3)))
            OPEXi4 = []
            mfi4 = []
            bli4 = []
            pri4 = []
            npvi4 = []
            finalbi4 = []
            CAPEX = m - ((2 - float(d)) * (m * 0.5))
            prof = ep * (cc * float(d)) * float(g) * 52 * eff * maxsoc
            energy_penalty = ((ep / 3) * (cc * float(d)) * maxsoc * float(g) * 52) / eff
            energy_penalty1 = 0
            inf = 0.013
            fixedop = 0.01
            finalb = -CAPEX
            OPEX = CAPEX * fixedop
            OPEX1 = 0
            prof1 = 0
            for u in T:
                OPEX = OPEX + (OPEX1 * inf)
                prof = prof + (prof1 * inf)
                energy_penalty = energy_penalty + (energy_penalty1 * inf)
                balance = prof - OPEX - energy_penalty
                NPV = balance / (1.00 + i4) ** g
                npvi4.append(NPV)
                finalb = finalb + NPV
                finalbi4.append(finalb)
                prof1 = prof
                OPEX1 = OPEX
                energy_penalty1 = energy_penalty
                OPEXi4.append(OPEX)
                mfi4.append(energy_penalty)
                bli4.append(balance)
                pri4.append(prof)
            previous_values4 = [n for n in finalbi4]  # create a copy of the original list to track previous values
            for i, j in zip(range(len(finalbi4)), T):
                if i > 0 and finalbi4[i] > 0 and previous_values4[i - 1] < 0:
                    emptylabel9.config(text=str(math.trunc(j)))
                previous_values4[i] = finalbi4[i]  # update the previous_values list with the current value
            positive_sum4 = 0
            for num in finalbi4:
                if num > 0:
                    positive_sum4 += num
            emptylabel8.config(text=str(math.trunc(positive_sum4)))
            emptylabel1.config(text="The power and capacity purchased are " + str(
                r) + " kW and " + str(c) + " kWh, " + " respectively")
            # Create two series
            x = numpy.array(T)
            y1 = numpy.array(finalbi1)
            y2 = numpy.array(finalbi2)
            y3 = numpy.array(finalbi3)
            y4 = numpy.array(finalbi4)

            # Set the width of the bars
            width = 0.2

            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the four series as grouped bars
            ax.bar(x - (1.5 * width), y1, width, label='i=1%', color="g")
            ax.bar(x - (0.5 * width), y2, width, label='i=3%', color="b")
            ax.bar(x + (0.5 * width), y3, width, label='i=5%', color="y")
            ax.bar(x + (1.5 * width), y4, width, label='i=7%', color="r")

            # Add x-axis tick labels
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.set_title("Project's cashflow")
            # Add a legend
            ax.legend()

            # Add axis labels
            ax.set_xlabel('Years')
            ax.set_ylabel('SEK')
            # Show the plot
            plt.show()
            load = []
            if d >= 1:
                for i in range(24):
                    if i < e:
                        l = a
                        f = 0
                    elif i == e:
                        l = cc
                        f = 1
                    elif f > 0 and f < d:
                        l = cc
                        f = f + 1
                    elif f == d:
                        l = a
                        f = 0
                    load.append(l)
                gg = []
                if g == 1:
                    for i in range(96):
                        gg.append(a)
                    ll = load + gg
                if g == 2:
                    for i in range(72):
                        gg.append(a)
                    ll = (load * 2) + gg
                if g == 3:
                    for i in range(48):
                        gg.append(a)
                    ll = (load * 3) + gg
                if g == 4:
                    for i in range(24):
                        gg.append(a)
                    ll = (load * 4) + gg
                if g == 5:
                    ll = load * 5
                fig, ax = plt.subplots()
                ax.plot(range(120), ll, label='Battery operation')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('Load (kW)')
                ax.set_title('Load profile for one week')
                ax.legend()
                plt.show()
                fig, ax = plt.subplots()
                lll = []
                for i in ll:
                    if i == cc:
                        sha = 0
                    else:
                        sha = 100
                    lll.append(sha)
                ax.plot(range(120), lll, label='State of charge')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('SoC (%)')
                ax.set_title('SoC profile for one week')
                ax.legend()
                plt.show()
            else:
                for i in range(24):
                    if i < e:
                        l = a
                        f = 0
                    elif i == e:
                        l = cc
                        f = 0.5
                    elif f == d:
                        l = a
                        f = 0
                    load.append(l)
                gg = []
                if g == 1:
                    for i in range(96):
                        gg.append(a)
                    ll = load + gg
                if g == 2:
                    for i in range(72):
                        gg.append(a)
                    ll = (load * 2) + gg
                if g == 3:
                    for i in range(48):
                        gg.append(a)
                    ll = (load * 3) + gg
                if g == 4:
                    for i in range(24):
                        gg.append(a)
                    ll = (load * 4) + gg
                if g == 5:
                    ll = load * 5
                fig, ax = plt.subplots()
                ax.plot(range(120), ll, label='Battery operation')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('Load (kW)')
                ax.set_title('Load profile for one week')
                ax.legend()
                plt.show()
                fig, ax = plt.subplots()
                lll = []
                for i in ll:
                    if i == cc:
                        sha = 0
                    else:
                        sha = 100
                    lll.append(sha)
                ax.plot(range(120), lll, label='State of charge')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('SoC (%)')
                ax.set_title('SoC profile for one week')
                ax.legend()
                plt.show()
            battsh = m * 0.5978
            peesh = m * 0.2105
            staffsh = m * 0.1014
            materialssh = m * 0.0199
            constrush = m * 0.0238
            subsh = m * 0.0209
            missh = m * 0.002
            persh = m * 0.0238
            shares = [battsh, peesh, staffsh, materialssh, constrush, subsh, missh, persh]

            # Create a list of 8 colors for the shares
            labels = ['Battery', 'PCS', 'Staff', 'Materials', 'Construction', 'Subcontractors', 'Miscellaneous',
                      'Personnel']
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:gray']

            # Create a pie chart with the shares and colors
            plt.pie(shares, colors=colors, labels=labels,
                    autopct=lambda pct: f'{pct:.2f}% ({int((pct * sum(shares)) / 100):,})')

            # Add a title to the pie chart
            plt.title('CAPEX distribution (SEK)')
            plt.legend(loc='best')

            # Show the plot
            plt.show()
        elif c<(a*float(d)):
            messagebox.showerror('Sizing error', "Error: Increase the budget, the power of the battery is lower than your average consumption!")
image=Image.open("C:/Users/santi/Downloads/download-removebg-preview.png")
photo=ImageTk.PhotoImage(image)
label=Label(image=photo)
label.pack()
label2=Label(root,text="Budget (SEK) (without special characters)",font=("bold",10),wraplength=100)
label2.place(x=135,y=110)
entry1=Entry(root,textvar=fn)
entry1.place(x=320,y=115)
label12=Label(root,text="RESULTS",font=("bold",20),wraplength=200)
label12.place(x=800,y=105)
label13=Label(root,text="Interest rate (i)",font=("bold",13),wraplength=100)
label13.place(x=650,y=300)
label14=Label(root,text="Total profit in the project's lifetime (SEK)",font=("bold",13),wraplength=100)
label14.place(x=810,y=300)
label15=Label(root,text="Payback time (years)",font=("bold",13),wraplength=70)
label15.place(x=980,y=303)
label16=Label(root,text="1%",font=("bold",13),wraplength=100)
label16.place(x=680,y=400)
label17=Label(root,text="3%",font=("bold",13),wraplength=100)
label17.place(x=680,y=470)
label18=Label(root,text="5%",font=("bold",13),wraplength=100)
label18.place(x=680,y=540)
label19=Label(root,text="7%",font=("bold",13),wraplength=100)
label19.place(x=680,y=610)
label3=Label(root,text="Type of service",font=("bold",10))
label3.place(x=140,y=190)
r1=Radiobutton(root,text="Frequency regulation",variable=rad,value="Frequency regulation") .place(x=320,y=170)
r2=Radiobutton(root,text="Arbitrage",variable=rad,value="Arbitrage") .place(x=320,y=190)
r3=Radiobutton(root,text="Peak shaving",variable=rad,value="Peak shaving") .place(x=320,y=210)
label4=Label(root,text="In case of peak shaving",font=("bold",12))
label4.place(x=170,y=260)
label11=Label(root,text="Energy penalty (SEK/kWh)",font=("bold",10))
label11.place(x=120,y=307)
entry7=Entry(root,textvar=zn)
entry7.place(x=320,y=310)
label9=Label(root,text="Average consumption (kW/day)",font=("bold",10))
label9.place(x=107,y=344)
entry5=Entry(root,textvar=sn)
entry5.place(x=320,y=347)
label5=Label(root,text="Electricity consumption limit (kW)",font=("bold",10))
label5.place(x=95,y=380)
entry2=Entry(root,textvar=ln)
entry2.place(x=320,y=383)
label6=Label(root,text="Consumption peak (kW/day)",font=("bold",10))
label6.place(x=113,y=417)
entry3=Entry(root,textvar=tn)
entry3.place(x=320,y=420)
label10=Label(root,text="Hour of the day at which peak consumption starts",font=("bold",10),wraplength=200)
label10.place(x=102,y=452)
list2=["00","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23"]
droplist=OptionMenu(root,var2,*list2)
var2.set("select option")
droplist.config(width=20)
droplist.place(x=300,y=457)
label7=Label(root,text=" For how many hours do you exceed your limit in a day?",font=("bold",10),wraplength=200)
label7.place(x=110,y=495)
list3=["0.5","1","2"]
droplist=OptionMenu(root,var3,*list3)
var3.set("select option")
droplist.config(width=20)
droplist.place(x=300,y=500)
label8=Label(root,text="Number of overconsumption days in a week",font=("bold",10),wraplength=200)
label8.place(x=95,y=540)
list1=["1","2","3","4","5"]
droplist=OptionMenu(root,var1,*list1)
var1.set("select option")
droplist.config(width=20)
droplist.place(x=300,y=540)
but1=Button(root,text="Calculate",command=val) .place(x=225,y=620)
emptylabel1=Label(root,font=("bold",13),wraplength=300)
emptylabel1.place(x=720,y=150)
emptylabel10=Label(root,font=("bold",13),wraplength=300)
emptylabel10.place(x=722,y=245)
emptylabel2=Label(root,font=("bold",13),wraplength=300)
emptylabel2.place(x=828,y=400)
emptylabel3=Label(root,font=("bold",13),wraplength=300)
emptylabel3.place(x=1005,y=400)
emptylabel4=Label(root,font=("bold",13),wraplength=300)
emptylabel4.place(x=828,y=470)
emptylabel5=Label(root,font=("bold",13),wraplength=300)
emptylabel5.place(x=1005,y=470)
emptylabel6=Label(root,font=("bold",13),wraplength=300)
emptylabel6.place(x=828,y=540)
emptylabel7=Label(root,font=("bold",13),wraplength=300)
emptylabel7.place(x=1005,y=540)
emptylabel8=Label(root,font=("bold",13),wraplength=300)
emptylabel8.place(x=828,y=610)
emptylabel9=Label(root,font=("bold",13),wraplength=300)
emptylabel9.place(x=1005,y=610)
root.mainloop()
