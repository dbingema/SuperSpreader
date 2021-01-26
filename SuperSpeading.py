import streamlit as st
import numpy as np
import pandas as pd
import math

import random
from pandas.core.common import flatten
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import plasma


"""

# Introduction

Corona virus infections seem to spread almost randomly. With no rhyme or reason. Hot spots pop up quickly out of nowhere in remote places and experts struggle to explain why there, why then, why so many.


# Infection Spread Distribution

We model the number of new infections caused by a single person in their neighborhood with a log normal function. The log normal function is defined by:

$$

f(x) = A \exp \left( - \ln(2) \left( \\frac{ \ln ( 1+2b \; (x-x_0)/\Delta )}{b} \\right) ^2 \\right)

$$

if the argument of the inner $\ln$ is larger than zero, otherwise the
function is zero.

Here the parameter $x_0$ describe the position of the peak, $\Delta$ its
width, and $b$ the asymmetry. 

The position defines the peak maximum
independent of the other parameters, the width measures the FWHM for the
equivalent symmetric peak directly in pixels (thanks to the factor of
$\ln(2)$ in the equation), and the asymmetry goes from negative values
(tailing left) to positive values (tailing right). An asymmetry of zero
describes a symmetric, Gaussian, peak. Asymmetry will increase the actual
FWHM over the value set by the width parameter, about 10% for an asymmetry
of 1.0.

"""


def logNormal(x, x0, Delta, b):
   # prevent 0/0
   eps = 1e-5
   if (abs(b) < eps): 
       b = eps

   if type(x) is list:
       arg = [1+2*b*(xi-x0)/Delta for xi in x]
       fx = [0 if (argi < eps) else math.exp(-math.log(2)*(math.log(argi)/b)**2) for argi in arg]
   else:
       arg = 1+2*b*(x-x0)/Delta
       fx = 0 if (arg < eps) else math.exp(-math.log(2)*(math.log(arg)/b)**2)
   return(fx)


# find integer random number with this distribution through inverse of cumulative function

def randomIntLogNormal(n, x0, Delta, b): 
  halfBin = 0.5
  maxX = max(200, round(20* x0 * Delta * b))
  xSim = range(maxX)
  prob = [logNormal(x + halfBin, x0, Delta, b) for x in xSim]
  cumulative = np.cumsum(prob)
  cumulative = cumulative / cumulative[-1]
  ranNum = [random.random() for i in range(n)]
  position = np.searchsorted(cumulative, ranNum)
  ranLogNorm = [xSim[i] for i in position]
  return ranLogNorm


"""

A possible spread of new infections per infectious person is given by this distribution, for example, with a peak position, $x_0$, of 1, a width, $\Delta$, of 1, and a right tail with an asymmetry, $b$, of 1 we get the following distribution:

"""



xPos = range(200)
halfBin = 0.5
xPosMid = [xi + halfBin for xi in xPos]

# quick hot spots, R = 1
#x0 <- 0.2
#Delta <- 0.01
#b <- 2.4

# not quite as slow hot spots
#x0 = 1
#Delta = 1.725
#b = 0.5

st.sidebar.write('### Parameters Describing Distribution of Number of Infected People')

x0 = st.sidebar.slider('Maximum Position', 0.0, 4.0, 1.0)
Delta = st.sidebar.slider('Width', 0.01, 3.0, 1.725)
b = st.sidebar.slider('Asymmetry', 0.0, 2.0, 0.5)


# slow hot spots
#x0 <- 1.2
#Delta <- 1.5
#b <- 0.35


# no hot spots, R = 1
#x0 <- 1.5
#Delta <- 1
#b <- 0


infections = pd.DataFrame({'Num': xPos,
                          'Prob': logNormal(xPosMid, x0, Delta, b)})

infections.Prob = infections.Prob / infections.Prob.sum()

st.write("## New Infection Distribution")
st.write("For Maximum Pos =", x0, "Width =", Delta, "Asymmetry =", b)

st.bar_chart(infections.Prob[infections.Num < 10])

st.sidebar.write("### New Infection Distribution")
st.sidebar.bar_chart(infections.Prob[infections.Num < 10], width = 300, height = 200)

"""

There are some people who infect a large number of others, many only one or none

The distribution has the following shape:

"""

infections['xy'] = infections.Num * infections.Prob
infections['x2y'] = infections.Num**2 * infections.Prob
infections['x3y'] = infections.Num**3 * infections.Prob

xAvg = infections.xy.sum()
x2Avg = infections.x2y.sum()
x3Avg = infections.x3y.sum()

st.write("Mean Infections, R:", round(xAvg, 3), "\n")
st.write("Spread:", round(math.sqrt(x2Avg - xAvg**2), 3), "\n")
st.write("Asymmetry:", round((x3Avg - 3*xAvg*x2Avg + 2*xAvg**3)**(1/3), 3), "\n")


st.sidebar.write("Mean Infections, R:", round(xAvg, 3), "\n")


"""


# Neighborhood Simulations

We simulate a grid of independent neighborhoods for visualization purposes. 

Each infection period lasts one week and each infected person can infect a certain number of other people in the neighborhood in this week. This number of new infections (for a single infected person) is drawn from the log normal distribution. After the first week the person is no longer infectious. A week is therefore a single simulation period. 

The model does not include any exchange between locations, such as travel, that is once an infection has died out in a neighborhood, it will not come back.

We start with a single infection at each location.

"""

gridSize = 20
plotJitter = 0.2

# init
week = 1
positions = list(range(gridSize))


allLocations = pd.DataFrame({'x': positions * gridSize,
                             'y': np.repeat(positions, gridSize)})

# st.write('allLocations')
# st.write(allLocations)

infectionsByLocation = allLocations.copy()
infectionsByLocation['n'] = 1

# st.write('infectionsByLocation')
# st.write(infectionsByLocation)

# transform into individual lines, one positon for every infection
infectionLocations = pd.DataFrame({'x': flatten([ np.repeat(infectionsByLocation.x.iloc[i], n) for i,n in enumerate(infectionsByLocation.n)]),
                                   'y': flatten([ np.repeat(infectionsByLocation.y.iloc[i], n) for i,n in enumerate(infectionsByLocation.n)])})

# st.write('infectionLocations')
# st.write(infectionLocations)

totalInfections = infectionLocations.copy()
totalInfections['Week'] = week

# st.write('totalInfections')
# st.write(totalInfections)



def jitter(df, width = 1):
    df_copy = df.copy()
    df_copy.x = df_copy.x + (np.random.randn(df_copy.shape[0])) * width
    df_copy.y = df_copy.y + (np.random.randn(df_copy.shape[0])) * width
    return df_copy
    

p = figure(title=f'Infection Locations - Week {week}')
p.grid.visible = False
p.axis.visible = False
infectionLocationsJitter = jitter(infectionLocations, plotJitter)

p.circle(infectionLocationsJitter.x, infectionLocationsJitter.y,
         size=5, color="navy", alpha=0.5)
st.bokeh_chart(p, use_container_width=True)



"""


# Infection Evolution

For each infected person in each week we randomly pick a number of people they infect for the following week (while turning not infectious themselves)


"""


# increment
week = week + 1

# find new random numbers
numberInfections = infectionLocations.shape[0]
newInfections = randomIntLogNormal(numberInfections, x0, Delta, b)

# assign
infectionsByLocation = infectionLocations.copy()
infectionsByLocation['n'] = newInfections

# transform into individual lines, one positon for every infection
infectionLocations = pd.DataFrame({'x': flatten([ np.repeat(infectionsByLocation.x.iloc[i], n) for i,n in enumerate(infectionsByLocation.n)]),
                                   'y': flatten([ np.repeat(infectionsByLocation.y.iloc[i], n) for i,n in enumerate(infectionsByLocation.n)])})


# st.write('totalInfections b4')
# st.write(totalInfections)

infectionLocations['Week'] = week
totalInfections = pd.concat([totalInfections, infectionLocations]) 



#st.write('totalInfections after')
#st.write(totalInfections)

#st.write('totalInfections added?')
#st.write(infectionLocations)


p = figure(title=f'Infection Locations - Week {week}')
p.grid.visible = False
p.axis.visible = False
infectionLocationsJitter = jitter(infectionLocations, plotJitter)

p.circle(infectionLocationsJitter.x, infectionLocationsJitter.y,
         size=5, color="navy", alpha=0.3)
st.bokeh_chart(p, use_container_width=True)



"""


We now repeat the simulation for a number of weeks.

"""


# have already plotted 1 and 2
startWeek = 3
lastWeek = 20
displayWeek = 5

weekToShow = st.slider('Which Week to Display', startWeek, lastWeek, displayWeek, step = 1) 


for week in range(startWeek, lastWeek+1):
    
    # find new random numbers
    numberInfections = infectionLocations.shape[0]
    newInfections = randomIntLogNormal(numberInfections, x0, Delta, b)

    # assign
    infectionsByLocation = infectionLocations.copy()
    infectionsByLocation['n'] = newInfections

    # transform into individual lines, one positon for every infection
    infectionLocations = pd.DataFrame({'x': flatten([ np.repeat(infectionsByLocation.x.iloc[i], n) for i,n in enumerate(infectionsByLocation.n)]),
                                       'y': flatten([ np.repeat(infectionsByLocation.y.iloc[i], n) for i,n in enumerate(infectionsByLocation.n)])})

    infectionLocations['Week'] = week
    totalInfections = pd.concat([totalInfections, infectionLocations]) 
  
  
    if week == weekToShow:
        p = figure(title=f'Infection Locations - Week {week}')
        p.grid.visible = False
        p.axis.visible = False
        infectionLocationsJitter = jitter(infectionLocations, plotJitter)

        p.circle(infectionLocationsJitter.x, infectionLocationsJitter.y,
                 size=5, color="navy", alpha=0.3)
        st.bokeh_chart(p, use_container_width=True)



"""

We notice the evolution of infection clustering with a few hot spots with many infections and many locations with few or no infections.

This simulation is based on a completely random model, with all locations experience the exact same infection probability distribution.
However, the system is highly nonlinear due to the low numbers on one hand with the potential for a local disappearance of the infection,
and the existence of a chance for highly infectious people on the other hand.

Given this high heterogeneity of the (random) local distributions of infectiousness, the system does not average across locations but develops clusters.

"""



"""

# Evolution in Time by Location

We finally count the number of infections at each location at each week.
This demonstrates the potential for a (completely random) run-away number of cases at some locations while the majority of the infections dies out.



"""


#st.write('totalInfections')
#st.write(totalInfections)

#st.write('allLocations')
#st.write(allLocations)



# need to find the locations omitted

# one df with all locations and no cases
noInfections = pd.concat([allLocations]*lastWeek)

#st.write('noInfections')
#st.write(noInfections)

noInfections['Week'] = np.repeat(np.arange(1,lastWeek+1), gridSize**2)
noInfections['n'] = 0

#st.write(noInfections)

# add no cases to sim cases

totalInfections['n'] = 1
totalInfectionsSum = pd.concat([noInfections, totalInfections]).groupby(['x', 'y', 'Week']).sum()
totalInfectionsSumWide = totalInfectionsSum.unstack(level=(0,1))

totalInfectionsSum = totalInfectionsSum.reset_index()

#st.write('totalInfectionsSum')
#st.write(totalInfectionsSum)

#st.write('totalInfectionsSumWide')
#st.write(totalInfectionsSumWide)


# as separate lines
# df2 = pd.concat([df0, df2])
# then sum
# df2.groupby(['x', 'y', 'w']).sum()

# define label for location
# df['loc'] = df.x.astype(str) + '.' + df.y.astype(str)
totalInfectionsSum['loc'] = totalInfectionsSum.x.astype(str) + '|' + totalInfectionsSum.y.astype(str)

#st.write('totalInfectionsSum')
#st.write(totalInfectionsSum)


numlines=len(totalInfectionsSumWide.columns)
numColors = 256
numRepeats = numlines // 256 + 1    
mypalette = (plasma(numColors)*numRepeats)[0:numlines]


p = figure(title='Infections by Week at each Location')

# add a line renderer
#group = totalInfectionsSum.groupby('loc')
#source = ColumnDataSource(group)
#source = ColumnDataSource(totalInfectionsSumWide)

p.multi_line(xs = [totalInfectionsSumWide.index.values]*numlines,
             ys = [totalInfectionsSumWide[name].values for name in totalInfectionsSumWide],
             line_color = mypalette)

st.bokeh_chart(p, use_container_width=True)




st.write("Despite the divergence, the average infection rate, R, is the same across all locations:", round(xAvg, 3), "\n")




