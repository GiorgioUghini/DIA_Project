import numpy as np

def hs(x):
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])

def getClickCurve(phase, userType, x_values):
    if(phase == 0):
        # High interest / No competitors
        if(userType == 0):
            return 10900 * np.log(x_values/ 19 + 1)
        elif(userType == 1):
            return 14500 * np.log(x_values/ 20 + 1)
        else:
            return 11500 * (1 - np.exp((-1 * x_values) / 40)) + 2000 * np.log(x_values/ 35 + 1)
    elif(phase == 1):
        # Low interest / No competitors
        if(userType == 0):
            return 2450 * np.log(x_values / 3.5 + 1) - 15*(x_values - 17)*hs(x_values - 17)
        elif(userType == 1):
            return 3900 * np.log(x_values / 5 + 1) - 25*(x_values - 17)*hs(x_values - 17)
        else:
            return 2300 * np.log(x_values / 7.5 + 1) - 13*(x_values - 17)*hs(x_values - 17)
    elif (phase == 2):
        # Low interest / With competitors
        if(userType == 0):
            return -7000 * np.exp(-np.power(x_values- 0, 2.) / (2 * np.power(38, 2.))) + 7000
        elif(userType == 1):
            return -9500 * np.exp(-np.power(x_values- 0, 2.) / (2 * np.power(40, 2.))) + 9500
        else:
            return -4500 * np.exp(-np.power(x_values- 0, 2.) / (2 * np.power(35, 2.))) + 4500
    else:
        # High interest / With competitors
        if (userType == 0):
            return -23500 * np.exp(-np.power(x_values- 0, 2.) / (2 * np.power(55, 2.))) + 23500
        elif (userType == 1):
            return -37000 * np.exp(-np.power(x_values- 0, 2.) / (2 * np.power(65, 2.))) + 37000
        else:
            return -12500 * np.exp(-np.power(x_values- 0, 2.) / (2 * np.power(42, 2.))) + 12500


def getProbabilities(userType):
    if userType == 0:
        return 0.3
    elif userType == 1:
        return 0.5
    elif userType == 2:
        return 0.2
    else:
        return 1


def getDemandCurve(userType, t):
    if userType == 0:
        return 0.48*(0.75*np.exp(-np.power(t - 200, 2.) / (2*np.power(90, 2.))) + 0.3*np.exp(-np.power(t - 60, 2.) / (2*np.power(60, 2.))))
    elif userType == 1:
        return 0.52*(0.75*np.exp(-np.power(t - 250, 2.) / (2*np.power(90, 2.))) + 0.55*np.exp(-np.power(t - 65, 2.) / (2*np.power(90, 2.))))
    elif userType == 2:
        return 0.42*(0.42*np.exp(-np.power(t - 180, 2.) / (2*np.power(90, 2.))) + 0.35*np.exp(-np.power(t - 85, 2.) / (2*np.power(120, 2.))))
    else:
        return getProbabilities(0)*getDemandCurve(0, t) + getProbabilities(1)*getDemandCurve(1, t) + getProbabilities(2)*getDemandCurve(2, t)
