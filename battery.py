import numpy as np

class Battery(object):
    """ Used to store information about the battery.

       :param current_charge: is the initial state of charge of the battery
       :param capacity: is the battery capacity in Wh
       :param charging_power_limit: the limit of the power that can charge the battery in W
       :param discharging_power_limit: the limit of the power that can discharge the battery in W
       :param battery_charging_efficiency: The efficiecny of the battery when charging
       :param battery_discharing_efficiecny: The discharging efficiency
    """
    def __init__(self,
                 current_charge=0.0,
                 capacity=0.0,
                 charging_power_limit=1.0,
                 discharging_power_limit=-1.0,
                 charging_efficiency=0.95,
                 discharging_efficiency=0.95):
        self.current_charge = current_charge
        self.capacity = capacity
        self.charging_power_limit = charging_power_limit
        self.discharging_power_limit = discharging_power_limit
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency


class Battery_ob(object):
    
    
    def __init__(self,
                # These are the storage device properties
                cycles = 3000.0,
                maxLi = 30.0,
                installedCap = 10.0, # kWh
                maxDOD = 0.85, # % equivalent full cycle (EFC)
                EoL = 0.8, # end of life % equivalent full charge 
                minSOC = 0.0, # kWh
                maxChg = 5.0, # kW
                maxDisChg = -5.0, # kW
                etaChg = 0.9,
                etaDisChg = 0.9):
            
    # Make a list of the storage properties to pass to the functions
    
    
        self.cycles = cycles
        self.maxLi = maxLi
        self.installedCap = installedCap # kWh
        self.maxDOD = maxDOD # % equivalent full cycle (EFC)
        self.EoL = EoL # end of life % equivalent full charge 
        # self.maxSOC = np.mean([EoL*installedCap, maxDOD*installedCap])# kWh #post-defined
        self.maxSOC = installedCap# kWh #post-defined
        self.minSOC = minSOC # kWh
        self.maxChg = maxChg # kW
        self.maxDisChg = maxDisChg # kW
        self.etaChg = etaChg
        self.etaDisChg = etaDisChg
        
    

    
    
    
    
    
    
    