import warnings
from mewpy.model.kinetic import ODEModel
from mewpy.solvers import KineticConfigurations
from mewpy.simulation import get_simulator
from mewpy.simulation.simulation import Simulator
from mewpy.simulation.kinetic import KineticSimulation
from collections import OrderedDict
from warnings import warn
import pandas as pd
import numpy as np
from numpy.random import normal
from re import search
from tqdm import tqdm


warnings.filterwarnings('ignore', 'Timeout')


def sample(vmaxs, sigma=0.1):
    k = vmaxs.keys()
    f = np.exp(normal(0, sigma, len(vmaxs)))
    v = np.array(list(vmaxs.values()))
    r = list(v*f)
    return dict(zip(k, r))


class HybridSimulation:

    def __init__(self, kmodel, cbmodel, gDW=564.0, envcond=dict(),
                 mapping=dict(), parameters=None, tSteps=[0, 1e9],
                 timeout=KineticConfigurations.SOLVER_TIMEOUT):

        if not isinstance(kmodel, ODEModel):
            raise ValueError('model is not an instance of ODEModel.')
        
        if not isinstance(cbmodel, Simulator):
            self.sim = get_simulator(cbmodel, envcond=envcond)
        else:
            self.sim = cbmodel

        self.kmodel = kmodel
        self.mapping = mapping
        self.parameters = parameters
        self.tSteps = tSteps
        self.timeout = timeout
        self.models_verification()
        self.gDW = gDW

    def __getstate__(self):
        state = OrderedDict(self.__dict__.copy())
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_parameters(self):
        return self.parameters

    def get_kinetic_model(self):
        return self.kmodel

    def get_mapping(self):
        return self.mapping

    def models_verification(self):
        """
        Function that verifies if it's possible to perform the Hibrid Simulation.
        """
        kmodel = self.get_kinetic_model()
        cbmodel = self.sim.model
        mapping = self.get_mapping()

        lst = list()
        for k in kmodel.ratelaws.keys():
            if k in cbmodel.reactions:
                lst.append(k)

        if len(lst) != 0:
            return True
        elif len(mapping) != 0:
            return False
        else:
            raise warn('Fluxes names not compatible, please give a mapping dictionary')

    def mapping_conversion(self, fluxes):
        """
        Function that converts the kinetic fluxes into constraint-based fluxes.

        :param fluxes: kinetic simulation fluxes
        :type fluxes: dict
        :return: kinetic fluxes compatible with the constraint-based model
        """
        mapping = self.get_mapping()
        flxs = dict()
        for k, value in fluxes.items():
            if k in mapping.keys():
                v = mapping[k]
                flxs[v[0]] = v[1]*value * 3600/self.gDW
        if len(flxs) != 0:
            return flxs
        else:
            raise warn('Mapping not done properly, please redo mapping')

    def mapping_bounds(self, lbs, ubs):
        """
        Function that converts the kinetic bounds into constraint-based flux bounds.

        :param lbs: kinetic lower bounds
        :type fluxes: dict
        :param ubs: kinetic upper bounds
        :type fluxes: dict
        :return: constraints
        """
        mapping = self.get_mapping()
        flxs = dict()
        for k, value in lbs.items():
            if k in mapping.keys():
                v = mapping[k]
                a = v[1]*value * 3600/self.gDW
                b = v[1]*ubs[k] * 3600/self.gDW
                flxs[v[0]] = (a, b) if a < b else (b, a)
        if len(flxs) != 0:
            return flxs
        else:
            raise warn('Mapping not done properly, please redo mapping')

    def nsamples(self, vmaxs, n=1, sigma=0.1):
        """
        Generates n fluxes samples varying vmax values on a log-norm distribtution
        with mean 0 and std sigma.

        """
        kmodel = self.get_kinetic_model()
        ksample = []
        ksim = KineticSimulation(model=kmodel, tSteps=self.tSteps, timeout=self.timeout)
        for _ in tqdm(range(n)):
            v = sample(vmaxs, sigma=sigma)
            try:
                res = ksim.simulate(parameters=v)
                if res.fluxes:
                    ksample.append(res.fluxes)
            except Exception as e:
                warn.warning(str(e))
        df = pd.DataFrame(ksample)
        # drop any NaN if exist
        df.dropna()
        return df

    def simulate(self, biomass=None, sample=None, parameters=None, constraints=None, method='pFBA'):
        """
        This method performs a phenotype simulation hibridizing a kinetic and a constraint-based model.

        :param biomass: Name of the biomass reaction.
        :type biomass: str, optional
        :param sample:List of numbers in which the kinetic simulation fluxes will be scaled.
        :param sample: list, optional
        :param parameters: Kinetic simulation parameters.
        :type parameters: dict, optional
        :param constraints: Constraint-based model simulation constraints.
        :type constraints: dict, optional
        :param method: the phenotype simulation method
        :type method: str. Default 'pFBA'
        :returns: Returns the solution of the hibridization.
        """
        mapp = self.models_verification()
        kmodel = self.get_kinetic_model()

        ksim = KineticSimulation(model=kmodel, tSteps=self.tSteps, timeout=self.timeout)
        result = ksim.simulate(parameters=parameters, initcon=sample)
        fluxes = result.fluxes
        
        if constraints is None:
            constraints = dict()

        if not mapp:
            fluxes = self.mapping_conversion(fluxes)
        constraints.update(fluxes)
        # pfba
        if biomass:
            self.sim.objective = biomass
        solution = self.sim.simulate(method=method, constraints=constraints)
        return solution


    def simulate_distribution(self, df, q1=0.1, q2=0.9, objective=None, method='pFBA', constraints=None):
        """Runs a pFBA on the steady-state model with fluxes constrained to ranges between the q1-th and q2-th percentile
        of fluxes distributions sampled from the kinetic model.
        The kinetic flux distributions are provided as panda dataframes.     
        """
        const = dict()
        lbs = df.quantile(q1, axis=0).to_dict()
        ubs = df.quantile(q2, axis=0).to_dict()
        if constraints:
            const.update(constraints)
        k_const = self.mapping_bounds(lbs, ubs)
        const.update(k_const)
        if objective:
            solution = self.sim.simulate(method=method, constraints=const, objective=objective)
        else:
            solution = self.sim.simulate(method=method, constraints=const)
        solution.kinetic_constraints = k_const
        return solution


if __name__ == '__main__':

    import os
    from cobra.io import read_sbml_model
    from mewpy.io.sbml import load_ODEModel
    from mewpy.util.constants import EAConstants


    
    # Maps kinetic reactions into steady-state reactions
    # Kinetic_reaction_id => (steady-state_reaction_id, sense)
    # The sense identifies if the reaction sense of reactions are the same in the kin
    # and in the steady state model (1: same sense; -1 reverse sense)
    map_iJR904 = {'vPGI': ('PGI', 1),
                  'vPFK':  ('PFK', 1),
                  'vALDO': ('FBA', 1),
                  'vGAPDH': ('GAPD', 1),
                  'vTIS': ('TPI', 1),
                  'vPGK': ('PGK', -1),
                  'vrpGluMu': ('PGM', -1),
                  'vENO': ('ENO', 1),
                  'vPK': ('PYK', 1),
                  'vPDH': ('PDH', 1),
                  'vG6PDH': ('G6PDH2r', 1),
                  'vPGDH': ('GND', 1),
                  'vR5PI': ('RPI', -1),
                  'vRu5P': ('RPE', 1),
                  'vTKA': ('TKT1', 1),
                  'vTKB': ('TKT2', 1),
                  'vTA': ('TALA', 1),
                  'vpepCxylase': ('PPC', 1),
                  'vPGM': ('PGMT', -1),
                  'vPTS': ('GLCpts', 1)
                  }

    map_iML1515 = {'vPGI': ('PGI', 1),
                  'vPFK': ('PFK', 1),
                  'vALDO': ('FBA', 1),
                  'vGAPDH': ('GAPD', 1),
                  'vTIS': ('TPI', 1),
                  'vPGK': ('PGK', -1),
                  'vrpGluMu': ('PGM', -1),
                  'vENO': ('ENO', 1),
                  'vPK': ('PYK', 1),
                  'vPDH': ('PDH', 1),
                  'vG6PDH': ('G6PDH2r', 1),
                  'vPGDH': ('GND', 1),
                  'vR5PI': ('RPI', -1),
                  'vRu5P': ('RPE', 1),
                  'vTKA': ('TKT1', 1),
                  'vTKB': ('TKT2', 1),
                  'vTA': ('TALA', 1),
                  'vpepCxylase': ('PPC', 1),
                  'vPGM': ('PGMT', -1),
                  'vPTS': ('GLCptspp', 1)
                  }

    kmodel = load_ODEModel("chassagnole2002.xml")
    envcond = {'EX_glc__D_e': (-10.0, 0.0)}
    cbmodel= read_sbml_model("iML1515.xml")


    simul = HybridSimulation(kmodel=kmodel, cbmodel=cbmodel, envcond=envcond, mapping=map_iML1515)

    # identify the vmax parameters 
    p = list(kmodel.get_parameters(exclude_compartments=True))
    target = []
    for k in p:
        if search(r'(?i)[rv]max', k):
            target.append(k)
    constants = kmodel.merge_constants()
    vmaxs = {k: constants[k] for k in target}

    # kinetic_df = simul.nsamples(vmaxs)
    kinetic_df = pd.read_csv('const_chassagnoles_nks10000_sigma1.cvs')

    solution = simul.simulate_distribution(kinetic_df)
    print(solution)
