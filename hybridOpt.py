import pandas as pd
from cobra.io import read_sbml_model
from mewpy.io.sbml import load_ODEModel
from mewpy.problems import GOUProblem
from mewpy.optimization.evaluation import BPCY, WYIELD
from mewpy.optimization import EA
from hybrid import HybridSimulation
from mewpy.simulation import get_simulator


class HybridGOUProblem(GOUProblem):

    def __init__(self, model, hconstraints, fevaluation=None,**kwargs):
        """ Overrides GOUProblem by applying constraints resulting from
        sampling a kinetic model.
        """
        super().__init__(model, fevaluation, **kwargs)
        self.hconstraints = hconstraints

    def solution_to_constraints(self, candidate):
        constraints = super().solution_to_constraints(candidate)
        # apply the hybrid contraints:
        for r, v in self.hconstraints.items():
            if r in constraints.keys():
                x = constraints[r]
                if isinstance(x, tuple):
                    lb, ub = x
                else:
                    lb = x
                    ub = x
                hlb, hub = v
                l = max(lb, hlb)
                u = min(ub, hub)
                if l <= u:
                    constraints[r] = (l, u)
                else:
                    constraints[r] = (hlb, hub)
            else:
                constraints[r] = v
        return constraints    


def simulate():
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

    df=pd.read_csv("const_chassagnoles_nks10000_sigma1.cvs")
    lbs = df.quantile(0.1, axis=0).to_dict()
    ubs = df.quantile(0.9, axis=0).to_dict()

    kmodel = load_ODEModel("chassagnole2002.xml")
    cbmodel_iML1515 = read_sbml_model("iML1515.xml")

    BIOMASS = "BIOMASS_Ec_iML1515_core_75p37M"
    TARGET = 'EX_tyr__L_e'

    envcond = {'EX_glc__D_e': (-10.0, 0.0)}

    hsim = HybridSimulation(kmodel, cbmodel_iML1515, envcond=envcond, mapping=map_iML1515, timeout=20)
    hconstraints = hsim.mapping_bounds(lbs, ubs)

    sim = get_simulator(cbmodel_iML1515)
    res = sim.simulate(constraints=hconstraints)
    print(res)


def optimize():

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

    df = pd.read_csv("const_chassagnoles_nks10000_sigma1.cvs")
    lbs = df.quantile(0.1, axis=0).to_dict()
    ubs = df.quantile(0.9, axis=0).to_dict()

    kmodel = load_ODEModel("chassagnole2002.xml")
    cbmodel_iML1515 = read_sbml_model("iML1515.xml")

    BIOMASS = "BIOMASS_Ec_iML1515_core_75p37M"
    TARGET = 'EX_tyr__L_e'

    envcond = {'EX_glc__D_e': (-10.0, 0.0)}

    sim = HybridSimulation(kmodel, cbmodel_iML1515, envcond=envcond, mapping=map_iML1515, timeout=20)
    hconstraints = sim.mapping_bounds(lbs, ubs)

    objectives = [BPCY(BIOMASS, TARGET, method='lMOMA'),
                 WYIELD(BIOMASS, TARGET)]

    problem = HybridGOUProblem(cbmodel_iML1515, 
                               hconstraints, 
                               fevaluation=objectives, 
                               envcond=envcond)
    ea = EA(problem, max_generations=100)
    ea.run(simplify=False)

    df_res = ea.dataframe()
    df_res.to_csv("result.csv")
    print('concluded')


if __name__ == '__main__':
    simulate()
    optimize()
