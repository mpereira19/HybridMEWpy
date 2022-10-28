from hybrid import HybridSimulation
from partialmoma import partialMOMA
from mewpy.simulation.kinetic import KineticSimulation
from mewpy.solvers import KineticConfigurations
from mewpy.solvers.solution import Status
import warnings


class HybridGeckoSimulation(HybridSimulation):

    def __init__(self, kmodel, 
                       cbmodel, 
                       gDW=564, 
                       envcond=..., 
                       mapping=...,
                       enzyme_mapping=None,
                       protein_prefix = 'draw_prot_',  
                       tSteps=..., 
                       timeout=KineticConfigurations.SOLVER_TIMEOUT):

        super().__init__(kmodel, cbmodel, gDW, envcond, mapping, tSteps, timeout)
        self.enzyme_mapping = enzyme_mapping
        self.protein_prefix = protein_prefix

    def simulate(self, objective=None, 
                       initcond=None, 
                       parameters=None, 
                       constraints=None,
                       min_frac = 0.1, 
                       method='pFBA'):
        """
        Runs a hybrid simulation on GECKO models by modifying the vmax values.
        Two type of constraints are applyed to the steady-state model:
            - metabolic constraints, that try to mimic the kinetic fluxes
            - enzymatic constraints, that limit enzyme usage in function of vmax and kcat values.

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
    
        1 sample, one at the time, the kinetic model, with different vmaxs (done outside this method) 
        2 take the obtained kinetic steady-state fluxes, convert them to the 
          GECKO model reaction flux units and define constraints on the GECKO 
          model reaction fluxes, 
        3 vmax = Kcat * [E], where [E] is the enzyme concentration. 
          The idea is to take the vmaxs (sampled in the kinetic model) 
          and compute the [E] using the GECKO Kcat. These values will 
          than be used to change the bounds of the respective enzyme 
          draw_protein pseudo reaction in the GECKO model.
        4 This last would need to be first a partial lMOMA, that is, a lMOMA that
          only minimizes the sum of differences on the reactions mapped from the kinetic
          to the GECKO model, as the CBM using FBA or pFBA will most probably be infeasible
          (the constraints will push the flux distributions outside the feasible space). 
          Using the partial lMOMA, you may find the closest solution in the steady-state space 
        5 run the phenotypic simulations
        """
        kmodel = self.get_kinetic_model()
        
        if parameters is None:
            from re import search
            p = list(kmodel.get_parameters(exclude_compartments=True))
            target = []
            for k in p:
                if search(r'(?i)[rv]max', k):
                    target.append(k)
            constants = kmodel.merge_constants()
            parameters = {k: constants[k] for k in target}

        # step 2
        ksim = KineticSimulation(model=kmodel, tSteps=self.tSteps, timeout=self.timeout)
        result = ksim.simulate(parameters=parameters, initcon=initcond)
        fluxes = result.fluxes
        metabolic_constraints = self.mapping_conversion(fluxes)

        # step 3
        enzymatic_constraints = dict()
        for vmax, protein in self.enzyme_mapping.items():
            vmax_value = parameters.get(vmax,None)
            if vmax_value:
                # The math:
                # ---------------------------------------------
                # v[j] <= Kcat[ij] E[i] = Vmax[ij]
                # v[j]/Kcat[ij] <=  E[i] = Vmax[ij]/Kcat[ij]

                # metabolic flux reaction constraints
                # ---------------------------------------------
                # v[j]/Kcat[ij] is covered by constraining the metabolic
                # fluxes in GECKO using the fluxes from the kinetic model

                # draw protein pool constraints
                # ---------------------------------------------
                #   0 <= e[i] <= E[i] 
                # Vmax[ij]/Kcat[ij] can be used to set an upper bound to

                # Note: a same protein may catalyze more than one
                # reaction (with different kcats) or be parte of
                # distinct enzyme complexes ... 
                # e = sum (vmax / kcat)
                # for now do not impose an upper bound

                reaction_kcats = self.sim.get_Kcat(protein)
                if len(reaction_kcats) == 1:  
                # for now assume kinetic model is in seconds 
                    max_enzyme_usage = vmax_value * 3600 / self.sim.get_Kcat(protein) 
                    enzymatic_constraints[f"{self.protein_prefix}{protein}"]=(0,max_enzyme_usage)  
                else:
                    warnings.warn(f'Protein {protein} is associated to more than 1 reaction')

        
        # step 4        
        # find the partial closest solution in the GEM space
        # if outside of the feasible space and update the metabolic constraints
        
        # Some relaxation on the biomass value might be need... make it iterable
        # and try to find the solution in the feasible space with the highest possible
        # fraction?
        if constraints is None:
            constraints = dict()
        constraints.update(enzymatic_constraints)
        feasible = False
        frac = 1
        p_sol = None
        while not feasible and frac >= min_frac: 
            p_sol = partialMOMA(self.sim,
                            objective=objective,
                            reactions=metabolic_constraints,
                            constraints=constraints,
                            # relax at the most the objective by 
                            obj_frac=0.8)
            if p_sol.status in [Status.OPTIMAL,Status.SUBOPTIMAL]:
                feasible = True 
            else:
                frac -= 0.1

        if not feasible:
            raise Exception('Unfeasible')

        for k in metabolic_constraints.keys():
            constraints[k] = p_sol.values[k]

        # step 5
        # run the phenotypic simulation

        if objective:
            solution = self.sim.simulate(objective= objective, method=method, constraints=constraints)
        else:
            solution = self.sim.simulate(method=method, constraints=constraints)

        return solution

