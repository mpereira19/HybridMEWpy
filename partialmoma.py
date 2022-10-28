from mewpy.simulation.simulation import Simulator
from mewpy.simulation import get_simulator
from mewpy.solvers import solver_instance
from math import inf

def partialMOMA(model, 
                biomass: str,
                reactions: dict,  
                constraints=None, 
                obj_frac=0):
		"""
		Run a (linear version of) Minimization Of Metabolic Adjustment (lMOMA) to find 
        a solution whose fluxes are the closest to those defined in the reactions dictionary.  

		:param model: a COBRAPY or REFRAMED model, or an instance of Simulator
		:param reactions: dictionary of reactions whose sum of fluxes is to be minimized
		:type reactions: dict
		:param biomass: name of the biomass reaction
		:type biomass: str
		:param constraints: constraints to be imposed, defaults to None
		:type constraints: dict, optional
        :param obj_frac: prefered fraction of the objective. Default 0 in which case only
        the exact objective is considered
		:type obj_frac: float
        :param min_obj_frac: minimun fraction of the objective. If zero, 
		:type obj_frac: float
		"""

		if isinstance(model, Simulator):
			simul = model
		else:
			simul = get_simulator(model)

		solver = solver_instance(simul)

		if not constraints:
			constraints = {}

		pre_solution = simul.simulate(objective={biomass: 1}, constraints=constraints)
		if obj_frac:
			reactions[biomass] = (obj_frac*pre_solution.objective_value,10000)
		else:
			reactions[biomass] = pre_solution.objective_value

		if not hasattr(solver, 'lMOMA_flag'):
			solver.lMOMA_flag = True
			for r_id in reactions.keys():
				d_pos, d_neg = r_id + '_d+', r_id + '_d-'
				solver.add_variable(d_pos, 0, inf, update=False)
				solver.add_variable(d_neg, 0, inf, update=False)
			solver.update()
			for r_id in reactions.keys():
				d_pos, d_neg = r_id + '_d+', r_id + '_d-'
				solver.add_constraint('c' + d_pos, {r_id: -1, d_pos: 1}, '>', -reactions[r_id], update=False)
				solver.add_constraint('c' + d_neg, {r_id: 1, d_neg: 1}, '>', reactions[r_id], update=False)
			solver.update()

		objective = dict()
		for r_id in reactions.keys():
			d_pos, d_neg = r_id + '_d+', r_id + '_d-'
			objective[d_pos] = 1
			objective[d_neg] = 1

		solution = solver.solve(objective, minimize=True, constraints=constraints)

		return solution
