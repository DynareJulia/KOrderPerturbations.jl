using Dynare
KOrderPerturbations = Dynare.KOrderPerturbations

context = @dynare "example1_2";

order = 2

Dynare.localapproximation!(order = 2, irf=0);

model = context.models[1]

ws = KOrderPerturbations.KOrderWs(model.endogenous_nbr,
                                  model.n_fwrd + model.n_both,
                                  model.n_states,
                                  model.n_current,
                                  model.exogenous_nbr,
                                  model.i_fwrd_b,
                                  model.i_bkwrd_b,
                                  model.i_current,
                                  1:model.n_states,
                                  order)

results = context.results.model_results[1]
GD = results.solution_derivatives
y0 = zeros(6)
n = 100
ut0 = sqrt(context.models[1].Sigma_e[1,1])
simulations = Dynare.KOrderPerturbations.simulate_run(GD, ut0, n, ws)
display(simulations)
ut1 = ut0
@show ut1
@show ut1*1.915 + ut1*ut1*1.4749 
@show ut1*1.915 + ut1*ut1*1.4749 + context.results.model_results[1].solution_derivatives[2][1,36]/2

       
