import evocomp

optimizer = evocomp.DifferentialEvolution(f=2, epochs=100)
optimizer.optimize(evocomp.Easom())

print(optimizer.best_candidate)

# Use with fitness fitness convergence halt criteria
optimizer = evocomp.DifferentialEvolution(f=2, halt_criteria=evocomp.FitnessConvergence(e=0.001))
optimizer.optimize(evocomp.Easom())

print(optimizer.best_candidate, f'epochs={optimizer.epochs}')

# Use with solution convergence
optimizer = evocomp.DifferentialEvolution(f=2, halt_criteria=evocomp.SolutionConvergence(e=0.001))
optimizer.optimize(evocomp.Easom())

print(optimizer.best_candidate, f'epochs={optimizer.epochs}')
