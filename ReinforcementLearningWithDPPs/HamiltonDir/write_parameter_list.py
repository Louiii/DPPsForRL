import json

beta_init = [1.1, 1.5, 2, 5, 7, 10, 15, 20]
beta_frac = [20000, 30000, 40000, 50000, 60000]
eta_start = [20000, 30000, 40000, 50000, 60000]
eta_const = [0.0002, 0.0005, 0.001, 0.003, 0.01]

parameters = []
for bi in beta_init:
	for bf in beta_frac:
		for es in eta_start:
			for ec in eta_const:
				parameters.append([bi, bf, es, ec])

with open('parameter_list.json', 'w') as outfile:
    outfile.write(json.dumps(parameters))