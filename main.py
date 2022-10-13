import numpy as np

import ga


def main():
    # entradas da equação
    equation_inputs = [4, -2, 3.5, 5, -11, -4.7]

    # número de pesos a otimizar
    num_weights = 6

    sol_per_pop = 8

    # população tem sol_per_pop cromossomos com num_weights gens
    population_size = (sol_per_pop, num_weights)

    # População inicial
    new_population = np.random.random_integers(low=0, high=1, size=population_size)

    # Algoritmo genético
    num_generations = 100
    num_parents_mating = 4

    for generation in range(num_generations):
        print(f"Geração: {generation}")
        print(f"POPULACAO: {new_population}")

        # medir o ‘fitness’ de cada cromossomo na população
        fitness = ga.cal_pop_fitness(new_population)

        print("Valores de fitness:")
        print(fitness)

        # Selecionar os melhores pais na população para o cruzamento
        parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)

        print("Genitores selecionados:")
        print(parents)

        # formar a próxima geração usando crossover
        offspring_crossover = ga.crossover(parents, offspring_size=(
            population_size[0] - parents.shape[0], num_weights
        ))

        print("Resultado do crossover:")
        print(offspring_crossover)

        # adicionar variações aos filhos usando mutação
        offspring_mutation = ga.mutation(offspring_crossover)
        print("Resultado da mutação:")
        print(offspring_mutation)

        # criar a nova população baseada nos pais e filhos
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        best_result = np.max(np.sum(new_population*equation_inputs, axis=1))
        print(f"Melhor resultado depois da geração {generation}: {best_result}")

    fitness = ga.cal_pop_fitness(new_population)
    best_match_idx = np.where(fitness == np.max(fitness))

    print("Melhor solução: ", new_population[best_match_idx, :])
    print("Fitness da melhor solução: ", fitness[best_match_idx])


if __name__ == '__main__':
    main()
