import numpy as np
import random

# Parameters
D = 7  # Number of days
H = 8  # Number of hours per day
maxHoursPerWeek = 40  # Maximum hours per nurse per week
maxHoursPerDay = 8  # Maximum hours per nurse per day
POP_SIZE = 20  # Population size
GENERATIONS = 1000  # Number of generations
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01

class Individual:
    def __init__(self, n):
        self.schedule = np.zeros((n, D, H), dtype=int)
        self.fitness = -np.inf

def calculate_fitness(schedule):
    n = schedule.shape[0]
    hours_per_nurse = np.sum(schedule, axis=(1, 2))
    daily_hours_per_nurse = np.sum(schedule, axis=2)

    mean = (D * H) / n
    variance = np.mean((hours_per_nurse - mean) ** 2)

    penalty = 0.0
    penalty += np.sum(np.maximum(hours_per_nurse - maxHoursPerWeek, 0) * 10)
    penalty += np.sum(np.maximum(daily_hours_per_nurse - maxHoursPerDay, 0) * 10)

    return -(variance + penalty)

def initialize_individual(ind):
    n = ind.schedule.shape[0]
    for i in range(n):
        weekly_hours = 0
        for j in range(D):
            daily_hours = 0
            for k in range(H):
                if daily_hours < maxHoursPerDay and weekly_hours < maxHoursPerWeek and random.random() < 0.5:
                    ind.schedule[i, j, k] = 1
                    daily_hours += 1
                    weekly_hours += 1
                else:
                    ind.schedule[i, j, k] = 0
    ind.fitness = calculate_fitness(ind.schedule)


def crossover(parent1, parent2):
    n = parent1.schedule.shape[0]
    crossover_point = random.randint(0, D * H - 1)
    offspring = Individual(n)
    for i in range(n):
        for j in range(D):
            for k in range(H):
                index = j * H + k
                if index < crossover_point:
                    offspring.schedule[i, j, k] = parent1.schedule[i, j, k]
                else:
                    offspring.schedule[i, j, k] = parent2.schedule[i, j, k]
    offspring.fitness = calculate_fitness(offspring.schedule)
    return offspring

def mutate(ind):
    n = ind.schedule.shape[0]
    for i in range(n):
        for j in range(D):
            for k in range(H):
                if random.random() < MUTATION_RATE:
                    ind.schedule[i, j, k] = 1 - ind.schedule[i, j, k]
    ind.fitness = calculate_fitness(ind.schedule)

def tournament_selection(population):
    tournament_size = 3
    best = random.choice(population)
    for _ in range(tournament_size - 1):
        challenger = random.choice(population)
        if challenger.fitness > best.fitness:
            best = challenger
    return best

def compare_schedules(schedule1, schedule2):
    return np.array_equal(schedule1, schedule2)

def print_schedule(schedule):
    n = schedule.shape[0]
    for i in range(n):
        print(f"Nurse {i + 1}:")
        for j in range(D):
            print(f"  Day {j + 1}: ", end="")
            for k in range(H):
                print(schedule[i, j, k], end="")
            print()
        print()

def main():
    random.seed()

    N = int(input("Enter the number of nurses: "))

    population = [Individual(N) for _ in range(POP_SIZE)]
    new_population = [Individual(N) for _ in range(POP_SIZE)]

    # Initialize population
    for ind in population:
        initialize_individual(ind)

    for gen in range(GENERATIONS):
        for i in range(POP_SIZE):
            if random.random() < CROSSOVER_RATE:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                new_population[i] = crossover(parent1, parent2)
            else:
                new_population[i] = tournament_selection(population)
            mutate(new_population[i])

        # Replace old population with new population
        population = new_population.copy()

        # Find best individual
        best = max(population, key=lambda ind: ind.fitness)
        ##print(f"Generation {gen}: Best fitness = {best.fitness}")

    # Find and print the best schedule
    best = max(population, key=lambda ind: ind.fitness)
    print("Best Schedule:")
    print_schedule(best.schedule)

    # Collect and count distinct best schedules
    best_schedules = []
    for ind in population:
        if ind.fitness == best.fitness:
            if not any(compare_schedules(ind.schedule, bs.schedule) for bs in best_schedules):
                best_schedules.append(ind)

    print(f"Number of distinct best schedules: {len(best_schedules)}")
    for i, ind in enumerate(best_schedules):
        print(f"Best Schedule {i + 1}:")
        print_schedule(ind.schedule)

if __name__ == "__main__":
    main()
