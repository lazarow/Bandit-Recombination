#include <algorithm>
#include <chrono>
#include <climits>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "json.hpp"

using namespace std;
using namespace json;

// Constants
const int MUTATION_STRATEGY_NAIVE = 0;
const int MUTATION_STRATEGY_SWAP = 1;
const int MUTATION_STRATEGY_RANDOM_REASSIGNMENT = 2;

const int SELECTION_STRATEGY_ROULETTE = 0;
const int SELECTION_STRATEGY_TOURNAMENT = 1;
const int SELECTION_STRATEGY_RANK = 2;
const int SELECTION_STRATEGY_RANDOM = 3;

const int CROSSOVER_STRATEGY_ONE_POINT = 0;
const int CROSSOVER_STRATEGY_TWO_POINT = 1;
const int CROSSOVER_STRATEGY_UNIFORM = 2;

// Genetic algorithm parameters
unsigned int population_size = 100;
unsigned int elitism_size = 5;
unsigned int generations = 10000;
double mutation_rate = 0.15;
unsigned int seed = 0;
int mutation_strategy = MUTATION_STRATEGY_NAIVE;
int selection_strategy = SELECTION_STRATEGY_ROULETTE;
int crossover_strategy = CROSSOVER_STRATEGY_TWO_POINT;
unsigned int tournament_size = 8;

// Probabilistic Bandit Recombination (PBR) operator BEGIN
unsigned int pbr_samples_size = 0;
unsigned int pbr_offsprings_size = 0;
double pbr_discount_factor = 0.9;
double pbr_tau = 1.0;
// Probabilistic Bandit Recombination (PBR) operator END

// Problem parameters and types
unsigned int nof_tasks;
unsigned int nof_agents;
vector<int> capacity;
/**
 * task_demand[agent][task]
 */
vector<vector<int>> demands;
/**
 * cost[agent][task]
 */
vector<vector<int>> cost;
typedef vector<int> Individual;
typedef vector<Individual> Population;

int rand_int(const int min, const int max);
double rand_double();
void initialize_population(Population &population);
int evaluate_individual(const Individual &individual);
vector<int> evaluate_population(Population &population);
void mutation(Individual &individual);
Individual selection(const Population &population, const vector<int> &fitnesses);
pair<Individual, Individual> crossover(const Individual &parent1, const Individual &parent2);

// Probabilistic Bandit Recombination (PBR) operator BEGIN
unsigned int getMaximumFitness() {
    unsigned int maximum_fitness = 0;
    for (int i = 0; i < nof_agents; i++) {
        maximum_fitness += 1000 * capacity[i];
        for (int j = 0; j < nof_tasks; j++) {
            maximum_fitness += cost[i][j];
        }
    }
    return maximum_fitness;
}
vector<vector<double>> calculateUCBValues(const vector<vector<double>> &discountedSumOfRewards,
                                          const vector<vector<double>> &discountedCountOfPulls,
                                          double total_count_of_pulls) {
    vector<vector<double>> ucb_values(nof_agents, vector<double>(nof_tasks, 0));
    for (int i = 0; i < nof_agents; i++) {
        for (int j = 0; j < nof_tasks; j++) {
            ucb_values[i][j] = discountedSumOfRewards[i][j] / discountedCountOfPulls[i][j] +
                               sqrt(2 * log(total_count_of_pulls) / discountedCountOfPulls[i][j]);
        }
    }
    return ucb_values;
}
vector<vector<double>> calculateSoftmaxProbabilities(const vector<vector<double>> &ucb_values, double tau) {
    vector<vector<double>> probabilities(nof_agents, vector<double>(nof_tasks, 0));
    for (int i = 0; i < nof_tasks; i++) {
        double sum = 0.0;
        for (int j = 0; j < nof_agents; j++) {
            sum += ucb_values[j][i];
        }
        for (int j = 0; j < nof_agents; j++) {
            probabilities[j][i] = ucb_values[j][i] / sum;
        }
    }
    return probabilities;
}
Individual sampleIndividual(const std::vector<std::vector<double>> &probabilities) {
    Individual individual(nof_tasks);
    for (int i = 0; i < nof_tasks; i++) {
        double rand_value = rand_double();
        double cumulative_prob = .0;
        for (int j = 0; j < nof_agents; j++) {
            cumulative_prob += probabilities[j][i];
            if (rand_value <= cumulative_prob) {
                individual[i] = j;
                break;
            }
        }
        // If not chosen yet, return a random agent (for safety)
        individual[i] = rand_int(0, nof_agents - 1);
    }
    return individual;
}
// Probabilistic Bandit Recombination (PBR) operator END

void run() {
    // Initialize the population and find the best individual
    Population population(population_size);
    initialize_population(population);
    vector<int> fitnesses = evaluate_population(population);
    int best_fitness = *min_element(fitnesses.begin(), fitnesses.end());
    int best_individual_index = distance(fitnesses.begin(), min_element(fitnesses.begin(), fitnesses.end()));
    Individual best_individual = population[best_individual_index];

    bool improvement = true;

    // Probabilistic Bandit Recombination (PBR) operator BEGIN
    unsigned int maximum_fitness = getMaximumFitness();
    vector<vector<double>> discountedSumOfRewards(nof_agents, vector<double>(nof_tasks, 0));
    vector<vector<double>> discountedCountOfPulls(nof_agents, vector<double>(nof_tasks, 1));
    unsigned int total_count_of_pulls = 0;
    // Probabilistic Bandit Recombination (PBR) operator END

    // Evolution process
    for (int g = 0; g < generations; g++) {
        Population next_population(population_size);

        // Elitism: preserve the best individuals
        vector<int> indices(population_size);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&fitnesses](int a, int b) { return fitnesses[a] < fitnesses[b]; });
        for (int i = 0; i < elitism_size; i++) {
            next_population[i] = population[indices[i]];
        }

        // Probabilistic Bandit Recombination (PBR) operator BEGIN
        for (int i = 0; i < pbr_samples_size; i++) {
            for (int j = 0; j < nof_tasks; j++) {
                unsigned int agent = population[indices[i]][j];
                discountedSumOfRewards[agent][j] = pbr_discount_factor * discountedSumOfRewards[i][j] +
                                                   (1 - (double)fitnesses[indices[i]] / maximum_fitness);
                discountedCountOfPulls[agent][j] = pbr_discount_factor * discountedCountOfPulls[i][j] + 1;
            }
            total_count_of_pulls++;
        }
        if (pbr_offsprings_size > 0) {
            vector<vector<double>> ucb_values =
                calculateUCBValues(discountedSumOfRewards, discountedCountOfPulls, total_count_of_pulls);
            vector<vector<double>> probabilities = calculateSoftmaxProbabilities(ucb_values, pbr_tau);
            // Print probabilities
            cout << "Probabilities:" << endl;
            for (int i = 0; i < nof_agents; i++) {
                cout << "Agent " << i << ": ";
                for (int j = 0; j < nof_tasks; j++) {
                    cout << probabilities[i][j] << " ";
                }
                cout << endl;
            }
            for (int i = elitism_size; i < elitism_size + pbr_offsprings_size; i++) {
                next_population[i] = sampleIndividual(probabilities);
            }
        }
        // Probabilistic Bandit Recombination (PBR) operator END

        for (int i = elitism_size + pbr_offsprings_size; i < population_size; i += 2) {
            Individual parent1 = selection(population, fitnesses);
            Individual parent2 = selection(population, fitnesses);
            pair<Individual, Individual> children = crossover(parent1, parent2);
            mutation(children.first);
            mutation(children.second);
            next_population[i] = children.first;
            if (i + 1 < population_size) {
                next_population[i + 1] = children.second;
            }
        }

        // Find the best individual in the next population
        population = next_population;
        for (int i = 0; i < population_size; i++) {
            fitnesses[i] = evaluate_individual(population[i]);
            if (fitnesses[i] < best_fitness) {
                best_fitness = fitnesses[i];
                best_individual = population[i];
                improvement = true;
            }
        }

        // Debug
        if (improvement) {
            cout << "Improvement in generation " << (g + 1) << ", the new best fitness: " << best_fitness << endl;
            improvement = false;
        }
    }

    // Debug
    // Output the best individual
    cout << "Best solution found: ";
    for (int i = 0; i < nof_tasks; i++) {
        cout << best_individual[i] << " ";
    }
    cout << endl;
    cout << "Total cost: " << best_fitness << endl;
}

int main(int argc, char *argv[]) {
    // Parsing arguments
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " <input_file> [--population-size <size>] [--elitism-size "
                "<size>] [--generations <num>] [--mutation-rate <rate>] "
                "[--seed <seed>] [--mutation-strategy <strategy>] "
                "[--selection-strategy <strategy>] [--crossover-strategy "
                "<strategy>] [--tournament-size <size>] [--pbr-offsprings-size "
                "<size>] [--pbr-samples-size <size>] [--pbr-discount-factor "
                "<rate>] [--pbr-tau <rate>]"
             << endl;
        return 1;
    }

    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--population-size") {
            if (i + 1 < argc) population_size = stoi(argv[++i]);
        } else if (arg == "--elitism-size") {
            if (i + 1 < argc) elitism_size = stoi(argv[++i]);
        } else if (arg == "--generations") {
            if (i + 1 < argc) generations = stoi(argv[++i]);
        } else if (arg == "--mutation-rate") {
            if (i + 1 < argc) mutation_rate = stod(argv[++i]);
        } else if (arg == "--seed") {
            if (i + 1 < argc) seed = stoi(argv[++i]);
        } else if (arg == "--mutation-strategy") {
            if (i + 1 < argc) mutation_strategy = stoi(argv[++i]);
        } else if (arg == "--selection-strategy") {
            if (i + 1 < argc) selection_strategy = stoi(argv[++i]);
        } else if (arg == "--crossover-strategy") {
            if (i + 1 < argc) crossover_strategy = stoi(argv[++i]);
        } else if (arg == "--tournament-size") {
            if (i + 1 < argc) tournament_size = stoi(argv[++i]);
        } else if (arg == "--pbr-offsprings-size") {
            if (i + 1 < argc) pbr_offsprings_size = stoi(argv[++i]);
        } else if (arg == "--pbr-samples-size") {
            if (i + 1 < argc) pbr_samples_size = stoi(argv[++i]);
        } else if (arg == "--pbr-discount-factor") {
            if (i + 1 < argc) pbr_discount_factor = stod(argv[++i]);
        } else if (arg == "--pbr-tau") {
            if (i + 1 < argc) pbr_tau = stod(argv[++i]);
        }
    }

    cout << "Configuration:" << endl;
    cout << "Population size: " << population_size << endl;
    cout << "Elitism size: " << elitism_size << endl;
    cout << "Number of generations: " << generations << endl;
    cout << "Mutation rate: " << mutation_rate << endl;
    cout << "Random seed: " << seed << endl;
    cout << "Mutation strategy: " << mutation_strategy << endl;
    cout << "Selection strategy: " << selection_strategy << endl;
    cout << "Crossover strategy: " << crossover_strategy << endl;
    cout << "Tournament size: " << tournament_size << endl;
    cout << "PBR offsprings size: " << pbr_offsprings_size << endl;
    cout << "PBR samples size: " << pbr_samples_size << endl;
    cout << "PBR discount factor: " << pbr_discount_factor << endl;
    cout << "PBR tau: " << pbr_tau << endl;
    cout << endl;

    // Set random seed
    srand(seed == 0 ? time(nullptr) : seed);

    // Reading the input file (JSON format)
    ifstream file(argv[1]);
    if (!file.is_open()) {
        cerr << "Error while opening the input file." << endl;
        return 1;
    }
    stringstream buffer;
    buffer << file.rdbuf();
    string json_str = buffer.str();
    file.close();

    // Parsing the JSON string
    JSON input = JSON::Load(json_str);
    cout << "Running the " << input["name"].ToString() << " instance." << endl;
    nof_tasks = input["numcli"].ToInt();
    nof_agents = input["numserv"].ToInt();
    for (auto &cap : input["cap"].ArrayRange()) {
        capacity.push_back(cap.ToInt());
    }
    for (auto &agent_demands : input["req"].ArrayRange()) {
        demands.push_back(vector<int>());
        for (auto &task_demand : agent_demands.ArrayRange()) {
            demands[demands.size() - 1].push_back(task_demand.ToInt());
        }
    }
    for (auto &agent_costs : input["cost"].ArrayRange()) {
        cost.push_back(vector<int>());
        for (auto &task_cost : agent_costs.ArrayRange()) {
            cost[cost.size() - 1].push_back(task_cost.ToInt());
        }
    }
    run();
    return 0;
}

int rand_int(const int min, const int max) { return min + rand() % (max - min + 1); }

double rand_double() { return (double)rand() / RAND_MAX; }

void initialize_population(Population &population) {
    for (int i = 0; i < population_size; i++) {
        population[i] = vector<int>(nof_tasks);
        for (int j = 0; j < nof_tasks; j++) {
            population[i][j] = rand_int(0, nof_agents - 1);
        }
    }
}

int evaluate_individual(const Individual &individual) {
    int total_cost = 0;
    vector<int> agent_load(nof_agents, 0);

    for (int task = 0; task < nof_tasks; task++) {
        int agent = individual[task];
        total_cost += cost[agent][task];
        agent_load[agent] += demands[agent][task];
    }

    int penalty = 0;
    for (int agent = 0; agent < nof_agents; agent++) {
        if (agent_load[agent] > capacity[agent]) {
            penalty += 1000 * (agent_load[agent] - capacity[agent]);
        }
    }

    return total_cost + penalty;
}

vector<int> evaluate_population(Population &population) {
    vector<int> fitness(population_size);
    for (int i = 0; i < population_size; i++) {
        fitness[i] = evaluate_individual(population[i]);
    }
    return fitness;
}

void mutation(Individual &individual) {
    if (mutation_strategy == MUTATION_STRATEGY_NAIVE) {
        if (rand_double() < mutation_rate) {
            int task = rand_int(0, nof_tasks - 1);
            individual[task] = rand_int(0, nof_agents - 1);
        }
    } else if (mutation_strategy == MUTATION_STRATEGY_SWAP) {
        if (rand_double() < mutation_rate) {
            int task1 = rand_int(0, nof_tasks - 1);
            int task2 = rand_int(0, nof_tasks - 1);
            swap(individual[task1], individual[task2]);
        }
    } else if (mutation_strategy == MUTATION_STRATEGY_RANDOM_REASSIGNMENT) {
        if (rand_double() < mutation_rate) {
            int task = rand_int(0, nof_tasks - 1);
            int current_agent = individual[task];
            vector<int> possible_agents;
            for (int agent = 0; agent < nof_agents; agent++) {
                if (agent == current_agent) continue;
                bool feasible = true;
                int used_capacity = 0;
                for (int t = 0; t < nof_tasks; t++) {
                    if (t == task) {
                        if (individual[t] == current_agent) continue;
                        used_capacity += demands[agent][t];
                    } else if (individual[t] == agent) {
                        used_capacity += demands[agent][t];
                    }
                }
                if (used_capacity <= capacity[agent]) {
                    possible_agents.push_back(agent);
                }
            }
            if (possible_agents.empty() == false) {
                int new_agent = possible_agents[rand_int(0, possible_agents.size() - 1)];
                individual[task] = new_agent;
            }
        }
    }
}

Individual selection(const Population &population, const vector<int> &fitnesses) {
    if (selection_strategy == SELECTION_STRATEGY_ROULETTE) {
        int total_fitness = 0;
        for (int i = 0; i < population_size; i++) {
            total_fitness += fitnesses[i];
        }
        int random_point = rand_int(0, total_fitness);
        int current_sum = 0;
        for (int i = 0; i < population_size; i++) {
            current_sum += fitnesses[i];
            if (current_sum > random_point) {
                return population[i];
            }
        }
        return population[population_size - 1];
    } else if (selection_strategy == SELECTION_STRATEGY_TOURNAMENT) {
        int best_fitness = INT_MAX;
        int best_index = -1;
        for (int i = 0; i < tournament_size; i++) {
            int random_index = rand_int(0, population_size - 1);
            if (fitnesses[random_index] < best_fitness) {
                best_fitness = fitnesses[random_index];
                best_index = random_index;
            }
        }
        return population[best_index];
    } else if (selection_strategy == SELECTION_STRATEGY_RANK) {
        vector<int> indices(population_size);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&fitnesses](int a, int b) { return fitnesses[a] < fitnesses[b]; });
        int total_rank = (population_size * (population_size + 1)) / 2;
        int random_point = rand_int(1, total_rank);
        int current_sum = 0;
        for (int i = 0; i < population_size; i++) {
            current_sum += (population_size - i);
            if (current_sum >= random_point) {
                return population[indices[i]];
            }
        }
        return population[indices[population_size - 1]];
    }
    // Random selection
    return population[rand_int(0, population_size - 1)];
}

pair<Individual, Individual> crossover(const Individual &parent1, const Individual &parent2) {
    Individual child1(nof_tasks);
    Individual child2(nof_tasks);
    if (crossover_strategy == CROSSOVER_STRATEGY_ONE_POINT) {
        int crossover_point = rand_int(0, nof_tasks - 1);
        for (int i = 0; i < crossover_point; i++) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
        for (int i = crossover_point; i < nof_tasks; i++) {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    } else if (crossover_strategy == CROSSOVER_STRATEGY_TWO_POINT) {
        int crossover_point1 = rand_int(0, nof_tasks - 1);
        int crossover_point2 = rand_int(0, nof_tasks - 1);
        if (crossover_point1 > crossover_point2) {
            swap(crossover_point1, crossover_point2);
        }
        for (int i = 0; i < crossover_point1; i++) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
        for (int i = crossover_point1; i < crossover_point2; i++) {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
        for (int i = crossover_point2; i < nof_tasks; i++) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
    } else if (crossover_strategy == CROSSOVER_STRATEGY_UNIFORM) {
        for (int i = 0; i < nof_tasks; i++) {
            if (rand_double() < 0.5) {
                child1[i] = parent1[i];
                child2[i] = parent2[i];
            } else {
                child1[i] = parent2[i];
                child2[i] = parent1[i];
            }
        }
    }
    return make_pair(child1, child2);
}
