#!/usr/bin/env python3
"""
Genetic algorithm implemented with DEAP solving the one max problem
(maximising number of 1s in a binary number).

"""
import random
import statistics
import multiprocessing

from deap import creator, base, tools, algorithms
from PIL import Image, ImageDraw, ImageChops


MAX = 255 * 200 * 200
TARGET = Image.open("in/target.png")
TARGET.load()
ITERATIONS=20000
STAT_FITNESSES=[]


def make_polygon():
      # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190

        R = random.randint(0,256)
        G = random.randint(0,256)
        B = random.randint(0,256)

        A = random.randint(30,60)


        x1 = random.randint(10,190)
        y1 = random.randint(10,190)

        x2 = random.randint(10,190)
        y2 = random.randint(10,190)

        x3 = random.randint(10,190)
        y3 = random.randint(10,190)

        return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3)]


def mutate(solution, indpb):
        if random.random() < 0.5:
            # mutate points
            polygon = random.choice(solution)
            coords = [x for point in polygon[1:] for x in point]
            tools.mutGaussian(coords, 0, 20, indpb)
            coords = [max(0, min(int(x), 200)) for x in coords]
            polygon[1:] = list(zip(coords[::2], coords[1::2]))

            #colors = [x for color in polygon[0] for x in color]
            #tools.mutGaussian(colors, 0, 10, indpb)
            #colors = [max(0, min(int(x), 255)) for x in colors]
            #polygon[0] = (colors[0],colors[1],colors[2],colors[3])
            # change color
            colors = [x for color in polygon[0] for x in polygon[0]]
            tools.mutGaussian(colors, 0, 20, indpb)
            colors = [max(0, min(int(x),255)) for x in colors]
            polygon[0] = (colors[0],colors[1],colors[2],colors[3])
        else:
            # reorder polygons
            tools.mutShuffleIndexes(solution, indpb)

        return solution,

def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])

    image.save("out/solution.png")
    return image

def evaluate(solution):

    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX,

def main():

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=20)

    pool = multiprocessing.Pool(4)
    toolbox.register("map", pool.map)

    toolbox.register("individual", tools.initRepeat, creator.Individual, make_polygon, n=100)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", mutate, indpb=0.5)
    toolbox.register("mate", tools.cxTwoPoint)

    population = toolbox.population(n=20)


    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda x: x.fitness.values[0])
    #stats.register("avg", statistics.mean)
    stats.register("std", statistics.stdev)
    CXPB=0.5
    MUTPB=0.3
    print("index,fitness,diff")
    f0 = 0
    for i in range(ITERATIONS):

        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fitnesses = list(toolbox.map(toolbox.evaluate, offspring))

        for value, individual in zip(fitnesses, offspring):
            individual.fitness.values = value
        population = toolbox.select(offspring, len(population))


        f = [x[0] for x in fitnesses]
        #STAT_FITNESSES.append(f)
        #print("fit:", f[0]," i=",i)

        print(f'{i},{f[0]},{f[0]-f0}')

        f0 = f[0]

        # print("avg:", statistics.mean(f))

    #population, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1,
    #    ngen=50, stats=stats, halloffame=hof, verbose=False)


    draw(population[0])
    #print(hof)
def report():
    csv = open("out/graph.csv","x")
    csv.write("i,fitness")
    a = 1
    for i[0] in STAT_FITNESSES:
        csv.write(f'{a},{i}\n')
        a=+1

    
if __name__ == "__main__":

    main()
    #print("\ntarget done!\n")
    #report()
    #print("\nreport done!")
    exit(0)

