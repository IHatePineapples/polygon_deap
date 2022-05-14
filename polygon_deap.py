#!/usr/bin/env python3
"""

"""
import sys
import random
import statistics
import multiprocessing

from deap import creator, base, tools, algorithms
from PIL import Image, ImageDraw, ImageChops
from configparser import ConfigParser

STAT_FITNESSES=[]

conf = ConfigParser()
conf.read("config.ini")

NEW_POLYPB = conf.getfloat('main', 'new-polygon-probability')
TARGET = Image.open(conf.get('main', 'target'))
START_POLYGON = conf.getint('main', 'starting-polygons')
ITERATIONS=conf.getint('main','number-of-iterations')
nolimit = conf.getboolean('main', 'no-limit')
verbose = conf.getboolean('main', 'verbose')
svg = conf.getboolean('main', 'draw-svg')
FAST_END = conf.getboolean('override', 'faster-ending-override')
MAX_POLYGONS=conf.getint('main', 'max-polygons')

MAX = 255 * 200 * 200
TARGET.load()


def make_polygon():
      # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190

        R = random.randint(0,255)
        G = random.randint(0,255)
        B = random.randint(0,255)

        A = random.randint(30,60)


        x1 = random.randint(10,190)
        y1 = random.randint(10,190)

        x2 = random.randint(10,190)
        y2 = random.randint(10,190)

        x3 = random.randint(10,190)
        y3 = random.randint(10,190)

        return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3)]


def mutate(solution, indpb):
    r = random.random()
    f = 0
    if FAST_END:
        f = evaluate(solution)

    if r < 0.25:
        polygon = random.choice(solution)
        coords = [x for point in polygon[1:] for x in point]
        tools.mutGaussian(coords, 0, 20, indpb)
        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))
    elif 0.25 <= r < 0.50:
        #colors = [x for color in polygon[0] for x in color]
        #tools.mutGaussian(colors, 0, 10, indpb)
        #colors = [max(0, min(int(x), 255)) for x in colors]
        #polygon[0] = (colors[0],colors[1],colors[2],colors[3])
        # change color 
        polygon = random.choice(solution)
        colors = [x for color in polygon[0] for x in polygon[0]]
        tools.mutGaussian(colors, 0, 20, indpb)
        colors = [max(0, min(int(x),255)) for x in colors]
        polygon[0] = (colors[0],colors[1],colors[2],colors[3])
    elif (nolimit or 5 < len(solution) < MAX_POLYGONS ) and 0.50 <= r < 0.75 :
        # reorder polygons
        tools.mutShuffleIndexes(solution, indpb)
    elif random.random() < NEW_POLYPB: 
        new_polygon = make_polygon()
        solution.append(new_polygon)
    elif FAST_END and not nolimit and f[0] > 0.95:
        new_polygon = make_polygon()
        solution.append(new_polygon)
    return solution,

def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])

    image.save("out/solution.png")
    return image

def draw_svg(solution):
    with open("out/svg.html", "w") as out:
        out.write('''
<!DOCTYPE html>
<html>
    <body>
        <svg width="200" height="200">
        ''')
        for polygon in solution:
            colors = polygon[0]
            coords1 = polygon[1] 
            coords2 = polygon[2] 
            coords3 = polygon[3] 
            out.write(f'''
            <polygon points="{coords1[0]},{coords1[1]} {coords2[0]},{coords2[1]} {coords3[0]},{coords3[1]}" style="fill:rgb({colors[0]},{colors[1]},{colors[2]}); fill-opacity:{colors[3]/256}" />
            
            ''')

        
        
        out.write('''
        </svg>
    </body>
</html>
''')

def evaluate(solution):

    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX,

def main():
    CXPB=conf.getfloat('main', 'crossover-probability')
    MUTPB=conf.getfloat('main', 'mutation-probability')
    IT_OVERRIDE = conf.getboolean('override', 'iteration-count-override')
    POP_SIZE = conf.getint('main', 'population-size')
    TOUNR_SIZE = conf.getint('main', 'tournament-size')

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=TOUNR_SIZE)

    pool = multiprocessing.Pool(conf.getint('main', 'jobs'))
    toolbox.register("map", pool.map)

    toolbox.register("individual", tools.initRepeat, creator.Individual, make_polygon, n=START_POLYGON)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", mutate, indpb=0.3)
    toolbox.register("mate", tools.cxOnePoint)

    population = toolbox.population(n=POP_SIZE)


    #hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda x: x.fitness.values[0])
    #stats.register("avg", statistics.mean)
    stats.register("std", statistics.stdev)
    if verbose:
        print("index,fitness,avg-fitness,avg-polygons")
    else:
        print("index,fitness")

    #print("index,fitness,diff")
    # for i in range(ITERATIONS):
    i=0 
    p=[0,0,0]
    while (i <ITERATIONS or (not nolimit and IT_OVERRIDE and statistics.median(p) <MAX_POLYGONS)):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fitnesses = list(toolbox.map(toolbox.evaluate, offspring))

        for value, individual in zip(fitnesses, offspring):
            individual.fitness.values = value
        population = toolbox.select(offspring, len(population))


        f = [x[0] for x in fitnesses]
        #STAT_FITNESSES.append(f)
        #print("fit:", f[0]," i=",i)

        fmean = statistics.fmean(f)
        if verbose:
            p = [len(x) for x in offspring]
            print(f'{i},{f[0]},{fmean},{statistics.fmean(p)}')
        else:
            print(f'{i},{f[0]}')
        #print(f'{i},{f[0]},{f[0]-f0}')

        #f0 = f[0]
        if fmean > 0.94:
            CXPB = 0
            MUTPB = 0.5
        i+=1


    if svg:
        draw_svg(population[0]) 
    else:
        draw(population[0])


    
if __name__ == "__main__":
    main()
    exit(0)

