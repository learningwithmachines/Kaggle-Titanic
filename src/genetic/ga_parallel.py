'''
* Implementation of symbolic classifier using genetic algorithms based off
  of the DEAP Library.
    ** Available deap algorithms are eaSimple and eaMuPlusLambda.
    ** Parallelized through Pathos-ProcessPool(with Dill) and Numba.
    ** Uses recommended StaticLimits on heights(=17) and DoubleTournaments
       for bloat control.
* CLI-based with argparse, saves best result with fitness and genealogy plots to file.
    ** Can also optionally resume training from the terminal-populations of 
       a previously saved *_state* file via the *resume* and *seedfile* parameter.
'''
import sys
import gc
import pickle
import argparse
import operator
from types import LambdaType, FunctionType
from typing import Tuple, Callable
import dill
from itertools import repeat
import numpy as np
from numba import jit, njit, float32, float64, boolean
from deap import base, creator, tools, gp, algorithms
from pathos import multiprocessing as mp
import matplotlib.pyplot as plt
import networkx
from networkx.drawing.nx_agraph import graphviz_layout
sys.path.append('../')
from data_utils.gadata import GAData

# globals
gcreator: creator = creator
gtoolbox: base.Toolbox = base.Toolbox()
# datafiles
featuresfile: str = './../../saves/features_data_dict.bin'
augdatafile: str = './../../saves/augmented_data_dict.bin'


def jmap(func: FunctionType, array: np.ndarray) -> np.ndarray:
    '''
    fast numpy map
    :param func: numba.targets.registry.CPUDispatcher, numba njit compiled chained gp-expression of primitives
    :param array: ndarray, array to apply func on
    :return: ndarray, func mapped array
    '''
    return np.array(tuple(func(*tuple(float(x) for x in subarray)) for subarray in array), dtype=np.float32)


# best logger
ghof: tools.support.HallOfFame = tools.HallOfFame(1)
# fitness logger
g_fit: tools.support.Statistics = tools.Statistics(lambda ind: ind.fitness.values)
g_len: tools.support.Statistics = tools.Statistics(len)
g_height: tools.support.Statistics = tools.Statistics(operator.attrgetter('height'))
gstats: tools.support.MultiStatistics = tools.MultiStatistics(fitness=g_fit,
                                                              length=g_len,
                                                              height=g_height)

rmean = lambda x: np.nanmean(x).round(2)
rstd = lambda x: np.nanstd(x).round(2)
rmin = lambda x: np.min(np.where(np.isnan(x), np.inf, x)).round(2)
rmax = lambda x: np.max(np.where(np.isnan(x), np.inf, x)).round(2)

gstats.register("avg", rmean)
gstats.register("std", rstd)
gstats.register("min", rmin)
gstats.register("max", rmax)


def parse_args() -> Tuple[dict, object, int, str, argparse.Namespace]:
    '''
    argument parser for GA
    :return: tuple[dict, object, int, str, Namespace] needed data for GA construction
    '''
    parser: argparse.ArgumentParser = argparse.ArgumentParser('GA Parallel')

    def type_max_tl(num: int) -> int:
        '''
        type checker for max tree length
        :param num: int, arg-parse input
        :return: int, type casted and checked output
        '''
        x: int = int(num)
        if x > 90 or x < 1:
            raise parser.error(f"max_tl must be between 1 and 17, passed value: {x}")
        return x


    def get_easimple_args(popsize: int, cxpb: float,
                          mutpb: float, ngen: int, stats: tools.support.MultiStatistics,
                          hof: tools.support.HallOfFame, verbose: bool,
                          toolbox: base.Toolbox = gtoolbox) -> dict:
        '''
        :param popsize: int, initial population size
        :param cxpb: float, crossover probability
        :param mutpb: float, mutation probability
        :param ngen: int, number of generation for evolution
        :param stats: tools.support.MultiStatistics, storage object for gp.toolbox
        :param hof: tools.support.HallOfFame, results storage for gp.toolbox, stores best performing functions
        :param verbose: bool, flag
        :param toolbox: gp.toolbox class factory object
        :return: dict, for passing to gp algorithm
        '''
        keys = ['population', 'toolbox', 'cxpb',
                'mutpb', 'ngen', 'stats',
                'halloffame', 'verbose']
        values = [popsize, toolbox, cxpb,
                  mutpb, ngen, stats, hof, verbose]
        return dict(zip(keys, values))


    def get_eamupluslambda_args(popsize: int, mu: int, lambda_: int,
                                cxpb: float, mutpb: float, ngen: int,
                                stats: tools.support.MultiStatistics,
                                hof: tools.support.HallOfFame, verbose: bool,
                                toolbox: base.Toolbox = gtoolbox) -> dict:
        '''
        :param popsize: int, initial population size
        :param mu: int, size of population after selection
        :param lambda_: int, number of childrens for mutation
        :param cxpb: float, crossover probability
        :param mutpb: float, mutation probability
        :param ngen: int, number of generation for evolution
        :param stats: tools.support.MultiStatistics, storage object for gp.toolbox
        :param hof: tools.support.HallOfFame, results storage for gp.toolbox, stores best performing functions
        :param verbose: bool, flag
        :param toolbox: gp.toolbox class factory object
        :return: dict, for passing to gp algorithm
        '''

        keys = ['population', 'toolbox', 'mu',
                'lambda_', 'cxpb', 'mutpb',
                'ngen', 'stats', 'halloffame', 'verbose']
        values = [popsize, toolbox, mu,
                  lambda_, cxpb, mutpb,
                  ngen, stats, hof, verbose]
        return dict(zip(keys, values))


    parser.add_argument('PopSize', help='Starting Population Size, discarded if resuming.\
                        \n Default is 50.',
                        type=int,
                        default=50)

    parser.add_argument('Numgens', help='Number of Generations.\
                        \n Default is 10.',
                        type=int,
                        default=10)

    parser.add_argument('--cxpb',
                        help='The probability that an offspring is produced by crossover.\
                        \n Default is 0.25.',
                        type=float,
                        default=0.25)

    parser.add_argument('--mutpb',
                        help='The probability that an offspring is produced by mutation.\
                        \n Default is 0.5.',
                        type=float,
                        default=0.5)

    parser.add_argument('--mu',
                        help='The number of individuals to select for the next generation,\
                             does not apply to eaSimple.\
                             \n Default is equal to the Population Size Argument.',
                        type=int,
                        default=False)

    parser.add_argument('--lambda_',
                        help='The number of children to produce at each generation,\
                             does not apply to eaSimple.\
                             \n Default is twice the Population Size Argument.',
                        type=int,
                        default=False)

    parser.add_argument('--max_tl',
                        help='The maximum height of the tree to construct for gp.PrimitiveTree.\
                        \n Default is 15, Maximum is 91.',
                        type=type_max_tl,
                        default=15)

    parser.add_argument('--algorithm',
                        type=str,
                        help='Choice of Evolutionary Algorithm, options are "simple" for eaSimple and\
                        \n"mupluslambda" for eaMuPlusLambda.\n Default is "simple".',
                        default='simple',
                        choices=['simple', 'mupluslambda'])

    parser.add_argument('--loss',
                        type=str,
                        help='Choice of Objective Function, options are "acc" for accuracy'\
                             '\n and "ce" for Binary Cross Entropy Loss.\n Default is "ce".',
                        default='ce',
                        choices=['acc', 'ce'])

    parser.add_argument('--verbose',
                        help='output verbosity.\
                        \n Default is True.',
                        action='store_true')

    parser.add_argument('--test',
                        help='reload and test the generated function on a random input.\
                        \n Default is True.',
                        default=True,
                        action='store_true')

    parser.add_argument('-i', '--data',
                        action='store',
                        help=f'Input Data Dictionary compatible with GAData()\
                        \n Default is {featuresfile}.',
                        type=str,
                        dest='data',
                        default=featuresfile)

    parser.add_argument('--normalize',
                        help='Normalize the training data.\
                        \n Default is True.',
                        type=bool, default=True)

    parser.add_argument('--resume',
                        action='store_true',
                        help='Resume training from a previous run,\
                        Will overwrite the starting population from\
                        saved populations of an existing savefile,\
                        provided by the seedfile argument.\
                        \n Default is False.',
                        default = False)

    parser.add_argument('-sf', '--seedfile',
                        action='store',
                        help='Resume training from a previous run,\
                        Will overwrite the starting population from\
                        saved populations of an existing savefile,\
                        provided by the seedfile argument.\
                        \n Default is False.',
                        type=argparse.FileType('rb'),
                        dest='seedfile', default=None)

    parser.add_argument('-o', '--ofile',
                        action='store',
                        help='Wrap the final function to a dictionary as\
                        {"func":func:object} and save to a file in path.\
                        \n Default is ./../../saves/evolved_genetic_function.npy.',
                        type=argparse.FileType('wb'), dest='ofile',
                        default='./../../saves/evolved_genetic_function.npy')

    ret_args = parser.parse_args()
    if ret_args.algorithm == 'simple':
        taskdict: dict = get_easimple_args(popsize=ret_args.PopSize,
                                           cxpb=ret_args.cxpb,
                                           mutpb=ret_args.mutpb,
                                           ngen=ret_args.Numgens,
                                           stats=gstats,
                                           hof=ghof,
                                           verbose=ret_args.verbose,
                                           toolbox=gtoolbox)
        algo_tag: str = 'eaSimple'
    else:
        if not ret_args.mu:
            ret_args.mu = ret_args.PopSize
            ret_args.lambda_ = ret_args.PopSize * 2
        if not ret_args.lambda_:
            ret_args.lambda_ = ret_args.PopSize * 2
            ret_args.mu = ret_args.PopSize
        taskdict: dict = get_eamupluslambda_args(popsize=ret_args.PopSize,
                                                 mu=ret_args.mu,
                                                 lambda_=ret_args.lambda_,
                                                 cxpb=ret_args.cxpb,
                                                 mutpb=ret_args.mutpb,
                                                 ngen=ret_args.Numgens,
                                                 stats=gstats,
                                                 hof=ghof,
                                                 verbose=ret_args.verbose,
                                                 toolbox=gtoolbox)
        algo_tag: str = 'eaMuPlusLambda'

    data: object= GAData(filepath=ret_args.data, do_norm=ret_args.normalize)

    print(f'Running GA with: {algo_tag}')
    if ret_args.max_tl > 17:
        print(f'Running GP with Extra Long Trees, Watch out for bloat (!)')

    return taskdict, data, algo_tag, ret_args


workdict, g_data, tag, args = parse_args()

g_max_tl: int = args.max_tl


@njit(fastmath=True, nogil=True, parallel=True)
def accuracy_score(ytrue: np.ndarray, yhat: np.ndarray) -> np.float32:
    ysig = (1e-15 + (1 / (1 + np.exp(-yhat)))) > 0.5
    return np.float32(np.sum(ytrue == ysig))


@njit(fastmath=True, nogil=True, parallel=True)
def nbnp_ce(ytrue, yhat):
    ysig = np.where(yhat >= 0, 1 / (1 + np.exp(-yhat)), np.exp(yhat) / (1 + np.exp(yhat)))
    return -1/len(ytrue) * (np.sum(ytrue*np.log(ysig) + (1-ytrue)*np.log(1-ysig)))


if args.loss == 'ce':
    lossfn = nbnp_ce
    # creator classes
    gcreator.create('FitnessMIN', base.Fitness, weights=(-1.0,))
    gcreator.create('Individual', gp.PrimitiveTree, fitness=gcreator.FitnessMIN)
else:
    lossfn = accuracy_score
    # creator classes
    gcreator.create('FitnessMAX', base.Fitness, weights=(1.0,))
    gcreator.create('Individual', gp.PrimitiveTree, fitness=gcreator.FitnessMAX)


def getpset(pset_data: object = g_data):
    '''
    function for generating primitives for gp.PrimitiveTree
    :param pset_data: object, GAData, for renaming of args in accordance with training data features.
    :return: gp.PrimitiveSet, set of primitives to consider for future symbolic operations in GA.
    '''
    p_set = gp.PrimitiveSetTyped('MAIN', tuple(repeat(float, g_data.numfeatures)), float)

    # boolean ops
    p_set.addPrimitive(np.logical_and, (bool, bool), bool)
    p_set.addPrimitive(np.logical_or, (bool, bool), bool)
    p_set.addPrimitive(np.logical_not, (bool,), bool)

    # logical ops
    # custom primitive
    @njit([float32(boolean, float32, float32),
           float64(boolean, float64, float64)],
          nogil=True, parallel=True)
    def ifte(input: bool, out1: float, out2: float) -> float:
        if input:
            return out1
        else:
            return out2

    p_set.addPrimitive(np.less, (float, float), bool)
    p_set.addPrimitive(np.equal, (float, float), bool)
    p_set.addPrimitive(ifte, (float, float), bool)

    # flops
    # custom primitive
    @njit([float32(float32, float32),
           float64(float64, float64)],
          nogil=True, parallel=True)
    def safediv(l: float, r: float) -> float:
        '''
        zero protected division
        :param l: float, int
        :param r: float, int
        :return: float
        '''
        if r == 0.0:
            return 1
        return l / r

    p_set.addPrimitive(np.add, (float, float), float)
    p_set.addPrimitive(np.subtract, (float, float), float)
    p_set.addPrimitive(np.multiply, (float, float), float)
    p_set.addPrimitive(safediv, (float, float), float)
    p_set.addPrimitive(np.negative, (float,), float)
    p_set.addPrimitive(np.tanh, (float,), float)
    p_set.addPrimitive(np.cos, (float,), float)
    p_set.addPrimitive(np.sin, (float,), float)
    p_set.addPrimitive(np.maximum, (float, float), float)
    p_set.addPrimitive(np.minimum, (float, float), float)

    # terminals
    p_set.addEphemeralConstant(f'rand', lambda: np.random.uniform(-1, 1), float)
    p_set.addTerminal(False, bool)
    p_set.addTerminal(True, bool)
    for cols in pset_data.c_args():
        p_set.renameArguments(**cols)
    return p_set
# pset instance
g_pset: gp.PrimitiveSet = getpset(g_data)

gtoolbox.register('expr', gp.genHalfAndHalf, pset=g_pset, min_=1, max_=g_max_tl)
gtoolbox.register('individual', tools.initIterate, gcreator.Individual, gtoolbox.expr)
gtoolbox.register('population', tools.initRepeat, list, gtoolbox.individual)
gtoolbox.register('compile', gp.compile, pset=g_pset)


@jit(nogil=True)
def evaltree(individual: Callable) -> Tuple[np.float32]:
    '''
    maximizing objective evaluation function
    :param individual: toolbox.individual, single tree with chainable symbols
    :return: tuple[float] maximizing score
    '''
    func = njit(fastmath=True, nogil=True, parallel=True)(gtoolbox.compile(expr=individual, pset=g_pset))
    funcmapped: np.ndarray = jmap(func, g_data.X_train)
    retval = lossfn(g_data.Y_train, funcmapped)
    return retval,


gtoolbox.register('evaluate', evaltree)
gtoolbox.register('select', tools.selDoubleTournament, fitness_size=6, parsimony_size=1.6, fitness_first=True)
gtoolbox.register('mate', gp.cxOnePoint)
gtoolbox.register('expr_mut', gp.genFull, min_=0, max_=g_max_tl)
gtoolbox.register('mutate', gp.mutUniform, expr=gtoolbox.expr_mut, pset=g_pset)

gtoolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=g_max_tl))
gtoolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=g_max_tl))
gtoolbox.decorate('mate', gp.staticLimit(key=len, max_value=121*g_max_tl))
gtoolbox.decorate('mutate', gp.staticLimit(key=len, max_value=121*g_max_tl))

ghistory = tools.History()
gtoolbox.decorate("mate", ghistory.decorator)
gtoolbox.decorate("mutate", ghistory.decorator)

# initiate population
print(f'resume: {args.resume}, seedfile status: {args.seedfile}')
if args.resume:
    if args.seedfile is not None:
        try:
            seedfile: dict = pickle.load(args.seedfile)
        except FileNotFoundError as fnf:
            print(fnf)
        else:
            workdict['population'] = seedfile['all_pops']
            print('Successfully loaded populations from seedfile!')
else:
    workdict['population'] = gtoolbox.population(n=workdict['population'])


def plot_records(logbook: dict) -> np.ndarray:
    '''
    plotting function for GA evolution
    :param logbook: dict, dictionary of statistics, hierarchically grouped by keywords
    :return: ndarray, 3-channel RGB array from plot
    '''
    gens = logbook.select('gen')
    fit_maxs = logbook.chapters['fitness'].select('max')
    fit_avgs = logbook.chapters['fitness'].select('avg')
    size_avgs = logbook.chapters['length'].select('avg')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    line1 = ax.plot(gens, fit_maxs, 'b-', label='Maximum Fitness')
    line2 = ax.plot(gens, fit_avgs, 'b-', label='Average Fitness', linestyle=':')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness', color='b')
    for tl in ax.get_yticklabels():
        tl.set_color('b')

    ax2 = ax.twinx()
    line3 = ax2.plot(gens, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="center right")
    plt.title('Evolution Statistics', fontsize='xx-large')
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def genealogy_plot(history: tools.support.History, toolbox: base.Toolbox) -> np.ndarray:
    '''
    plotting function for genealogical history of the GA run.
    :param history: dict, dict with history
    :param toolbox: gp.toolbox, for using the appropriate eval
    :return: ndarray, 3-Channel RGB array from plot
    '''
    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()  # Make the graph top-down

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    try:
        colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    except:
        # catch all for failures
        colors = [i for i in range(history.genealogy_index)]
    positions = graphviz_layout(graph, prog="dot")
    networkx.draw(graph, positions, node_color=colors,
                  with_labels=True,
                  font_size1=10,
                  alpha=0.75, ax=ax)

    plt.title('Evolution: Genealogy Tree', fontsize='xx-large')
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


# saving the population state in the same file as the other non-dependent
# parameters like the final function/lambda and plots will lead to a
# deserialization failure / pickling problem unless the entire
# toolbox and creator classes are re-instantiated and made available
# in the scope, so we just save the population states into a separate file.
def append_savefile(filename: str, toappend: str='_saved_state') -> str:
    '''
    creates name for the population states file
    :param filename: name of the main save file from the args.ofile parameter
    :param toappend: the string to append as a suffix onto the ofile string
    :return: the final string to be used for population states file.
    '''
    sep_pos: int = filename.rfind('.')
    return filename[:sep_pos]+toappend+filename[sep_pos:]


def main():
    # pool
    gc.collect()
    pool = mp.ProcessingPool(nodes=8)

    # actual work
    workdict['toolbox'].register('map', pool.map)
    ghistory.update(workdict['population'])
    gc.collect()
    if tag == 'eaSimple':
        pop, logs = algorithms.eaSimple(**workdict)
    elif tag == 'eaMuPlusLambda':
        pop, logs = algorithms.eaMuPlusLambda(**workdict)
    pool.close()
    pool.join()

    plot_image: np.ndarray = plot_records(logbook=logs)
    plot_genealogy: np.ndarray = genealogy_plot(history=ghistory, toolbox=workdict['toolbox'])
    tosave = {'func': gtoolbox.compile(expr=workdict['halloffame'][0]),
              'plot': plot_image,
              'history': plot_genealogy}
    to_state = {'all_pops': pop,
                'history': ghistory}
    print(str(workdict['halloffame'][0]))
    dill.dump(tosave, args.ofile, protocol=-1)
    args.ofile.close()
    with open(append_savefile(args.ofile.name), 'wb') as state_file:
        dill.dump(to_state, state_file, protocol=-1)
        state_file.close()
    print(f'file written as dict object to {args.ofile.name}')

    if args.test:
        with open(args.ofile.name, 'rb') as ifile:
            refunc: dict = pickle.load(ifile)
            ifile.close()
        if isinstance(refunc, dict) and isinstance(refunc['func'], LambdaType):
            print(f'File correctly reloaded, testing for output..')
            temp = refunc["func"](*np.random.randn(g_data.numfeatures))
            temp = f"OK, output: {temp:.2f} is a float" if isinstance(temp, float) \
                else f"Error!, Lambda produced output of type {str(type(temp))}"
            print(f'{temp}')


if __name__ == '__main__':
    main()
