'''
* Implementation of symbolic classifier using genetic algorithms based off
  of the DEAP Library.
    ** Multiple Objectives Version. [abs_accuracy + cross_entropy]
    ** Available deap algorithms are eaSimple and eaMuPlusLambda.
    ** Parallelized through Pathos-ProcessPool(with Dill) and Numba.
    ** Uses recommended StaticLimits on heights(=17) and NSGA2/SPEA2 selection
       for bloat control.
* CLI-based with argparse, saves best result with fitness and genealogy plots to file.
    ** Can also optionally resume training from the terminal-populations of
       a previously saved *_state* file via the *resume* and *seedfile* parameter.
'''
import sys
sys.path.append('../')
from data_utils.gadata import GAData
import gc
import pickle
import argparse
import operator
from types import LambdaType, FunctionType
from typing import Tuple, Callable
import dill
import numpy as np
from numba import jit, njit, float32, float64
from deap import base, creator, tools, gp, algorithms
import pathos as pt
import matplotlib.pyplot as plt
import networkx
from networkx.drawing.nx_agraph import graphviz_layout

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


# additional primitive
@njit([float32(float32, float32),
       float64(float64, float64)])
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


# creator classes
gcreator.create('FitnessMULTI', base.Fitness, weights=(1.0, -1.0,))
gcreator.create('Individual', gp.PrimitiveTree, fitness=gcreator.FitnessMULTI)
# best logger
ghof: tools.support.HallOfFame = tools.HallOfFame(1)
# fitness logger
g_fit0: tools.support.Statistics = tools.Statistics(lambda ind: ind.fitness.values[0])
g_fit1: tools.support.Statistics = tools.Statistics(lambda ind: ind.fitness.values[1])
g_len: tools.support.Statistics = tools.Statistics(len)
g_height: tools.support.Statistics = tools.Statistics(operator.attrgetter('height'))
gstats: tools.support.MultiStatistics = tools.MultiStatistics(accuracy=g_fit0,
                                                              cross_entropy=g_fit1,
                                                              length=g_len,
                                                              height=g_height)

rmean = lambda x: np.mean(x).round(2)
rstd = lambda x: np.std(x).round(2)
rmin = lambda x: np.min(x).round(2)
rmax = lambda x: np.max(x).round(2)

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
        if x > 17 or x < 1:
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

    parser.add_argument('PopSize', help='Starting Population Size, discarded if resuming.\n Default is 50.', type=int, default=50)

    parser.add_argument('Numgens', help='Number of Generations.\n Default is 10.', type=int, default=10)

    parser.add_argument('--cxpb',
                        help='The probability that an offspring is produced by crossover.\n Default is 0.25.',
                        type=float, default=0.25)

    parser.add_argument('--mutpb',
                        help='The probability that an offspring is produced by mutation.\n Default is 0.5.',
                        type=float, default=0.5)

    parser.add_argument('--mu',
                        help='The number of individuals to select for the next generation,\
                         does not apply to eaSimple.\n Default is equal to the Population Size Argument.',
                        type=int, default=False)

    parser.add_argument('--lambda_',
                        help='The number of children to produce at each generation,\
                         does not apply to eaSimple.\
                         \n Default is twice the Population Size Argument.',
                        type=int, default=False)

    parser.add_argument('--max_tl',
                        help='The maximum height of the tree to construct for gp.PrimitiveTree.\
                             \n Default is 11, Maximum is 17.',
                        type=type_max_tl,
                        default=11)

    parser.add_argument('--algorithm',
                        type=str,
                        help='Choice of Evolutionary Algorithm, options are "simple" for eaSimple and\
                            \n"mupluslambda" for eaMuPlusLambda.\n Default is "simple".',
                        default='simple',
                        choices=['simple', 'mupluslambda'])

    parser.add_argument('--selector',
                        type=str,
                        help='Choice of Multi-Objective Selection Algorithm, Options are:\
                         "nsga" for NSGA-II and "spea" for SPEA-II.\n Default is "nsga"/NSGA-II.',
                        default='nsga',
                        choices=['nsga', 'spea'])

    parser.add_argument('--verbose',
                        help='output verbosity.\n Default is True.',
                        action='store_true')

    parser.add_argument('--test',
                        help='reload and test the generated function on a random input.\
                        \n Default is True.',
                        default=True,
                        action='store_true')

    parser.add_argument('-i', '--data',
                        action='store',
                        type=str, dest='data',
                        default=featuresfile,
                        help=f'Input Data Dictionary compatible with GAData()\
                        \n Default is {featuresfile}.')

    parser.add_argument('--normalize',
                        type=bool, default=True,
                        help='Normalize the training data.\
                        \n Default is True.')

    parser.add_argument('--resume',
                        action='store_true', default=False,
                        help='Resume training from a previous run,\
                        Will overwrite the starting population from\
                        saved populations of an existing savefile,\
                        provided by the seedfile argument.\
                        \n Default is False.')

    parser.add_argument('-sf', '--seedfile',
                        action='store', type=argparse.FileType('rb'),
                        dest='seedfile', default=None,
                        help='Resume training from a previous run,\
                        Will overwrite the starting population from\
                        saved populations of an existing savefile,\
                        provided by the seedfile argument.\
                        \n Default is False.')

    parser.add_argument('-o', '--ofile',
                        action='store',
                        type=argparse.FileType('wb'), dest='ofile',
                        default='./../../saves/evolved_multi_genetic_function.npy',
                        help='Wrap the final function to a dictionary as\
                        {"func":func:object} and save to a file in path.\
                        \n Default is ./../../saves/evolved_multi_genetic_function.npy.')

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
    max_tl: int = ret_args.max_tl

    print(f'Running GA with: {algo_tag}')
    return taskdict, data, max_tl, algo_tag, ret_args


workdict, g_data, g_max_tl, tag, args = parse_args()


def getpset(pset_data: object = g_data):
    '''
    function for generating primitives for gp.PrimitiveTree
    :param pset_data: object, GAData, for renaming of args in accordance with training data features.
    :return: gp.PrimitiveSet, set of primitives to consider for future symbolic operations in GA.
    '''
    p_set = gp.PrimitiveSet('MAIN', arity=pset_data.numfeatures)
    p_set.addPrimitive(np.add, arity=2)
    p_set.addPrimitive(np.subtract, arity=2)
    p_set.addPrimitive(np.multiply, arity=2)
    p_set.addPrimitive(safediv, arity=2)
    p_set.addPrimitive(np.negative, arity=1)
    p_set.addPrimitive(np.tanh, arity=1)
    p_set.addPrimitive(np.cos, arity=1)
    p_set.addPrimitive(np.sin, arity=1)
    p_set.addPrimitive(np.maximum, arity=2)
    p_set.addPrimitive(np.minimum, arity=2)
    p_set.addEphemeralConstant(f'rand', lambda: np.random.uniform(-1, 1))
    for cols in pset_data.c_args():
        p_set.renameArguments(**cols)
    return p_set
# pset instance
g_pset: gp.PrimitiveSet = getpset(g_data)

gtoolbox.register('expr', gp.genHalfAndHalf, pset=g_pset, min_=1, max_=g_max_tl)
gtoolbox.register('individual', tools.initIterate, gcreator.Individual, gtoolbox.expr)
gtoolbox.register('population', tools.initRepeat, list, gtoolbox.individual)
gtoolbox.register('compile', gp.compile, pset=g_pset)


@njit(fastmath=True, nogil=True, parallel=True)
def accuracy_score(ytrue: np.ndarray, yhat: np.ndarray) -> np.float32:
    ysig = (1e-15 + (1 / (1 + np.exp(-yhat)))) > 0.5
    return np.float32(np.sum(ytrue == ysig))


@njit(fastmath=True, nogil=True, parallel=True)
def nbnp_ce(ytrue, yhat):
    ysig = 1e-15 + (1 / (1 + np.exp(-yhat)))
    return -1/len(ytrue) * (np.sum(ytrue*np.log(ysig) + (1-ytrue)*np.log(1-ysig)))


@jit(nogil=True)
def evaltree(individual: Callable) -> Tuple[np.float32, np.float32]:
    '''
    maximizing objective evaluation function
    :param individual: toolbox.individual, single tree with chainable symbols
    :return: tuple[float] maximizing score
    '''
    func = njit(fastmath=True, nogil=True, parallel=True)(gtoolbox.compile(expr=individual, pset=g_pset))
    funcmapped: np.ndarray = jmap(func, g_data.X_train)
    accuracy = accuracy_score(g_data.Y_train, funcmapped)
    loss = nbnp_ce(g_data.Y_train, funcmapped)
    return accuracy, loss,


gtoolbox.register('evaluate', evaltree)
gtoolbox.register('select', tools.selSPEA2 if args.selector == 'spea' else tools.selNSGA2)
gtoolbox.register('mate', gp.cxOnePoint)
gtoolbox.register('expr_mut', gp.genFull, min_=0, max_=17)
gtoolbox.register('mutate', gp.mutUniform, expr=gtoolbox.expr_mut, pset=g_pset)

gtoolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=17))
gtoolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=17))

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


def plot_records(logbook: dict, xdata: object=g_data ) -> np.ndarray:
    '''
    plotting function for GA evolution
    :param logbook: dict, dictionary of statistics, hierarchically grouped by keywords
    :return: ndarray, 3-channel RGB array from plot
    '''
    gens = logbook.select('gen')
    acc_avgs = np.array(logbook.chapters['accuracy'].select('avg')) / xdata.training_size
    ce_avgs = logbook.chapters['cross_entropy'].select('avg')
    size_avgs = logbook.chapters['length'].select('avg')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    line1 = ax.plot(gens, acc_avgs, 'b-', label='Average Fitness 0: ACC')
    line2 = ax.plot(gens, ce_avgs, 'b-', label='Average Fitness 1: BCE', linestyle=':')
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
    except: # catch all for failures
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
    pool = pt.multiprocessing.ProcessingPool(nodes=8)

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
