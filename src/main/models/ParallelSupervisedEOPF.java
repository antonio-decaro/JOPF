package models;

import core.DistributedArray;
import core.Node;
import core.OPF;
import core.Graph;
import math.Distance;
import org.nd4j.linalg.api.ndarray.INDArray;
import utils.Constants;
import utils.exceptions.BuildError;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.logging.Logger;

/**
 * This class implements the supervised opf classifier, that uses multithreading to parallelize operations.
 *
 * @author De Caro Antonio
 */
public class ParallelSupervisedEOPF extends SupervisedEOPF {
    private static final Logger logger = Logger.getLogger(ParallelSupervisedEOPF.class.getName());

    public ParallelSupervisedEOPF() {super();}

    public ParallelSupervisedEOPF(Distance distance) {super(distance);}

    @Override
    public void fit(INDArray xTrain, INDArray yTrain) {
        int cores = Runtime.getRuntime().availableProcessors() / 2;

        this.fit(xTrain, yTrain, cores);
    }

    /**
     * Uses multithreading to fit data in the classifier.
     *
     * @param xTrain  Array of features.
     * @param yTrain  Array of labels.
     * @param threads number of threads to use.
     */
    public void fit(INDArray xTrain, INDArray yTrain, int threads) {
        logger.info(String.format("Fitting classifier (%d threads)...", threads));

        // creating the subgraph
        graph = new Graph(xTrain, yTrain);

        // checks if it is supposed to use pre-computed distance
        if (this.isDistancesPrecomputed()) {
            logger.info("Working with precomputed distances ...");

            // checks if its size is the same as the subgraph's amount of nodes
            if (preComputedDistances.shape()[0] != graph.getNodes().size() ||
                    preComputedDistances.shape()[1] != graph.getNodes().size()) {
                throw new BuildError("Pre-computed distance matrix should have the size of `n_nodes x n_nodes`");
            }
        }

        // finding prototypes
        findPrototypes();

        // initialize the timer
        Instant start = Instant.now();

        // initialize the node s
        int s = Constants.NIL;

        // initialize the cost vector
        Double[] costs = new Double[graph.getNodes().size()];
        boolean[] available = new boolean[graph.getNodes().size()];

        // initialize each node
        for (int i = 0; i < graph.getNodes().size(); i++) {
            Node node = graph.getNodes().get(i);
            // if is a prototype
            if (node.getStatus() == Constants.PROTOTYPE) {
                // set predicted label as its label
                node.setPredictedLabel(node.getLabel());
                // set cost to 0
                costs[i] = 0d;
                // set predecessor to NIL
                node.setPred(Constants.NIL);
                // set s as the first prototype
                if (s == Constants.NIL)
                    s = node.getIndex();
            } else {
                // set its cost as FLOAT_MAX
                costs[i] = (double) Constants.FLOAT_MAX;
            }
            available[i] = true;
        }

        // get ExecutorService from Executors utility class, thread pool size is 10
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        // create a list to hold the Future object associated with Callable
        List<Future<Integer>> futuresList = new ArrayList<>();

        // defining the concurrent array
        DistributedArray<Double> distributedArray = new DistributedArray<>(costs, threads);
        // defining filter function
        distributedArray.setFilter(idx -> available[idx]);


        try {
            // while s is not NIL
            while (s != Constants.NIL) {
                available[s] = false;
                // insert s in the ordered set
                graph.getOrderedNodes().add(s);

                // gather its cost
                graph.getNodes().get(s).setCost(costs[s]);

                // updating threads slices
                //distributedArray.updateSlices();
                distributedArray.remove(s);

                // for each thread we want to use
                for (int i = 0; i < threads; i++) {
                    // create callable instance
                    Callable<Integer> worker = new Worker(i, s, distributedArray, available, this);
                    // create the future object
                    Future<Integer> future = executor.submit(worker);
                    // append the future to the futures list
                    futuresList.add(future);
                }
                // now get the threads result
                int min = Constants.NIL;
                for (Future<Integer> future : futuresList) {
                    int curr = future.get();
                    if (curr == Constants.NIL)
                        continue;
                    if (min == Constants.NIL || costs[curr] < costs[min])
                        min = curr;
                }

                // now set s as the lowest value found
                s = min;
                // clear futures list
                futuresList.clear();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            logger.severe(e.getMessage());
            return;
        } finally {
            // shutdown the executor
            executor.shutdown();
        }

        // set the subgraph trained
        graph.setTrained(true);

        // reset the precomputed distances
        this.setPreComputedDistances(null);

        // end the timer
        Instant end = Instant.now();

        // calculating training task time
        Duration trainTime = Duration.between(start, end);

        logger.info("Classifier has been fitted.");
        logger.info("Training time: " + trainTime.toMillis() + " milliseconds.");
    }

    /**
     * This private class implements a thread behaviour
     * */
    private class Worker implements Callable<Integer> {
        public Worker(int id, int s, DistributedArray<Double> distributedArray, boolean[] available, OPF instance) {
            this.id = id;
            this.s = s;
            this.distributedArray = distributedArray;
            this.available = available;
            this.instance = instance;
        }

        @Override
        public Integer call() {
            int p = Constants.NIL;
            for (int q : distributedArray.getSlice(id)) {
                if (q == s)
                    continue;
                if (distributedArray.get(q) > distributedArray.get(s)) {
                    double weight;
                    if (instance.isDistancesPrecomputed())
                        weight = preComputedDistances.getDouble(s, q);
                    else
                        weight = distance.calculate(graph.getNodes().get(s).getFeatures(),
                                graph.getNodes().get(q).getFeatures());

                    double currentCost = Math.max(distributedArray.get(s), weight);
                    if (currentCost < distributedArray.get(q)) {
                        // `q` node has `p` as its predecessor
                        graph.getNodes().get(q).setPred(s);

                        // and its predicted label is the same as `p`
                        graph.getNodes().get(q).setPredictedLabel(graph.getNodes().get(s).getPredictedLabel());

                        // updates the heap `q` node and the current cost
                        distributedArray.update(q, currentCost);
                    }
                }
                if ((p == Constants.NIL || distributedArray.get(q) < distributedArray.get(p)) && available[q]) {
                    p = q;
                }
            }
            return p;
        }

        int id, s;
        private final DistributedArray<Double> distributedArray;
        boolean[] available;
        private final OPF instance;
    }
}
