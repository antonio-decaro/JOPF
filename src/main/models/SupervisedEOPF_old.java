package models;

import core.Graph;
import core.Heap;
import core.Node;
import core.OPF;
import math.Distance;
import math.General;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import utils.Constants;
import utils.exceptions.BuildError;
import utils.exceptions.ValueError;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

/**
 * A SupervisedOPF which implements the supervised version of OPF classifier.
 *
 * @author De Caro Antonio
 * */
public class SupervisedEOPF_old extends OPF {
    private static final Logger logger = Logger.getLogger(SupervisedEOPF_old.class.getName());

    /**
     * @see OPF
     * */
    public SupervisedEOPF_old() {super();}

    /**
     * @see OPF
     * */
    public SupervisedEOPF_old(Distance distance) {
        super(distance);
    }

    @Override
    public void fit(INDArray xTrain, INDArray yTrain) {
        logger.info("Fitting classifier ...");

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

        // creating a minimum heap
        Heap heap = new Heap(graph.getNodes().size(), Heap.Policy.MIN);

        // for each possible node
        for (int i = 0; i < graph.getNodes().size(); i++) {
            Node node = graph.getNodes().get(i);
            // checks if node is prototype
            if (node.getStatus() == Constants.PROTOTYPE) {
                // if yes, it does not have predecessor nodes
                node.setPred(Constants.NIL);

                // its predicted label is the same as its true label
                node.setPredictedLabel(node.getLabel());

                // its cost equals to zero
                heap.getCost()[i] = 0;

                // inserts the node into the heap
                heap.insert(i);
            }

            // if node is not a prototype
            else {
                heap.getCost()[i] = Constants.FLOAT_MAX;
            }
        }

        // while the heap is not empty
        while (!heap.isEmpty()) {
            // removes a node
            int p = heap.remove();
            // gather the associated node
            Node pNode = graph.getNodes().get(p);

            // appends its index to the oredered list
            graph.getOrderedNodes().add(p);

            // gather its cost
            pNode.setCost(heap.getCost()[p]);

            // for every possible node
            for (int q = 0; q < graph.getNodes().size(); q++) {
                // if we are dealing with different node
                if (p == q)
                    continue;

                // Retrive the associated node
                Node qNode = graph.getNodes().get(q);

                // if `p` node cost is smaller than `q` node cost
                if (heap.getCost()[p] < heap.getCost()[q]) {
                    double weight;
                    // checks if we are using a pre-computed distance
                    if (this.isDistancesPrecomputed())
                        weight = preComputedDistances.getDouble(pNode.getIndex(), qNode.getIndex());
                    else
                        // calls the corresponding distance function
                        weight = distance.calculate(pNode.getFeatures(), qNode.getFeatures());

                    // the current cost will be the maximum cost between the node's and its weight (arc)
                    double currentCost = Math.max(heap.getCost()[p], weight);

                    // if the current cost is smaller than `q` node's cost
                    if (currentCost < heap.getCost()[q]) {
                        // `q` node has `p` as its predecessor
                        qNode.setPred(p);

                        // and its predicted label is the same as `p`
                        qNode.setPredictedLabel(pNode.getPredictedLabel());

                        // updates the heap `q` node and the current cost
                        heap.update(q, currentCost);
                    }
                }
            }
        }

        // the subgraph has been properly trained
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

    @Override
    public INDArray predict(INDArray xVal) {
        // check if there is a subgraph
        if (graph == null)
            throw new BuildError("Subgraph has not been properly created.");

        // check if the subgraph has been trained
        if (!graph.isTrained())
            throw new BuildError("Subgraph has not been properly trained.");

        if (this.isDistancesPrecomputed())
            logger.info("Working on precomputed distances ...");

        logger.info("Predicting data ...");

        // initializing timer
        Instant start = Instant.now();

        // creating a prediction subgraph
        Graph predGraph = new Graph(xVal, null);

        // for every possible node
        for (int i = 0; i < predGraph.getNodes().size(); i++) {
            // initialize the conqueror node
            int conqueror = -1;
            // initializing the `j` counter
            int j = 0;

            // gathers the first node from the ordered list
            int k = graph.getOrderedNodes().get(j);

            // initialize the weight
            double weight;
            // checks if we are using a pre-computed distances
            if (this.isDistancesPrecomputed())
                // gathers the distance from the distance's matrix
                weight = preComputedDistances.getDouble(graph.getNodes().get(k).getIndex(),
                        predGraph.getNodes().get(i).getIndex());
            else
                // calls the corresponding distance function
                weight = distance.calculate(graph.getNodes().get(k).getFeatures(),
                        predGraph.getNodes().get(i).getFeatures());

            // the minimum cost will be the maximum between the `k` node cost and its weight (arc)
            double minCost = Math.max(graph.getNodes().get(k).getCost(), weight);

            // the current label will be `k` node's predicted label
            int currentLabel = graph.getNodes().get(k).getPredictedLabel();

            // while `j` os a possible node and the minimum cost is bigger than the current node's cost
            while (j < graph.getNodes().size() - 1 &&
                    minCost > graph.getNodes().get(graph.getOrderedNodes().get(j + 1)).getCost()) {
                // gathers the next node from the ordered list
                int l = graph.getOrderedNodes().get(j + 1);

                // check if we are using a pre-computed distance
                if (this.isDistancesPrecomputed())
                    weight = preComputedDistances.getDouble(graph.getNodes().get(l).getIndex(),
                            predGraph.getNodes().get(i).getIndex());
                else
                    // calls the corresponding distance function
                    weight = distance.calculate(graph.getNodes().get(l).getFeatures(),
                            predGraph.getNodes().get(i).getFeatures());

                // the temporary minimum cost will be the maximum between `l` node cost and its weight (arc)
                double tempMinCost = Math.max(graph.getNodes().get(l).getCost(), weight);

                // if temporary minimum cost is smaller than the minimum cost
                if (tempMinCost < minCost) {
                    // replaces the minimum cost
                    minCost = tempMinCost;

                    // gathers the identifier of `l` node
                    conqueror = l;

                    // updates the current label as `l` node's predicted label
                    currentLabel = graph.getNodes().get(l).getPredictedLabel();
                }
                // increments the `counter` and makes `k` and `l` equals
                j++;
            }
            // node's `i` predicted label is the same as current label
            predGraph.getNodes().get(i).setPredictedLabel(currentLabel);

            // checks if any node has been conquered
            if (conqueror > -1)
                // marks the conqueror node and its path
                graph.markNodes(conqueror);
        }
        // creating the list of predictions
        int[] pred = new int[predGraph.getNodes().size()];

        // populate the pred list
        for (int i = 0; i < predGraph.getNodes().size(); i++)
            pred[i] = predGraph.getNodes().get(i).getPredictedLabel();

        // ending timer
        Instant end = Instant.now();

        // calculating prediction task time
        logger.info( "Data has been predicted.");
        logger.info( "Prediction time: " + Duration.between(start, end).toMillis() + " millis.");

        // reset the precomputed distances
        this.setPreComputedDistances(null);

        return Nd4j.createFromArray(pred);
    }

    /**
     * Learns the best classifier over a validation set.
     * @param xTrain array of training features
     * @param yTrain array of training labels
     * @param xVal array of validation features
     * @param yVal array of validation labels
     * @param iterations number of iterations, must be grater then 0
     * */
    public void learn(INDArray xTrain, INDArray yTrain, INDArray xVal, INDArray yVal, int iterations) {
        logger.info("Learning the best classifier ...");

        // create a random instance
        Random random = new Random();

        // check if iterations number is valid
        if (iterations <= 0)
            throw new ValueError("The iterations number must be grater than 0.");

        // define max accuracy
        double maxAccuracy = 0;

        // define the previous accuracy
        double prevAccuracy = 0;

        // define the iterations counter
        int t = 0;

        // define the best iteration
        int bestIteration = -1;

        // define the best classifier
        OPF bestOPF = null;

        while (true) {
            logger.info("Running iteration " + (t + 1) + "/" + iterations);

            // fits training data into the classifier
            this.fit(xTrain, yTrain);

            // predicts new data
            INDArray pred = this.predict(xVal);

            // calculating accuracy
            double acc = General.opfAccuracy(yVal, pred);

            // checks if current accuracy is better than the best one
            if (acc > maxAccuracy) {
                // if yes, replace the maximum accuracy
                maxAccuracy = acc;

                // makes a copy of the best OPF classifier
                try {
                    bestOPF = this.clone();
                } catch (CloneNotSupportedException e) {
                    logger.severe(e.getMessage());
                    throw new IllegalStateException(e.getMessage());
                }

                // and saves the iteration number
                bestIteration = t;
            }

            // gathers which samples were missclassified
            ArrayList<Integer> errors = new ArrayList<>();
            for (int i = 0; i < yVal.length(); i++) {
                if (!yVal.getNumber(i).equals(pred.getNumber(i)))
                    errors.add(i);
            }

            // defining the initial number of non-prototypes as 0
            int nonPrototypes = 0;

            // for every possible subgraph's node
            for (Node node : this.graph.getNodes()) {
                // if the node is not a prototype
                if (node.getStatus() != Constants.PROTOTYPE)
                    nonPrototypes++;
            }

            // for every possible error
            for (int i = 0; i < errors.size(); i++) {
                // counter will receive the number of non-prototypes
                int ctr = nonPrototypes;

                // while the counter is bigger than zero
                while (ctr > 0) {
                    // generate a random index
                    int j = random.nextInt((int) xTrain.shape()[0]);

                    // if the node on that particular index is not a prototype
                    if (this.graph.getNodes().get(j).getStatus() != Constants.PROTOTYPE) {
                        // swap the input nodes
                        INDArray tmpRow = xTrain.getRow(j);
                        xTrain.putRow(j, xVal.getRow(i));
                        xVal.putRow(i, tmpRow);

                        // swap the target nodes
                        Number tmpNumber = yTrain.getNumber(j);
                        yTrain.putScalar(j, yVal.getNumber(i).floatValue());
                        yVal.putScalar(i, tmpNumber.floatValue());

                        // decrements the number of non-prototypes
                        nonPrototypes--;
                    } else {
                        // decrements the counter
                        ctr--;
                    }
                }
            }

            // calculating the difference between current accuracy and previous one
            double delta = Math.abs(acc - prevAccuracy);

            // replacing the previous accuracy as current accuracy
            prevAccuracy = acc;

            // increment the counter
            t++;

            logger.info("Accuracy: " + acc + " | Delta: " + delta + " | Maximum Accuracy: " + maxAccuracy);

            // if the difference is smaller then 10e-4 or iterations are finished
            if (Double.compare(delta, 0.0001) < 0 || t == iterations) {
                if (bestOPF != null)
                    this.graph = bestOPF.getGraph();

                logger.info("Best classifier has been learned over iteration " + (bestIteration + 1));

                // breaks the loop
                break;
            }
        }
    }

    /**
     * Prunes a classifier over a validation set.
     * @param xTrain array of training features
     * @param yTrain array of training labels
     * @param xVal array of validation features
     * @param yVal array of validation labels
     * @param iterations number of iterations, must be grater then 0
     * */
    public void prune(INDArray xTrain, INDArray yTrain, INDArray xVal, INDArray yVal, int iterations) {
        logger.info("Pruning classifier ...");

        // fits training data into the classifier
        this.fit(xTrain, yTrain);

        // predicts new data
        this.predict(xVal);

        // gathering initial number of nodes
        float initialNodes = this.graph.getNodes().size();

        // for every possible iteration
        for (int t = 0; t < iterations; t++) {
            logger.info(String.format("Running iterations %d/%d ...", t + 1, iterations));

            // creating temporary lists
            ArrayList<INDArray> xTemp = new ArrayList<>();
            ArrayList<Integer> yTemp = new ArrayList<>();

            // removing irrelevant nodes
            for (int i = 0; i < this.graph.getNodes().size(); i++) {
                Node node = this.graph.getNodes().get(i);
                if (node.getRelevant() != Constants.IRRELEVANT) {
                    xTemp.add(xTrain.getRow(i));
                    yTemp.add(yTrain.getInt(i));
                }
            }

            // copying lists back to original data
            xTrain = Nd4j.create(xTemp, xTemp.size(), xVal.shape()[1]);
            yTrain = Nd4j.create(yTemp);

            // fits training data into the classifier
            this.fit(xTrain, yTrain);

            // predicts new data
            INDArray preds = this.predict(xVal);

            // calculating accuracy
            double accuracy = General.opfAccuracy(yVal, preds);

            logger.info("Current accuracy: " + accuracy);
        }

        // gathering final number of nodes
        float finalNodes = this.graph.getNodes().size();

        // calculating pruning ratio
        double pruneRatio = 1 - finalNodes / initialNodes;

        logger.info("Prune ratio: " + pruneRatio);
    }

    /**
     * Find prototype nodes using the Minimum Spanning Tree (MST) approach.
     * */
    protected void findPrototypes() {
        logger.info("Finding prototypes...");

        // initialize timer
        Instant start = Instant.now();

        // creating a heap of size equals to number of nodes
        Heap heap = new Heap(graph.getNodes().size(), Heap.Policy.MIN);

        // marking first node without any predecessor
        graph.getNodes().get(0).setPred(Constants.NIL);

        // adding first node to the heap
        heap.insert(0);

        // creating a list of prototype nodes
        List<Integer> prototypes = new ArrayList<>();
        // while the heap is not empty

        // defining an iteration counter to track progress
        int iterations = 1;
        final int SECS = 15;

        // initialize progress timer
        Instant progressStart = Instant.now();

        // while the queue is not empty
        while (!heap.isEmpty()) {
            // show progress if SECS are elapsed
            double progress = ((double) iterations++ * 100 / graph.getNodes().size());
            if (Duration.between(progressStart, Instant.now()).toMillis() >= SECS * 1000) {
                progressStart = Instant.now();
                logger.info(String.format("Progress: %.2f%%", progress));
            }

            // remove a node from the heap
            int p = heap.remove();

            // gathers its cost from the heap
            graph.getNodes().get(p).setCost(heap.getCost()[p]);

            // and also its predecessor
            int pred = graph.getNodes().get(p).getPred();

            // if the predecessor is not NIL
            if (pred != Constants.NIL) {
                // checks if the label of current node is the same as its predecessor
                if (graph.getNodes().get(p).getLabel() != graph.getNodes().get(pred).getLabel()) {
                    // if current node is not a prototype
                    if (graph.getNodes().get(p).getStatus() != Constants.PROTOTYPE) {
                        // marks it as a prototype
                        graph.getNodes().get(p).setStatus(Constants.PROTOTYPE);

                        // append current node identifier to the prototype's list
                        prototypes.add(p);
                    }

                    // if predecessor node is not a prototype
                    if (graph.getNodes().get(pred).getStatus() != Constants.PROTOTYPE) {
                        // marks it as a prototype
                        graph.getNodes().get(pred).setStatus(Constants.PROTOTYPE);

                        // append predecessor node identifier to the prototype's list
                        prototypes.add(pred);
                    }
                }
            }

            // for every possible node
            for (int q = 0; q < graph.getNodes().size(); q++) {
                // checks if the color of current node in the heap is not black
                if (heap.getColor()[q] != Constants.BLACK) {
                    // if `p` and `q` identifiers are the same then skip
                    if (p == q)
                        continue;

                    // get distance weight
                    double weight;
                    // if it is supposed to use pre-computed distances
                    if (this.isDistancesPrecomputed()) {
                        // gathers the arc from the distances matrix
                        weight = preComputedDistances.getDouble(graph.getNodes().get(p).getIndex(),
                                graph.getNodes().get(q).getIndex());
                    } else {
                        // calculate the distance
                        weight = distance.calculate(graph.getNodes().get(p).getFeatures(),
                                graph.getNodes().get(q).getFeatures());
                    }
                    // if current arc's cost is smaller the the path's cost
                    if (weight < heap.getCost()[q]) {
                        // marks `q` predecessor node as `p`
                        graph.getNodes().get(q).setPred(p);

                        // updates the arc on the heap
                        heap.update(q, weight);
                    }
                }
            }
        }

        // end the timer
        Instant end = Instant.now();

        // calculating training task time
        Duration trainTime = Duration.between(start, end);

        logger.info("Finding prototypes time: " + trainTime.toMillis() + " milliseconds.");
        logger.info("Prototypes: " + prototypes);
    }
}
