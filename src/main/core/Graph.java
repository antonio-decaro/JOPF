package core;

import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import stream.Loader;
import stream.Parser;
import utils.Constants;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.logging.Logger;

/**
 * A Graph class is used as a collection of Nodes and the basic structure to work with OPF.
 * @author De Caro Antonio
 * */
public class Graph implements Serializable {

    private static final Logger logger = Logger.getLogger(Graph.class.getName());

    /**
     * Class constructor
     * @param xArray array of features.
     * @param yArray array of labels.
     * */
    public Graph(INDArray xArray, INDArray yArray) {
        this.nodes = new ArrayList<>();
        this.orderedNodes = new ArrayList<>();

        // check if xArray is properly loaded
        if (xArray != null) {
            // check if yArray is properly loaded
            if (yArray == null)
                // if not, creates an empty array
                yArray = Nd4j.ones(xArray.shape()[0]);
            // build the graph
            this.build(xArray, yArray);
        }
        // else info this error
        else
            logger.severe("graph has not been properly created");
    }

    /**
     * Class constructor.
     * Construct the graph reading the xArray and yArray from a file.
     * @param filename the filename of the dataset. Format: .json, .csv, .txt.
     * */
    public Graph(@NotNull String filename) {
        // create the loader
        Loader loader = new Loader();
        Parser parser;

        // read the file
        if (filename.endsWith("csv"))
            parser = new Parser(loader.loadCSV(filename));
        else if (filename.endsWith("txt"))
            parser = new Parser(loader.loadText(filename));
        else if (filename.endsWith("json"))
            parser = new Parser(loader.loadJSON(filename));
        else
            throw new IllegalArgumentException("File extension not recognized. It should be `.csv`, `.json` or `.txt`");

        // build the graph
        build(parser.getX(), parser.getY());
    }

    /**
     * Destroy the arcs present in the graph.
     * */
    public void destroyArcs() {
        // for every possible node
        for (Node node : nodes) {
            // reset the number of adjacent nodes
            node.setNPlateaus(0);

            // reset the list of adjacent nodes
            node.getAdjacency().clear();
        }
    }

    /**
     * Marks a node and its whole path as relevant.
     * @param i An identifier of the node to start the marking.
     * */
    public void markNodes(int i) {
        // while the node still has a predecessor
        while (nodes.get(i).getPred() != Constants.NIL) {
            // mark current node as RELEVANT
            nodes.get(i).setRelevant(Constants.RELEVANT);

            // update the index with its predecessor
            i = nodes.get(i).getPred();
        }

        // mark the first node as relevant
        nodes.get(i).setRelevant(Constants.RELEVANT);
    }

    /**
     * Resets the graph predecessors and arcs.
     * */
    public void reset(int i) {
        // for every node
        for (Node node : nodes) {
            // reset its predecessor
            node.setPred(Constants.NIL);
            // reset whether its relevant or not
            node.setRelevant(Constants.IRRELEVANT);
        }
        // destroy all arcs
        this.destroyArcs();
    }

    /**
     * @return list of nodes
     * */
    public ArrayList<Node> getNodes() {
        return nodes;
    }

    /**
     * @return list of indexes of ordered nodes
     * */
    public ArrayList<Integer> getOrderedNodes() {
        return orderedNodes;
    }

    /**
     * @return whether the graph is trained or not
     * */
    public boolean isTrained() {
        return trained;
    }

    /**
     * Set if the graph is trained
     * */
    public void setTrained(boolean trained) {
        this.trained = trained;
    }

    /**
     * @return the number of features
     * */
    public int getFeatures() {
        return nFeatures;
    }

    /**
     * This method serves as the object building process.
     * One can define several commands here that does not necessarily needs to be on its initialization.
     *
     * @param x features array.
     * @param y labels array.
     * */
    private void build(INDArray x, INDArray y) {
        // iterate for every possible  in the x array
        for (int i = 0; i < x.shape()[0]; i++) {
            INDArray feature = x.getRow(i);
            int label = y.getInt(i);
            Node node = new Node(i, label, feature);
            nodes.add(node);
        }

        // calculates the number of features
        this.nFeatures = (int) nodes.get(0).getFeatures().shape()[0];
    }

    // list of nodes
    private ArrayList<Node> nodes;

    // list of indexes of ordered nodes
    private ArrayList<Integer> orderedNodes;

    // whether the graph is trained or not
    private boolean trained;

    // the number of features
    private int nFeatures;
}
