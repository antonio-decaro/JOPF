package core;

import org.nd4j.linalg.api.ndarray.INDArray;
import utils.Constants;
import utils.exceptions.ValueError;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * A Node class is used as the lowest structure level in the OPF workflow.
 * @author De Caro Antonio
 * */
public class Node implements Serializable {
    /**
     * Class constructor.
     * @param idx the node's identifier.
     * @param label the node's label.
     * @param features an INDArray of features.
     * */
    public Node(int idx, int label, INDArray features) {
        this.index = idx;
        this.label = label;
        this.features = features;

        this.adjacency = new ArrayList<>();
        this.status = Constants.STANDARD;
        this.pred = Constants.NIL;
        this.relevant = Constants.IRRELEVANT;
    }

    /**
     * @return node's index
     * */
    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        if (index < 0)
            throw new ValueError("`index` must be >= 0");
        this.index = index;
    }

    /**
     * @return node's label
     * */
    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        if (label < 1)
            throw new ValueError("`label` must be >= 1");
        this.label = label;
    }

    /**
     * @return its possible predicted label
     * */
    public int getPredictedLabel() {
        return predictedLabel;
    }

    public void setPredictedLabel(int predictedLabel) {
        if (predictedLabel < 0)
            throw new ValueError("`predictedLabel` should be >= 0'");
        this.predictedLabel = predictedLabel;
    }

    /**
     * @return its possible cluster label
     * */
    public int getClusterLabel() {
        return clusterLabel;
    }

    public void setClusterLabel(int clusterLabel) {
        if (clusterLabel < 0)
            throw new ValueError("`clusterLabel` should be >= 0'");
        this.clusterLabel = clusterLabel;
    }

    /**
     * @return node's features array
     * */
    public INDArray getFeatures() {
        return features;
    }

    public void setFeatures(INDArray features) {
        this.features = features;
    }

    /**
     * @return cost of the node
     * */
    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }

    /**
     * @return density of the node
     * */
    public double getDensity() {
        return density;
    }

    public void setDensity(double density) {
        this.density = density;
    }

    /**
     * @return amount of adjacent nodes on plateaus
     * */
    public int getNPlateaus() {
        return nPlateaus;
    }

    public void setNPlateaus(int nPlateaus) {
        if (nPlateaus < 0)
            throw new ValueError("`nPlateaus` should be >= 0");
        this.nPlateaus = nPlateaus;
    }

    /**
     * @return list of adjacent nodes
     * */
    public ArrayList<Node> getAdjacency() {
        return adjacency;
    }

    public void setAdjacency(ArrayList<Node> adjacency) {
        this.adjacency = adjacency;
    }

    /**
     * @return the cluster node identifier
     * */
    public int getRoot() {
        return root;
    }

    public void setRoot(int root) {
        if (root < 0)
            throw new ValueError("`root` should be >= 0");
        this.root = root;
    }

    /**
     * @return whether the node is a prototype or not
     * */
    public int getStatus() {
        return status;
    }

    public void setStatus(int status) {
        if (status != Constants.STANDARD && status != Constants.PROTOTYPE)
            throw new ValueError("`status` should be `PROTOTYPE` or `STANDARD`");
        this.status = status;
    }

    /**
     * @return identifier to the predecessor node
     * */
    public int getPred() {
        return pred;
    }

    public void setPred(int pred) {
        if (pred < Constants.NIL)
            throw new ValueError("`pred` should have a value larger than `NIL`, e.g., -1");
        this.pred = pred;
    }

    /**
     * @return whether the node is relevant or not
     * */
    public int getRelevant() {
        return relevant;
    }

    public void setRelevant(int relevant) {
        if (relevant != Constants.RELEVANT && relevant != Constants.IRRELEVANT)
            throw new ValueError("`status` should be `RELEVANT` or `IRRELEVANT`");
        this.relevant = relevant;
    }

    @Override
    public String toString() {
        return "Node{" +
                "index=" + index +
                ", label=" + label +
                ", predictedLabel=" + predictedLabel +
                ", clusterLabel=" + clusterLabel +
                ", features=" + features +
                ", cost=" + cost +
                ", density=" + density +
                ", nPlateaus=" + nPlateaus +
                ", adjacency=" + adjacency +
                ", root=" + root +
                ", status=" + status +
                ", pred=" + pred +
                ", relevant=" + relevant +
                '}';
    }

    // initially, we need to set the node's index
    private int index;

    // we also need to set its label (true label)
    private int label;

    // its possible predicted label
    private int predictedLabel;

    // and finally, its cluster assignment label (if used)
    private int clusterLabel;

    // array of features
    private INDArray features;

    // cost of the node
    private double cost;

    // density of the node
    private double density;

    // amount of adjacent nodes on plateaus
    private int nPlateaus;

    // list of adjacent nodes
    private ArrayList<Node> adjacency;

    // the cluster node identifier
    private int root;

    // whether the node is a prototype or not
    private int status;

    // identifier to the predecessor node
    private int pred;

    // whether the node is relevant or not
    private int relevant;
}
