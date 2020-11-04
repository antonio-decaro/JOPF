package core;

import math.Distance;
import math.DistancesImplementor;
import org.nd4j.linalg.api.ndarray.INDArray;
import stream.Loader;
import utils.exceptions.ValueError;

import java.io.*;
import java.util.logging.Logger;

/**
 * A basic class to define all common OPF-related methods.
 * @author De Caro Antonio
 * */
public abstract class OPF implements Serializable, Cloneable {
    private static final Logger logger = Logger.getLogger(OPF.class.getName());

    /**
     * Read OPF state from a file.
     * @param fname filename in which read the OPF state
     */
    public static OPF load(String fname) throws IOException, ClassNotFoundException {
        // creates the input stream
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fname))) {
            // read object from file
            Object obj = ois.readObject();
            // if the object is properly read, cast and return
            if (obj instanceof OPF) {
                return (OPF) obj;
            }
            // else returns null
            return null;
        }
    }

    /**
     * @see OPF
     * */
    public OPF() {
        this(DistancesImplementor.euclideanDistance);
    }

    /**
     * Class constructor
     * @param distance the distance function to use.
     * */
    public OPF(Distance distance) {
        logger.info("Creating class: OPF.");

        // set the distance function
        this.distance = distance;
    }

    /**
     * Save OPF state into a file.
     * @param fname filename of the file in which store OPF state.
     * */
    public void save(String fname) throws IOException {
        // creates the output stream
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fname))) {
            // save the object into the file
            oos.writeObject(this);
        }
    }

    /**
     * Reads precomputed distances from a file.
     * @param filename name of the pre computed distances file.
     * @throws ValueError if cannot properly load distances from file.
     * */
    public void readDistancesFromFile(String filename) throws ValueError {
        logger.info("Reading distances from file");

        // create loader
        Loader loader = new Loader();

        // check if extension is .csv
        if (filename.endsWith("csv"))
            // if yes, call the method that actually loads csv
            preComputedDistances = loader.loadCSV(filename);

        // check i extension is .txt
        else if (filename.endsWith("txt"))
            // if yes, call the method that actually loads txt
            preComputedDistances = loader.loadText(filename);

        // if extension is not recognized
        else
            throw new IllegalArgumentException("File extension not recognized, It should be `.csv` or `.txt`");

        // check if distances have been properly loaded
        if (preComputedDistances == null)
            throw new ValueError("Pre-computed distances could not been properly loaded");
    }

    /**
     * Fits data in the classifier.
     * @param xTrain Array of features.
     * @param yTrain Array of labels.
     * */
    public abstract void fit(INDArray xTrain, INDArray yTrain);

    /**
     * Predicts new data using the pre-trained classifier.
     * @param X Array of features.
     * @return A list of predictions for each record of the data.
     * */
    public abstract INDArray predict(INDArray X);

    /**
     * Gets subgraph.
     *
     * @return the subgraph
     */
    public Graph getGraph() {
        return graph;
    }

    /**
     * Sets graph.
     *
     * @param graph the graph
     */
    public void setGraph(Graph graph) {
        this.graph = graph;
    }

    /**
     * Is distances precomputed boolean.
     *
     * @return the boolean
     */
    public boolean isDistancesPrecomputed() {
        return preComputedDistances != null;
    }


    /**
     * Sets pre computed distances.
     *
     * @param preComputedDistances the pre computed distances; can be null to make distances not precomputed.
     */
    public void setPreComputedDistances(INDArray preComputedDistances) {
        this.preComputedDistances = preComputedDistances;
    }

    /**
     * Gets distance.
     *
     * @return the distance
     */
    public Distance getDistance() {
        return distance;
    }

    /**
     * Sets distance.
     *
     * @param distance the distance
     */
    public void setDistance(Distance distance) {
        this.distance = distance;
    }

    protected OPF clone() throws CloneNotSupportedException {
        return (OPF) super.clone();
    }

    private static final long serialVersionUID = 1L;

    protected Distance distance;
    protected Graph graph;
    protected INDArray preComputedDistances;
}
