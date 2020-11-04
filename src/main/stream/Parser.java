package stream;

import math.General;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import utils.exceptions.ValueError;

import java.util.logging.Logger;

/**
 * Parses data in OPF file format that was pre-loaded (.csv, .txt or .json)
 * @author De Caro Antonio
 * */
public class Parser {
    private static final Logger logger = Logger.getLogger(Loader.class.getName());

    /**
     * Class constructor.
     * @param data the array containing all data (id, label, features).
     * */
    public Parser(INDArray data) {
        this.data = data;
        parse();
    }

    /**
     * Gets the data to parse.
     * */
    public INDArray getData() {
        return data;
    }

    /**
     * Set data to parse.
     * */
    public void setData(INDArray data) {
        this.data = data;
        // parse again the data
        parse();
        // set to null the halves
        xHalf = yHalf = null;
    }

    /**
     * Get the features array.
     * */
    public INDArray getX() {
        return xHalf;
    }

    /**
     * Get the labels array.
     * */
    public INDArray getY() {
        return yHalf;
    }

    /**
     * Parse the data array.
     * */
    private void parse() {
        logger.info("Parsing data ...");

        // tries to parse the dataframe
        try {
            // get features part
            xHalf = data.get(NDArrayIndex.all(), NDArrayIndex.interval(2, data.shape()[1]));
            // get labels part
            yHalf = data.getColumn(1);
            yHalf = yHalf.castTo(DataType.INT8);

            // get the number of different elements in label set
            int count = General.unique(yHalf).size();

            // check if there are at least two different labels
            if (count < 2) {
                System.out.println(yHalf);
                throw new ValueError("Parsed data should have at least two distinct labels");
            }

            // check if the elements are in order
            if (count != yHalf.maxNumber().intValue())
                throw new ValueError("Parsed data should have sequential labels, e.g., 1, 2, ..., n");
            logger.info("Data parsed.");

        } catch (Exception e) {
            e.printStackTrace();
            logger.severe("Bad format.");
        }
    }

    private INDArray data, xHalf, yHalf;
}
