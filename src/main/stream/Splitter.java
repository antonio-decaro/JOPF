package stream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import utils.exceptions.SizeError;

import java.util.ArrayList;
import java.util.Collections;
import java.util.logging.Logger;

/**
 * Splits data in different sets.
 *
 * @author De Caro Antonio
 */
public class Splitter {

    private static final Logger logger = Logger.getLogger(Splitter.class.getName());

    /**
     * Class constructor.
     * @param features array of features.
     * @param labels array of labels.
     * @param percentage the percentage of the elements in the training set.
     * */
    public Splitter(INDArray features, INDArray labels, float percentage) {
        this.x = features;
        this.y = labels;
        this.percentage = percentage;
        split();
    }

    /**
     * Class constructor
     * @see Splitter
     * */
    public Splitter(INDArray features, INDArray labels) {
        this(features, labels, 0.5f);
    }

    /**
     * Gets features.
     *
     * @return the features
     */
    public INDArray getX() {
        return x;
    }

    /**
     * Gets labels.
     *
     * @return the labels
     */
    public INDArray getY() {
        return y;
    }

    /**
     * Gets features train.
     *
     * @return the features train
     */
    public INDArray getX1() {
        return x1;
    }

    /**
     * Gets features test.
     *
     * @return the features test
     */
    public INDArray getX2() {
        return x2;
    }

    /**
     * Gets labels train.
     *
     * @return the labels train
     */
    public INDArray getY1() {
        return y1;
    }

    /**
     * Gets labels test.
     *
     * @return the labels test
     */
    public INDArray getY2() {
        return y2;
    }

    /**
     * This method splits features and labels in two sets, one for testing, and the other for training.
     * */
    private void split() {
        logger.info("Splitting data ...");

        // check if `Labels` and `Features` have the same size.
        if (y.shape()[0] != x.shape()[0])
            throw new SizeError("`X` and `Y` should have the same amount of samples");

        // calculating when sets should be halted
        int halt = (int) (x.shape()[0] * percentage);

        // generating random indexes
        ArrayList<Integer> idx = new ArrayList<>();
        for (int i = 0; i < x.shape()[0]; i++)
            idx.add(i);
        Collections.shuffle(idx);


        ArrayList<INDArray> featuresLst = new ArrayList<>();
        ArrayList<INDArray> labelLst = new ArrayList<>();

        for (int i = 0; i < halt; i++) {
            featuresLst.add(x.getRow(idx.get(i)));
            labelLst.add(y.getScalar(idx.get(i)));
        }
        x1 = Nd4j.create(featuresLst, halt, x.shape()[1]);
        y1 = Nd4j.create(labelLst, halt);

        for (int i = halt; i < x.shape()[0]; i++) {
            featuresLst.add(x.getRow(idx.get(i)));
            labelLst.add(y.getScalar(idx.get(i)));
        }
        x2 = Nd4j.create(featuresLst, x.shape()[0] - halt, x.shape()[1]);
        y2 = Nd4j.create(labelLst, x.shape()[0] - halt);

        logger.info(String.format("Data split: X1=(%d,%d) | Y1=(%d) | X2=(%d,%d) | Y2=(%d)",
                x1.shape()[0], x2.shape()[1], y1.length(),
                x2.shape()[0], x2.shape()[1], y2.length()));
    }

    private float percentage;
    private INDArray x, y, x1, x2, y1, y2;
}
