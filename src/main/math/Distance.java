package math;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * This is an interface that gives an abstraction of the distance between two nodes.
 * @author De Caro Antonio
 * */
public interface Distance extends Serializable {

    /**
     * Calculate the distance between two points.
     * @param x INDArray of x coordinate.
     * @param y INDArray of y coordinate.
     * @return the distance between x and y.
     * */
    double calculate(INDArray x, INDArray y);
}
