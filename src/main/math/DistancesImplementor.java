package math;

import org.nd4j.linalg.api.ndarray.INDArray;
import utils.Constants;

/**
 * This class implements different ways to calculate the distance between two points.
 * @author De Caro Antonio
 * */
public class DistancesImplementor {

    public static final Distance euclideanDistance = (Distance) (x, y) -> {
        float dist = 0.0f;
        for (int i = 0; i < x.length(); i++) {
            float tmp = x.getFloat(i) - y.getFloat(i);
            dist += tmp * tmp;
        }
        return dist;
    };

    /**
     * Log Euclidean Distance
     * */
    public static final Distance logEuclideanDistance = (Distance) (x, y) -> {
        // calculates the squared euclidean distance for each dimension
        return Constants.MAX_ARC_WEIGHT * Math.log(Math.sqrt(euclideanDistance.calculate(x, y) + 1));
    };

    /**
     * Log Squared Euclidean Distance
     * */
    public static final Distance logSquaredEuclideanDistance = (Distance) (x, y) -> {
        // calculates the squared euclidean distance for each dimension
        return Constants.MAX_ARC_WEIGHT * Math.log(euclideanDistance.calculate(x, y) + 1);
    };
}
