package math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * General purpose class for OPF implementation.
 * @author De Caro Antonio
 * */
public class General {

    private static final Logger logger = Logger.getLogger(General.class.getName());

    /**
     * Calculates the accuracy between true and predicted labels using OPF-style measure.
     * @param labels INDArray holding the true labels.
     * @param predict INDArray holding the predicted labels.
     * @return The OPF accuracy measure between 0 and 1.
     * */
    public static double opfAccuracy(INDArray labels, INDArray predict) {
        // calculating the number of classes
        Number tmp = labels.maxNumber();

        // generating the vector of labels and predict
        int[] labelsVector = labels.toIntVector();
        int[] predictVector = predict.toIntVector();

        int nClasses = 0;
        for (int value : labelsVector) {
            if (nClasses < value)
                nClasses = value;
        }

        // creating ab empty error matrix
        INDArray errors = Nd4j.zeros(nClasses, 2);

        // gathering the amount of labels per class
        Map<Number, Integer> countsMap = unique(labels);
        INDArray counts = Nd4j.zeros(countsMap.size());
        int k = 0;
        for (Number number : countsMap.keySet()) {
            counts.putScalar(k, countsMap.get(number));
            k++;
        }

        // for every label and prediction
        for (int i = 0; i < labelsVector.length; i++) {
            int label = labelsVector[i];
            int pred = predictVector[i];
            // if actual label is different from prediction
            if (label != pred) {
                // increments the corresponding cell from the error matrix
                errors.putScalar(new int[]{pred - 1, 0}, errors.getInt(pred - 1, 0) + 1);

                // increments the corresponding cell from the error matrix
                errors.putScalar(new int[]{label - 1, 1}, errors.getInt(label - 1, 1) + 1);
            }
        }

        // calculating the float value of the true label errors
        errors.putColumn(1, errors.getColumn(1).div(counts));

        // calculating the float value of the predicted label errors
        errors.putColumn(0, errors.getColumn(0).div(counts.rsub(counts.sumNumber())));

        // calculating the sum of errors per class
        errors = errors.sum(1);

        // calculates the OPF accuracy and return it
        double sum = errors.sumNumber().doubleValue();
        return 1.0 - (sum / (2 * nClasses));
    }

    /**
     * Generates a vector that not contains elements repeated.
     * @param array INDArray of elements.
     * @return a map containing all elements as a key, and as a value the amount of times that element repeats himself.
     * */
    public static Map<Number, Integer> unique(INDArray array) {
        Map<Number, Integer> counts = new HashMap<>();

        for (int i = 0; i < array.length(); i++) {
            Number val = array.getNumber(i);
            counts.merge(val, 1, Integer::sum);
        }

        return counts;
    }

    /**
     * Computes distances of a given dataset.
     * @param data features data
     * @param distance distance functions
     * @return the distances
     * */
    public static INDArray precomputeDistances(INDArray data, Distance distance) {
        logger.info("Pre computing distances ...");
        int len = data.rows();

        INDArray distances = Nd4j.zeros(len, len);

        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                if (i == j)
                    continue;
                distances.putScalar(new int[]{i, j}, distance.calculate(data.getRow(i), data.getRow(j)));
            }
        }

        logger.info("Distances precomputed.");
        return distances;
    }

    /**
     * Computes distances of a given dataset with parallelization.
     * @param data features data
     * @param distance distance functions
     * @param threads number of threads
     * @return the distances
     * */
    public static INDArray precomputeDistances(INDArray data, Distance distance, int threads) {

        if (threads <= 0) {
            throw new IllegalArgumentException("The number of threads has to be greater or equals 1");
        }

        else if (threads == 1) {
            return General.precomputeDistances(data, distance);
        }

        int len = data.rows();

        INDArray distances = Nd4j.zeros(len, len);

        class Worker implements Runnable {
            Worker(int len, ArrayList<Integer> slice, INDArray distances) {
                this.len = len;
                this.slice = slice;
                this.distances = distances;
            }

            @Override
            public void run() {
                for (int i : slice) {
                    for (int j = 0; j < len; j++) {
                        if (i == j)
                            continue;
                        distances.putScalar(new int[]{i, j}, distance.calculate(data.getRow(i), data.getRow(j)));
                    }
                }
            }

            private final int len;
            private final ArrayList<Integer> slice;
            private final INDArray distances;
        }

        ArrayList<Integer>[] slices = new ArrayList[threads];
        for (int i = 0, j = 0; i < len; i++) {
            if (slices[j] == null)
                slices[j] = new ArrayList<>();
            slices[j].add(i);
            j = (j + 1) % threads;
        }

        logger.info("Pre computing distances ...");

        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(threads);

        for (int i = 0; i < threads; i++) {
            executor.execute(new Worker(len, slices[i], distances));
        }
        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            logger.severe(e.getLocalizedMessage());
        }

        logger.info("Distances precomputed.");
        return distances;
    }

    // TODO purity function
}
