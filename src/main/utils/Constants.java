package utils;

/**
 * This class defines constants used by the module.
 * @author De Caro Antonio
 * */
public class Constants {

    // A constant value used to avoid division by zero, zero logarithms
    // and any possible mathematical error
    public static double EPSILON = 1e-10;
    
    // When the costs are initialized, their value are defined as
    // the maximum float value possible
    public static float FLOAT_MAX = Float.MAX_VALUE;
    
    // Defining color constants for the Heap structure
    // Note that these constants should not be modified
    public static final int WHITE = 0;
    public static final int GRAY = 1;
    public static final int BLACK = 2;
    
    // Defining constant to identify whether a node in
    // the subgraph has a predecessor or not
    public static final int NIL = -1;
    
    // Defining constant to identify whether a node is
    // a prototype or not
    public static final int STANDARD = 0;
    public static final int PROTOTYPE = 1;
    
    // Defining constant to identify whether a node is
    // relevant or not
    public static final int IRRELEVANT = 0;
    public static final int RELEVANT = 1;
    
    // Defining constant to reflect the maximum arc weight
    // used to calculate the distance measures
    public static final int MAX_ARC_WEIGHT = 100000;
    
    // Defining constant to reflect the maximum density
    // used to calculate in unsupervised approaches
    public static final int MAX_DENSITY = 1000;
}
