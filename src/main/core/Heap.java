package core;

import utils.Constants;
import utils.exceptions.SizeError;

import java.util.Arrays;

/**
 * Standard implementation of Heap structure with a fixed size.
 * @author De Caro Antonio
 * */
public class Heap {

    /**
     * Class constructor.
     * @param size fixed size of the heap.
     * @param policy whether the heap pop out the minimum element or the maximum.
     * */
    public Heap(int size, Policy policy) {
        // set heap size;
        this.size = size;

        // cost of each element
        cost = new double[size];
        Arrays.fill(cost, Constants.FLOAT_MAX);

        // color of each element
        color = new int[size];
        Arrays.fill(color, Constants.WHITE);

        // list of nodes value
        p = new int[size];
        Arrays.fill(p, -1);

        // list of nodes position
        pos = new int[size];
        Arrays.fill(pos, -1);

        // last element identifier
        this.last = -1;

        // set policy
        this.policy = policy;
    }

    /**
     * Gets the size of the heap.
     * */
    public int getSize() {
        return size;
    }

    /**
     * Gets the node costs.
     * */
    public double[] getCost() {
        return cost;
    }


    /**
     * Get color int [ ].
     *
     * @return the int [ ]
     */
    public int[] getColor() {
        return color;
    }

    /**
     * Sets the node costs.
     * */
    public void setCost(double[] cost) {
        if (cost.length != size)
            throw new SizeError("The size of the array must be equals to: " + size);
        this.cost = cost;
    }

    /**
     * Check if the heap is empty or not
     * @return true if the heap is empty, false otherwise
     */
    public boolean isEmpty() {
       // returns true if the last position is equals to -1, false otherwise
       return last == -1;
    }

    /**
     * Check if the heap is full or not
     * @return true if the heap is full, false otherwise
     */
    public boolean isFull() {
       // returns true if the last position is equals to -1, false otherwise
       return last == size - 1;
    }

    /**
     * Inserts a new node into the heap.
     * @param p node's value to be inserted.
     * @return boolean indicating whether insertion wa performed correctly
     */
    public boolean insert(int p) {
        // insert only if is not full
        if (!isFull()) {
            // increases the last node's counter
            ++last;

            // adds the new node to the heap
            this.p[last] = p;

            // marks it as gray
            color[p] = Constants.GRAY;

            // marks its positioning
            pos[p] = last;

            // go up in the heap
            goUp(last);
            return true;
        }
        return false;
    }

    /**
     * Removes the first node from the heap.
     * @return the removed node value, or NIL if the heap is empty.
     * */
    public int remove() {
        // if the heap is empty returns NIL
        if (isEmpty())
            return Constants.NIL;

        // gathers the node's value
        int p = this.p[0];

        // marks it as not positioned
        pos[p] = -1;

        // change its color to black
        color[p] = Constants.BLACK;

        // gathers the new position of the first node
        this.p[0] = this.p[last];

        // marks it as positioned
        pos[this.p[0]] = 0;

        // remove its value
        this.p[last] = -1;

        // decrease the last counter
        --last;

        // go down in the heap
        goDown(0);

        // returns the removing node's value
        return p;
    }

    /**
     * Update a node with new value.
     * @param p node's position.
     * @param cost node's cost.
     * */
    public void update(int p, double cost) {
        // applies the new cost
        this.cost[p] = cost;

        // if the node's color is white
        if (color[p] == Constants.WHITE)
            // insert a new node
            insert(p);
        // if the node's color is grey
        else if (color[p] == Constants.GRAY)
            // go up in the heap to desired position
            goUp(pos[p]);
    }

    /**
     * Get the first item to remove, without actually remove it.
     * @return the minimum (or maximum) element in the heap.
     * */
    public int getFirst() {
        return this.p[0];
    }

    /**
     * Gathers the position of the node's dad.
     * @param i node's position.
     * @return the position of node's dad.
     */
    private int dad(int i) {
        return (i - 1) / 2;
    }

    /**
     * Gathers the position of the left son.
     * @param i node's position.
     * @return the position of left son.
     */
    private int leftSon(int i) {
        return (2 * i + 1);
    }

    /**
     * Gathers the position of the right son.
     * @param i node's position.
     * @return the position of the right son.
     */
    private int rightSon(int i) {
        return (2 * i + 2);
    }

    /**
     * Goes up in the heap.
     * @param i position to be achieved.
     * */
    private void goUp(int i) {
        // gathers the dad position
        int j = dad(i);

        // while the heap exists and the cost of post-node is bigger (or smaller) than current node
        while (i > 0 && checkPolicyCost(cost[p[i]], cost[p[j]])) {
            // swap the positions
            int temp = p[i];
            p[i] = p[j];
            p[j] = temp;

            // applies node's i value to the positioning list
            pos[p[i]] = i;

            // applies node's j value to the positioning list
            pos[p[j]] = j;

            // makes both indexes equal
            i = j;

            // gathers the new dad's position
            j = dad(i);
        }
    }

    /**
     * Goes down in the heap.
     * @param i position to be achieved.
     * */
    private void goDown(int i) {
        // gathers the left son's position
        int left = leftSon(i);

        // gathers the right son's position
        int right = rightSon(i);

        // equals the value of `j` and `i` counters
        int j = i;

        // checks if left node is not the last and its cost is smaller than previous
        if (left <= last && checkPolicyCost(cost[p[left]], cost[p[i]]))
            // apply `j` counter as the left node
            j = left;

        // checks if right node is not the last and its cost is smaller than previous
        if (right <= last && checkPolicyCost(cost[p[right]], cost[p[j]]))
            // apply `j` counter as the right node
            j = right;

        // checks if `j` is not equal to `i`
        if (j != i) {
            // swap the positions
            int temp = p[i];
            p[i] = p[j];
            p[j] = temp;

            // marks the new position in `i` and in `j`
            pos[p[i]] = i;
            pos[p[j]] = j;

            // go down in the heap
            goDown(j);
        }
    }

    /**
     * Check the cost regarding the policy.
     * @param first current node cost.
     * @param second other node cost.
     * @return True if the policy is min and costFirstNode < costSecondNode, or if policy is max and
     * costFirstNode > costSecondNode. False otherwise
     * */
    private boolean checkPolicyCost(double first, double second) {
        // check if the cost of post-node is bigger than current node
        if (policy == Policy.MIN)
            return first < second;
        // check if the cost of post-node is smaller than current node
        else
            return first > second;
    }

    /**
     * Heap policy enumeration class.
     * */
    public enum Policy {
        MIN,
        MAX
    }

    private double[] cost;
    private int[] color;
    private int[] p;
    private int[] pos;
    private int size, last;
    private Policy policy;
}
