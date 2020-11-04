package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class gives a representation of an array, in which multiple threads can access.
 *
 * @param <T> the type parameter
 * @author De Caro Antonio
 */
public class DistributedArray<T> {

    /**
     * Class constructor
     * @param array array of elements.
     * */
    public DistributedArray(T[] array, int threads) {
        this.array = array;
        this.threads = threads;
        this.heap = new Heap(threads, Heap.Policy.MAX);
        Arrays.fill(heap.getCost(), 0);
        for (int i = 0; i < threads; i++)
            heap.insert(i);
        this.house = new HashMap<>();
        updateSlices(); // populate each thread slice
    }

    /**
     * Get array.
     *
     * @return the array
     */
    public T[] getArray() {
        return array;
    }

    /**
     * Sets array.
     *
     * @param array the array to set
     */
    public void setArray(T[] array) {
        this.array = array;
    }

    /**
     * Update an element of the array.
     * @param idx index of the element to update.
     * @param element new element's value.
     * @throws IndexOutOfBoundsException if the index is less then 0 or grater then the length of the array.
     */
    public synchronized void update(int idx, T element) throws IndexOutOfBoundsException {
        if (idx < 0 || idx >= array.length)
            throw new IndexOutOfBoundsException();
        this.array[idx] = element;
    }

    /**
     * Get a specified element of the array.
     * @param idx the index of the element.
     * @return the element at i-th position.
     * @throws IndexOutOfBoundsException if the index is less then 0 or grater then the length of the array.
     * */
    public T get(int idx) throws IndexOutOfBoundsException {
        if (idx < 0 || idx >= array.length)
            throw new IndexOutOfBoundsException();
        return array[idx];
    }

    /**
     * Remove an element from the distributed array.
     * @param idx index of element to remove.
     * @throws IndexOutOfBoundsException if the input index is out of bounds of the array.
     * */
    public void remove(int idx) {
        if (idx < 0 || idx >= array.length) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }

        int currentThread = house.get(idx);
        house.remove(idx);

        int maxThread = heap.getFirst();
        int currentVal = (int) heap.getCost()[currentThread] - 1;

        if (currentVal < heap.getCost()[maxThread] - 1) {
            int toTransfer = slices[maxThread].get(slices[maxThread].size() - 1);
            slices[currentThread].add(toTransfer);
            slices[maxThread].remove((Integer) toTransfer);
            house.put(toTransfer, currentThread);
            heap.update(maxThread, heap.getCost()[maxThread] - 1);
        } else {
            heap.update(currentThread, currentVal); // O(log_n)
        }

        slices[currentThread].remove((Integer) idx);
    }

    /**
     * Gets threads number.
     *
     * @return the threads number.
     */
    public int getThreads() {
        return threads;
    }

    /**
     * Checks if an element is available.
     * @param idx index of element to check.
     * @return True if the element is available or if no Filter function is defined, False if the element is not available.
     * */
    public boolean isAvailable(int idx) {
        if (filter == null)
            return true;
        return filter.filter(idx);
    }

    /**
     * Update the slice of each threads, according to the filter function.
     * This method should be called only when more then one element are been modified in once.
     * */
    @SuppressWarnings("unchecked")
    public synchronized void updateSlices() {
        slices = new ArrayList[threads];
        for (int i = 0; i < slices.length; i++)
            slices[i] = new ArrayList<>();

        for (int i = 0, j = 0; i < array.length; i++) {
            if (!isAvailable(i))
                continue;

            slices[j].add(i);
            heap.getCost()[j] += 1;
            house.put(i, j);
            j = (j + 1) % threads;
        }
    }

    /**
     * In concurrency, the threads should be able to work on a slice of the array.
     * This methods is used to get a slice for a specified thread.
     * @param thread the threads identifier.
     * @return a slice for that thread.
     * @throws IndexOutOfBoundsException if the index is less then 0 or grater then threads number.
     * */
    public ArrayList<Integer> getSlice(int thread) {
        return slices[thread];
    }

    /**
     * Gets filter.
     *
     * @return the filter
     */
    public Filter getFilter() {
        return filter;
    }

    /**
     * Sets filter.
     *
     * @param filter the filter
     */
    public void setFilter(Filter filter) {
        this.filter = filter;
    }

    /**
     * This interface is used to filter the elements of an array.
     * */
    public interface Filter {

        /**
         * @param element index the element's index to check.
         * @return return True if the element is available, False otherwise.
         * */
        boolean filter(int element);
    }

    private Heap heap;
    private HashMap<Integer, Integer> house;
    private ArrayList<Integer>[] slices;
    private Filter filter;
    private int threads;
    private T[] array;
}
