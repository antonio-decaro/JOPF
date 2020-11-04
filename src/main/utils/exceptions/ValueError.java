package utils.exceptions;

/**
 * A ValueError class for logging errors related to wrong value of variables.
 * @author De Caro Antonio
 * */
public class ValueError extends RuntimeException {
    public ValueError(String msg) {
        super(msg);
    }

    public ValueError() {
        super();
    }
}
