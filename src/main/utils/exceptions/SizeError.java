package utils.exceptions;

/**
 * A SizeError class for logging errors related to wrong length or size of variables.
 * @author De Caro Antonio
 * */
public class SizeError extends RuntimeException {
    public SizeError(String msg) {
        super(msg);
    }
    public SizeError() {
        super();
    }
}
