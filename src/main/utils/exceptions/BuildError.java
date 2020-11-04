package utils.exceptions;

/**
 * A BuildError class for logging errors related to classes not being built.
 * @author De Caro Antonio
 * */
public class BuildError extends RuntimeException{
    public BuildError(String msg) {
        super(msg);
    }
    public BuildError() {
        super();
    }
}
