package stream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * Loads a file into a ND4J Array.
 * @author De Caro Antonio
 * */
public class Loader {

    private static final Logger logger = Logger.getLogger(Loader.class.getName());

    /**
     * Load a JSON file into a ND4J Array.
     * @param filename string holding the *.json path.
     * @return an Array containing data.
     * */
    public INDArray loadJSON(String filename) {
        logger.info("Trying to load file: " + filename);
        try (InputStream inputStream = new FileInputStream(filename)) {
            // creating json reader and json object
            JsonReader jsonReader = Json.createReader(inputStream);
            JsonObject jsonObject = jsonReader.readObject();

            logger.info("File loaded.");

            // read data from json object
            JsonArray data = jsonObject.getJsonArray("data");
            List<INDArray> records = new LinkedList<>();
            int recordSize = -1;
            for (int i = 0; i < data.size(); i++) {
                // read whole record
                JsonObject jsonRecord = data.getJsonObject(i);
                // get id
                int id = jsonRecord.getInt("id");
                // get label
                int label = jsonRecord.getInt("label");
                // read features json array
                JsonArray jsonFeatures = jsonRecord.getJsonArray("features");
                // transform the json record in a double array.
                double[] record = new double[jsonFeatures.size() + 2];
                // set the record size if not already set
                if (recordSize < 0)
                    recordSize = record.length;
                // populate the record
                record[0] = id;
                record[1] = label;
                for (int f = 0; f < jsonFeatures.size(); f++)
                    record[f + 2] = jsonFeatures.getJsonNumber(f).doubleValue();
                //create an INDArray record and insert it in the record list
                records.add(Nd4j.create(record));
            }

            // create the array with all records, and
            // return the array
            return Nd4j.create(records, records.size(), recordSize);

        } catch (IOException e) {
            logger.severe(e.getMessage());
        }

        return null;
    }

    /**
     * Load a text file into a ND4J Array.
     * @param filename string holding the *.json path.
     * @return an Array containing data.
     * */
    public INDArray loadText(String filename) {
        logger.info("Loading file: " + filename);

        // initialize the record size
        int recordSize = -1;
        // initialize the list containing the dataset
        ArrayList<INDArray> records = new ArrayList<>();

        // try to load the file
        try (Scanner scanner = new Scanner(new FileInputStream(filename))){
            logger.info("File loaded.");
            // for each line in the file
            while (scanner.hasNextLine()) {
                // build a line scanner
                Scanner recordScanner = new Scanner(scanner.nextLine());

                // initialize an array containing all elements in a line
                ArrayList<String> fields = new ArrayList<>();
                // read all elements of the line
                while (recordScanner.hasNext())
                    fields.add(recordScanner.next());

                // if not set yet, set the record size
                if (recordSize < 0)
                    recordSize = fields.size();

                // parse the field array in a record of double
                double[] record = new double[recordSize];
                for (int i = 0; i < recordSize; i++)
                    record[i] = Double.parseDouble(fields.get(i));

                // insert the record in the records list
                records.add(Nd4j.create(record, recordSize));
            }

            // create the dataset,
            // and return
            return Nd4j.create(records, records.size(), recordSize);

        } catch (FileNotFoundException e) {
            // if can not properly read the file
            logger.severe(e.getMessage());
        }

        return null;
    }

    /**
     * Load a CSV file into a ND4J Array.
     * @param filename string holding the *.json path.
     * @return an Array containing data.
     * */
    public INDArray loadCSV(String filename) {
        logger.info("Loading file CSV: " + filename);

        // initialize the record size
        int recordSize = -1;
        // initialize the list containing the dataset
        ArrayList<INDArray> records = new ArrayList<>();

        // try to load the file
        try (Scanner scanner = new Scanner(new FileInputStream(filename))){
            logger.info("File loaded.");
            // for each line in the file
            while (scanner.hasNextLine()) {
                // build a line scanner
                Scanner recordScanner = new Scanner(scanner.nextLine());
                recordScanner.useDelimiter(Pattern.compile(","));

                // initialize an array containing all elements in a line
                ArrayList<String> fields = new ArrayList<>();
                // read all elements of the line
                while (recordScanner.hasNext())
                    fields.add(recordScanner.next());

                // if not set yet, set the record size
                if (recordSize < 0)
                    recordSize = fields.size();

                // parse the field array in a record of double
                double[] record = new double[recordSize];
                for (int i = 0; i < recordSize; i++)
                    record[i] = Double.parseDouble(fields.get(i));

                // insert the record in the records list
                records.add(Nd4j.create(record, recordSize));
            }

            // create the dataset,
            // and return
            return Nd4j.create(records, records.size(), recordSize);

        } catch (FileNotFoundException e) {
            // if can not properly read the file
            logger.severe(e.getMessage());
        }

        return null;
    }
}
