package tools;

import com.google.gson.*;
import core.Graph;
import core.OPF;
import math.Distance;
import math.DistancesImplementor;
import models.SupervisedEOPF;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Type;
import java.util.Scanner;

/**
 * This class uses library Gson to convert a classifier to JSON object and vice-versa.
 * */
public class JSONConverter {

    static {
        gson = new GsonBuilder()
                .registerTypeAdapter(INDArray.class, new INDArrayAdapter())
                .setPrettyPrinting().create();
    }

    /**
     * Gets the JSON object from a given instance.
     * @param instance the OPF instance.
     * @return a string in JSON format that describes the classifier.
     * */
    public static String toJSON(OPF instance) {
        JsonObject jo = new JsonObject();
        jo.add(GRAPH_PROPERTY, gson.toJsonTree(instance.getGraph()));
        jo.addProperty(INSTANCE_PROPERTY, instance.getClass().getSimpleName());
        jo.addProperty(DISTANCE_PROPERTY, getDistance(instance.getDistance()));
        return gson.toJson(jo);
    }

    /**
     * Save OPF state into a file.
     * @param instance the OPF instance.
     * @param filename the filename of the file in wich save the instance.
     * */
    public static void toJSONFile(OPF instance, String filename) throws IOException {
        try (FileWriter fw = new FileWriter(filename)) {
            fw.write(JSONConverter.toJSON(instance));
        }
    }

    /**
     * Load an OPF instance from a JSON element.
     * @param jsonString a JSON string representing an OPF instantce.
     * @return the OPF instance.
     * */
    public static OPF fromJSON(String jsonString) {
        JsonObject jo = new Gson().fromJson(jsonString, JsonObject.class);

        Distance distance = getDistance(jo.get(DISTANCE_PROPERTY).getAsString());

        try {
            String packageName = SupervisedEOPF.class.getPackage().getName();
            Class<?> instanceClass = Class.forName(packageName + "." + jo.get(INSTANCE_PROPERTY).getAsString());
            OPF opf = (OPF) instanceClass.getConstructor(Distance.class).newInstance(distance);

            opf.setGraph(gson.fromJson(jo.get(GRAPH_PROPERTY),  Graph.class));
            return opf;

        } catch (ClassNotFoundException | NoSuchMethodException | InstantiationException | IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
            return null;
        }

        //return gson.fromJson(jsonString, SupervisedEOPF.class);
    }

    /**
     * Load an OPF instance from a JSON element.
     * @param fname a JSON file containing an OPF instantce.
     * @return the OPF instance.
     * */
    public static OPF fromJSONFile(String fname) {
        try (Scanner scanner = new Scanner(new FileInputStream(fname))) {
            scanner.useDelimiter("\\Z");
            String jsonString = scanner.next();
            return fromJSON(jsonString);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }


    private static class INDArrayAdapter implements JsonSerializer<INDArray>, JsonDeserializer<INDArray> {
        @Override
        public JsonElement serialize(INDArray indArray, Type type, JsonSerializationContext jctx) {
            Gson gson = new Gson();

            return gson.toJsonTree(indArray.toDoubleVector());
        }

        @Override
        public INDArray deserialize(JsonElement je, Type type, JsonDeserializationContext jctx) throws JsonParseException {
            Gson gson = new Gson();
            double[] arr = gson.fromJson(je, double[].class);
            return Nd4j.create(arr, arr.length);
        }
    }

    private static String getDistance(Distance distance) {
        if (distance == DistancesImplementor.euclideanDistance)
            return EUCLIDEAN_DISTANCE;
        else if (distance == DistancesImplementor.logEuclideanDistance)
            return LOG_EUCLIDEAN_DISTANCE;
        else if (distance == DistancesImplementor.logSquaredEuclideanDistance)
            return LOG_SQUARED_EUCLIDEAN_DISTANCE;
        else
            return "";
    }

    private static Distance getDistance(String name) {
        switch (name) {
            case LOG_EUCLIDEAN_DISTANCE:
                return DistancesImplementor.logEuclideanDistance;
            case LOG_SQUARED_EUCLIDEAN_DISTANCE:
                return DistancesImplementor.logSquaredEuclideanDistance;
            case EUCLIDEAN_DISTANCE:
            default:
                return DistancesImplementor.euclideanDistance;
        }
    }

    private static final Gson gson;

    private static final String GRAPH_PROPERTY = "graph";
    private static final String INSTANCE_PROPERTY = "instance";
    private static final String DISTANCE_PROPERTY = "distance";

    private static final String EUCLIDEAN_DISTANCE = "euclideanDistance";
    private static final String LOG_EUCLIDEAN_DISTANCE = "logEuclideanDistance";
    private static final String LOG_SQUARED_EUCLIDEAN_DISTANCE = "logSquaredEuclideanDistance";
}
