package net.devk.regression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class DataPoints {


    private final NDArray x, y;

    public DataPoints(NDArray x, NDArray y) {
        this.x = x;
        this.y = y;
    }

    public NDArray getX() {
        return x;
    }

    public NDArray getY() {
        return y;
    }

    // y = Xw +bias + noise
    public static DataPoints syntheticData(NDManager manager, float[] weights, float bias, int numExamples) {
        NDArray x = manager.randomNormal(new Shape(numExamples, weights.length));
        NDArray w = manager.create(weights);
        NDArray y = x.dot(w).add(bias);
        // generate some noise
        NDArray noise = manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32);
        y = y.add(noise);
        return new DataPoints(x, y);
    }

}
