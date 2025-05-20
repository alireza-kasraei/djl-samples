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
        System.out.printf("x (independent variable), generated with random normal distribution for %d examples " +
                "and %d features.%n", numExamples, weights.length);
        NDArray w = manager.create(weights);
        System.out.println("w: " + w);
        System.out.println("x: " + x);
        //apply on a linear regression
        NDArray y = x.dot(w).add(bias);
        System.out.println("y (dependent variable) is calculated based on y= w*x + bias ");
        System.out.println("y: " + y);
        // generate some noise
        NDArray noise = manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32);
        System.out.println("noise to be added to the y: " + noise);
        y = y.add(noise);
        System.out.println("after adding noise, y: " + y);
        return new DataPoints(x, y);
    }

}
