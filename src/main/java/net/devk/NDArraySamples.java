package net.devk;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class NDArraySamples {
    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            var x = manager.arange(12);
            x = x.reshape(3, 4);
            System.out.println(x);
            System.out.println(manager.create(new Shape(3, 4)));
            // tensor
            System.out.println(manager.ones(new Shape(2, 3, 4)));
            // Each of its elements is randomly sampled from a standard Gaussian (normal) distribution with a mean of  0
            //  and a standard deviation of  1
            System.out.println(manager.randomNormal(0f, 1f, new Shape(3, 4), DataType.FLOAT32));
            // same as above
            System.out.println(manager.randomNormal(new Shape(3, 4)));
            System.out.println(manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4)));
        }
    }
}
