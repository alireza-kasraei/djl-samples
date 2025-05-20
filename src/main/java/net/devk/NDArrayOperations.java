package net.devk;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

/**
 * DJL’s ndarray is an extension to NumPy’s ndarray with a few killer advantages that make it suitable for deep learning.
 *
 * DJL’s ndarray provides a variety of functionalities including basic mathematics operations, broadcasting, indexing,
 * slicing, memory saving, and conversion to other Python objects.
 */
public class NDArrayOperations {
    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            var x = manager.create(new float[]{1f, 2f, 4f, 8f});
            var y = manager.create(new float[]{2f, 2f, 2f, 2f});
            System.out.println("x= " + x + " y= " + y);
            System.out.println("x+y= " + x.add(y));
            System.out.println("x-y= " + x.sub(y));
            System.out.println("x*y= " + x.mul(y));
            System.out.println("x/y= " + x.div(y));
            System.out.println("x^y= " + x.pow(y));
            System.out.println("e^x= " + x.exp());

            x = manager.arange(12f).reshape(3, 4);
            y = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4));
            System.out.println(x.concat(y)); // default axis = 0
            System.out.println(x.concat(y, 1)); //axis = 1

            System.out.println(x.eq(y));

            // broadcasting mechanism
            var a = manager.arange(3f).reshape(3, 1);
            var b = manager.arange(2f).reshape(1, 2);
            System.out.println("a= " + a);
            System.out.println("b= " + b);
            System.out.println("a+b= " + a.add(b));

            // indexing?
            System.out.println("x[-1] = " + x.get(":-1"));

            // 2 first rows
            x.set(new NDIndex("0:2, :"), 12);

            var original = manager.zeros(new Shape(3, 4));
            var actual = original.addi(x);
            System.out.println(original == actual);

        }
    }
}
