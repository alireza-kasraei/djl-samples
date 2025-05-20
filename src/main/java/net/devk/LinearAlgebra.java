package net.devk;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

public class LinearAlgebra {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            // A scalar is represented by a NDArray with just one element.
            NDArray x = manager.create(3f);
            NDArray y = manager.create(2f);
            System.out.println(x.add(y));

            // You can think of a vector as simply a list of scalar values
            NDArray v = manager.arange(4f);
            System.out.println(v.get(3));

            NDArray A = manager.arange(20f).reshape(5, 4);
            System.out.println("A= " + A);
            System.out.println("AT (Transposed)= " + A.transpose());

            // symmetric matrix
            NDArray B = manager.create(new float[][]{{1, 2, 3}, {2, 0, 4}, {3, 4, 5}});
            System.out.println(B.eq(B.transpose()));

            /*
            NDArrays will become more important when we start working with images, which arrive as  n-dimensional
            arrays with 3 axes corresponding to the height, width, and a channel axis for stacking the color channels
            (red, green, and blue). For now, we will skip over higher order NDArrays and focus on the basics.
             */
            NDArray X = manager.arange(24f).reshape(2, 3, 4);
            System.out.println("X= tensor 2 * (3*4)" + X);

            NDArray AA = manager.arange(20f).reshape(5, 4);
            NDArray BB = AA.duplicate();

            System.out.println("AA==BB" + AA.eq(BB));

            System.out.println("AA : " + AA);

            System.out.println("AA Shape = " + AA.getShape());

            //Reduction

            System.out.println("AA sum = " + AA.sum());

            System.out.println("AA , to reduce the row dimension (axis 0) = " + AA.sum(new int[]{0}));
            System.out.println("AA , to reduce the row dimension (axis 1) = " + AA.sum(new int[]{1}));
            System.out.println("AA , to reduce the row dimension (axis 0 and 1) same as sum = " + AA.sum(new int[]{0, 1}));

            // same as above for mean function

            System.out.println("AA , to reduce the row dimension (mean)  = " + AA.mean());
            System.out.println("AA , to reduce the row dimension (mean)  = " + A.sum().div(A.size()));
            System.out.println("AA , to reduce the row dimension (mean) (axis 0) = " + AA.mean(new int[]{0}));
            System.out.println("AA , to reduce the row dimension (mean) (axis 0) = " + A.sum(new int[]{0}).div(A.getShape().get(0)));
            System.out.println("AA , to reduce the row dimension (mean) (axis 1) = " + AA.mean(new int[]{1}));
            System.out.println("AA , to reduce the row dimension (mean) (axis 0 and 1) same as mean = " + AA.mean(new int[]{0, 1}));

            // dot product

            NDArray ones = manager.ones(new Shape(4));
            System.out.println("ones: " + ones);


        }
    }

}
